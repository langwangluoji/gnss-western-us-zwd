"""Locked monthly weather blocks and resumable input-only compact field cache."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import time
from pathlib import Path
import numpy as np
import torch
from grid_pair_data import GridStudy,local_runtime,row_digest,verify_row
from local_context_features import LocalFeatureBuilder,VertexQuery
from local_innovation_components import aggregate_innovation
from local_innovation_pilot import Pilot
from local_coverage_provenance import dependency_snapshot,lock_dependencies
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.data import PackedStore
from gnss_field.data.context import cycles
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.training.baselines import _idw
from gnss_field.training.engine import PauseRequested

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'results/20261122_local_coverage_expansion_v01'
PRIOR=PROJECT/'results/20261122_local_innovation_gaussian_v02'


def coverage_schedule(hours,training_allowed,outer_starts):
    hours=np.sort(np.asarray(hours,'i8'));allowed=np.sort(np.asarray(training_allowed,'i8'))
    if not len(hours) or len(np.unique(hours))!=len(hours):raise ValueError('Nonunique calendar')
    year=str(hours[0].astype('datetime64[h]').astype('datetime64[Y]'))
    calendar=set(hours.tolist());blocked=set()
    for start in outer_starts:blocked.update(range(int(start)-18,int(start)+168+18))
    starts=[]
    for month in range(1,13):
        begin=int(np.datetime64(f'{year}-{month:02d}-01','h').astype('i8'))
        end=int((np.datetime64(f'{year}-{month:02d}','M')+1).astype('datetime64[h]').astype('i8'))
        candidates=[s for s in range(begin,end-167,24)
                    if all(h in calendar and h not in blocked for h in range(s,s+168))]
        if not candidates:raise ValueError(f'No complete safe weather week in month {month}')
        starts.append(candidates[0])
    keep=np.ones(len(allowed),bool)
    for start in starts:keep&=(allowed<start-18)|(allowed>=start+168+18)
    for start in outer_starts:keep&=(allowed<int(start)-18)|(allowed>=int(start)+168+18)
    pool=allowed[keep];months=pool.astype('datetime64[h]').astype('datetime64[M]').astype(int)%12+1
    training=[];counts=[]
    for month in range(1,13):
        available=pool[months==month];counts.append(len(available))
        if len(available)<48:raise ValueError(f'Month {month} has fewer than48 training hours')
        training.extend(available[np.rint(np.linspace(0,len(available)-1,48)).astype(int)])
    return dict(train_hours=np.sort(training),inner_hours=np.ravel(np.array(starts)[:,None]+np.array([12,54,96,138])[None]),
        inner_week_starts=np.asarray(starts),remaining_training_hours_by_month=counts)


def compact67(builder,frames,m0,cycle):
    value=builder.build(frames,m0,np.zeros(32,'f4'),cycle)[:,:67].copy()
    density=[5,11,17,23,29,35,41,47,49,51]
    value[:,density]=np.log1p(value[:,density])/np.log1p(174.)
    return value


def progress(stage,**values):
    write_json(ROOT/'status.json',dict(stage=stage,updated=time.time(),**values))
    print(stage,values,flush=True)


def prepare():
    ROOT.mkdir(exist_ok=True,parents=True)
    runtime=local_runtime();bundle=Path(runtime['bundle']);prior_lock=read_json(PRIOR/'experiment.lock.json')
    lock_dependencies(ROOT,dependency_snapshot(PROJECT,runtime))
    manifest=read_json(bundle/'manifest.json')
    if read_json(Path(runtime['data_root'])/'manifest.json')['fingerprint']!=manifest['frozen_contract']['foundation']['data']:
        raise ValueError('Live packed release differs from frozen B bundle')
    if sha256(bundle/'data.npz')!=manifest['data_sha256']:raise ValueError('Frozen B bundle changed')
    with np.load(bundle/'data.npz') as archive:hours=np.array(archive['hours']);station_ids=np.array(archive['station_ids'])
    original=read_json(PROJECT/'results/20261122_generation_postrun_research_v01/downloaded/data.lock.json')
    audit=read_json(PROJECT/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json')
    outer_starts=[int(np.datetime64(s,'h').astype('i8')) for s in audit['week_starts']]
    schedule=coverage_schedule(hours,original['train_hours'],outer_starts)
    selected=np.unique(np.r_[schedule['train_hours'],schedule['inner_hours'],prior_lock['outer_hours']])
    splits=[]
    for old in prior_lock['splits']:
        splits.append(dict(fit=old['fit'],inner=old['inner'],outer=old['outer'],
            train_hours=schedule['train_hours'],inner_hours=schedule['inner_hours'],week_starts=schedule['inner_week_starts']))
    contract=dict(source_data=manifest['data_sha256'],prior_split=sha256(PRIOR/'experiment.lock.json'),
        frozen_M0=sha256(runtime['m0']),parameters=sha256(runtime['parameters']),
        preparation=sha256(__file__),schedule=schedule,splits=splits,hours=selected,
        outer_hours=prior_lock['outer_hours'],original_training_pool=original['train_hours'],
        input_schema='legacy C67 log-density + separate current-base innovation2; head69 has no absoluteXY',
        exposure='B2018 incremental development; historical base/prior and former trials have B exposure',
        inner_scoring='four fixed hours in each of12complete excluded weeks; not full-week hourly scores',
        C2019=False,D2020=False,server_jobs=False)
    immutable_json(ROOT/'experiment.lock.json',contract)
    if (ROOT/'data_complete.json').exists():
        if sha256(ROOT/'data.npz')!=read_json(ROOT/'data_complete.json')['sha256']:raise ValueError('Expanded data changed')
        return
    if sha256(PRIOR/'data.npz')!=read_json(PRIOR/'data_complete.json')['sha256']:raise ValueError('Old compact cache changed')
    old=dict(np.load(PRIOR/'data.npz'));np.testing.assert_array_equal(old['station_ids'],station_ids)
    old_m0_root=PROJECT/'results/20261120_local_context_three_arm_v01/M0_cache'
    if sha256(old_m0_root/'mean.npy')!=read_json(old_m0_root/'complete.json')['sha256']:raise ValueError('Old M0 cache changed')
    old_m0=np.load(old_m0_root/'mean.npy',mmap_mode='r')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    n=len(selected);nodes=old['global_nodes'];v=len(nodes)
    maps={}
    for name,shape,dtype in [('features',(n,v,67),'f4'),('innovation',(n,v,2),'f4'),('mean',(n,len(station_ids)),'f4'),('M0',(n,144,96),'f4'),('row_hashes',(n,32),'u1')]:
        path=cache/f'{name}.npy'
        if path.exists():
            maps[name]=np.load(path,mmap_mode='r+')
            if maps[name].shape!=shape or maps[name].dtype!=np.dtype(dtype):raise ValueError('Expanded cache shape/dtype changed')
        else:maps[name]=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
    done=np.array(read_json(cache/'cursor.json')['done'],bool) if (cache/'cursor.json').exists() else np.zeros(n,bool)
    store=PackedStore(runtime['data_root'],purpose='context');grid=store.grid;aid=store.ids('A')
    aq,_=safe_query_geometry(store,aid);allnodes=np.arange(np.prod(store.shape))
    xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    maker=LocalFeatureBuilder(xy,store.xy[aid],store.stations.height.to_numpy()[aid],grid['height'],grid['region'],nodes,store.stats,aq,store.query_valid[aid])
    qA=VertexQuery(store.shape,aq,np.zeros(len(aid)),allnodes);acorners=np.unique(qA.indices.numpy());aa=np.searchsorted(acorners,qA.indices.numpy())
    settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')
    scratch=object.__new__(GridStudy);scratch.runtime=runtime;scratch.store=store;scratch.device='cuda';scratch.net=None
    qi,qw=old['query_indices'],old['query_weights'];oldlookup={int(h):i for i,h in enumerate(old['hours'])}
    started=time.monotonic();reused=generated=0
    def commit():
        for arr in maps.values():arr.flush()
        write_json(cache/'cursor.json',dict(done=done))
    for i,hour in enumerate(selected):
        if done[i]:
            verify_row([maps[k][i] for k in ('features','innovation','mean','M0')],maps['row_hashes'][i]);continue
        if (ROOT/'PAUSE').exists():
            commit();raise PauseRequested('Expanded preparation paused before next hour')
        if int(hour) in oldlookup:
            j=oldlookup[int(hour)]
            for k in ('features','innovation','mean'):maps[k][i]=old[k][j]
            maps['M0'][i]=old_m0[j];reused+=1
        else:
            m0=scratch.frozen_m0(int(hour));generated+=1
            frames=np.array([store.context_values([int(hour)-o])[0] for o in (0,6,12,18)])
            f=compact67(maker,frames,m0,cycles([hour])[0])
            base=m0.ravel()[nodes]+settings['baseline_weight']*(f[:,59]*settings['grid_scale']-m0.ravel()[nodes])
            current=frames[0];ok=np.isfinite(current).all(1)
            idw=_idw(store.xy[aid][ok],current[ok,0],xy[acorners],store.stats['input_mean'][0])
            ma=qA(torch.tensor(m0.ravel())).numpy()
            ba=ma+settings['baseline_weight']*((idw[aa]*qA.weights.numpy()).sum(1)-ma)
            innovation=np.stack([aggregate_innovation(current[:,0],ba,k,ok&store.query_valid[aid]) for k in maker.kernels],-1)
            maps['features'][i]=f;maps['innovation'][i]=innovation;maps['mean'][i]=(base[qi]*qw).sum(1);maps['M0'][i]=m0
        if not all(np.isfinite(maps[k][i]).all() for k in ('features','innovation','mean','M0')):raise ValueError('Nonfinite constructed inputs')
        maps['row_hashes'][i]=row_digest([maps[k][i] for k in ('features','innovation','mean','M0')]);done[i]=True
        if (i+1)%24==0 or done.all():
            commit();progress('PREPARING',completed=int(done.sum()),total=n,new_M0_this_session=generated,reused_this_session=reused,seconds=time.monotonic()-started)
    # Targets are joined only after all input rows and temporal roles are locked.
    with np.load(bundle/'data.npz') as archive:
        ix=np.searchsorted(hours,selected);truth=np.array(archive['truth'][ix]);valid=np.array(archive['valid'][ix]);reference=np.array(archive['mean'][ix])
        amplitude=np.array(archive['amplitude'][ix]);projection=np.array(archive['projection']);remainder=np.array(archive['remainder'])
    error=float(np.max(abs(np.asarray(maps['mean'])[valid]-reference[valid])))
    if error>.001:raise ValueError(f'Frozen base requery differs from bundle: {error}')
    for split in splits:
        ix=np.searchsorted(selected,split['train_hours'])
        if (valid[ix][:,np.array(split['fit'])].sum(1)<4).any():raise ValueError('Registered training hour has fewer than4 fitted observations')
    np.savez_compressed(ROOT/'data.npz',hours=selected,features=np.asarray(maps['features']),innovation=np.asarray(maps['innovation']),
        mean=np.asarray(maps['mean']),truth=truth,valid=valid,amplitude=amplitude,projection=projection,remainder=remainder,
        query_indices=qi,query_weights=qw,station_ids=station_ids,global_nodes=nodes,xy=old['xy'],height=old['height'])
    write_json(ROOT/'data_complete.json',dict(sha256=sha256(ROOT/'data.npz'),hours=n,train_hours=576,inner_hours=48,outer_hours=len(prior_lock['outer_hours']),
        baseline_query_max_mm=error,completed=True,seconds=time.monotonic()-started))
    progress('DATA_READY',hours=n,train_hours=576,inner_hours=48)


class CoverageData(Pilot):
    def __init__(self):
        self.a=dict(np.load(ROOT/'data.npz'));self.lock=read_json(ROOT/'experiment.lock.json')
        self.settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')
        self.splits=[{k:np.asarray(v,bool if k in ('fit','inner','outer') else 'i8') for k,v in s.items()} for s in self.lock['splits']]
        self.t={k:torch.tensor(self.a[k],device='cuda') for k in ('features','innovation','query_indices','query_weights','mean','truth','valid')}
