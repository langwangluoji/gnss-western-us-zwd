"""Frozen inputs and lazy full-grid cache for the paired spatial processors."""
import os
import shutil
import hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import time
import numpy as np
import torch

from local_context_features import LocalFeatureBuilder, VertexQuery
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT/'results/20261121_grid_processor_pair_v01'
BUNDLE_FILES=('data.npz','manifest.json','spatial.npz')


def row_digest(arrays):
    digest=hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return np.frombuffer(digest.digest(),dtype='u1').copy()


def verify_row(arrays,expected):
    if not np.array_equal(row_digest(arrays),expected):
        raise ValueError('Committed cache row checksum mismatch')


def nominal_hour(value):
    return int(np.datetime64(value,'h').astype('i8')) if isinstance(value,str) else int(value)


def exclude_weeks(hours, starts, buffer=18):
    hours = np.asarray(hours,dtype='i8')
    keep = np.ones(len(hours),bool)
    for start in starts:
        start=nominal_hour(start)
        keep &= (hours < start-buffer) | (hours >= start+168+buffer)
    return hours[keep]


def feature69(builder, frames, m0, cycle, shape):
    raw = builder.build(frames,m0,np.zeros(32,'f4'),cycle)[:,:67].copy()
    density = [5,11,17,23,29,35,41,47,49,51]
    raw[:,density] = np.log1p(raw[:,density])/np.log1p(174.)
    xy = builder.grid_xy
    span = np.ptp(xy,axis=0)
    coords = 2*(xy-xy.min(0))/np.maximum(span,1)-1
    value = np.concatenate([raw,coords.astype('f4')],-1)
    return value.T.reshape(69,*shape).copy()


def verify_bundle(root,data_fingerprint,station_ids):
    root=Path(root); m=read_json(root/'manifest.json')
    if sha256(root/'data.npz')!=m['data_sha256']:
        raise ValueError('Frozen B bundle bytes changed')
    if m['frozen_contract']['foundation']['data']!=data_fingerprint:
        raise ValueError('Packed input release differs from frozen bundle')
    with np.load(root/'data.npz') as z:
        if not np.array_equal(z['station_ids'],station_ids):
            raise ValueError('B station role/order differs from frozen bundle')


def local_runtime():
    return dict(data_root=str((PROJECT/'../../02_data/processed/gnss_field_server_v1').resolve()),
        m0=str(PROJECT/'results/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/seed_42/mean_base/candidate_005000.pt'),
        bundle=str(PROJECT/'results/20261119_local_edm_subspace_v01/server_bundle'),
        design=str(PROJECT/'results/20261120_architecture_research_v01/training_design_v01.json'),
        outer=str(PROJECT/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json'),
        parameters=str(PROJECT/'results/20261119_local_fused_probability_v02/B_only_parameters.lock.json'),
        )


class GridStudy:
    def __init__(self, root=RUN, *, device='cuda', runtime=None):
        from gnss_field.data import PackedStore
        from gnss_field.training.prediction import safe_query_geometry
        self.root,self.device = Path(root),device
        self.root.mkdir(parents=True,exist_ok=True)
        self.runtime = runtime or local_runtime()
        self.store = PackedStore(self.runtime['data_root'],purpose='train')
        bundle = Path(self.runtime['bundle'])
        verify_bundle(bundle,self.store.manifest['fingerprint'],self.store.ids('B'))
        self.a = dict(np.load(bundle/'data.npz'))
        design = read_json(self.runtime['design'])
        hours = self.a['hours']
        self.train_hours = exclude_weeks(hours[self.a['training_indices']],design['inner_week_starts'])
        self.inner_hours = np.asarray(design['inner_hours'],dtype='i8')
        outer = read_json(self.runtime['outer'])
        self.outer_hours = np.unique(np.concatenate([np.arange(nominal_hour(s),nominal_hour(s)+168)
                                                     for s in outer['week_starts']]))
        if len(self.train_hours)!=5472 or np.intersect1d(self.train_hours,self.inner_hours).size:
            raise ValueError('Full training/inner split differs from approved protocol')
        if np.intersect1d(self.train_hours,self.outer_hours).size:
            raise ValueError('Outer weather weeks overlap training')
        self.indices(self.train_hours); self.indices(self.inner_hours); self.indices(self.outer_hours)
        self.folds = np.array(design['B_fold_assignment'])
        self.shape = self.store.shape
        if tuple(self.shape)!=(144,96):
            raise ValueError('Unexpected publication grid')
        self.scale = float(self.store.stats['z0_std'])
        self.weight = float(read_json(self.runtime['parameters'])['weight'])
        self.tau = float(read_json(bundle/'manifest.json')['tau_surface_mm'])
        self.m0_hash=sha256(self.runtime['m0'])
        if self.m0_hash!='605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0':
            raise ValueError('Frozen M0 changed')
        self.sources={key:sha256(self.runtime[key]) for key in ('m0','design','outer','parameters')}
        self.sources.update(bundle={name:sha256(bundle/name) for name in BUNDLE_FILES},
            training_data=self.store.manifest['training_hash'],grid=self.store.grid_id,
            scientific_config=sha256(PROJECT/'configs/grid_processor_pair_v01.json'),
            code={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('local_context_features.py'),
                  PROJECT/'src/gnss_field/models/grid_processor.py']})
        immutable_json(self.root/'data.lock.json',dict(sources=self.sources, training_hours=self.train_hours,
            inner_hours=self.inner_hours,outer_hours=self.outer_hours,folds=self.folds,
            schema='69_fixed_train_stats_logdensity_lambert_v1_M0_innovation',C2019=False,D2020=False,
            old_base_prior_exposure='incremental development; historical base and covariance have B exposure'))
        grid=self.store.grid
        self.nodes=np.arange(np.prod(self.shape))
        self.gridxy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
        aid=self.store.ids('A')
        axy,_=safe_query_geometry(self.store,aid)
        self.builder=LocalFeatureBuilder(self.gridxy,self.store.xy[aid],
            self.store.stations.height.to_numpy()[aid],grid['height'],grid['region'],self.nodes,
            self.store.stats,axy,self.store.query_valid[aid])
        xy,height=safe_query_geometry(self.store,self.a['station_ids'])
        query=VertexQuery(self.shape,xy,height,self.nodes)
        self.qindices=query.indices.to(device); self.qweights=query.weights.to(device)
        self.t={key:torch.as_tensor(self.a[key],device=device) for key in
                ('truth','valid','amplitude','projection','remainder')}
        cache=self.root/'cache'; cache.mkdir(exist_ok=True)
        immutable_json(cache/'contract.json',dict(sources=self.sources, hours=hours,shape=self.shape,
            dtype='float32',feature_shape=[69,*self.shape]))
        self.done_path=cache/'cursor.json'
        committed=self.done_path.exists()
        self.features=self._memmap(cache/'features69.npy',(len(hours),69,*self.shape),required=committed)
        self.m0=self._memmap(cache/'M0.npy',(len(hours),*self.shape),required=committed)
        self.mean=self._memmap(cache/'mean.npy',(len(hours),len(self.a['station_ids'])),required=committed)
        self.row_hashes=self._memmap(cache/'row_hashes.npy',(len(hours),32),required=committed,dtype='u1')
        self.done=np.array(read_json(self.done_path)['done'],bool) if self.done_path.exists() else np.zeros(len(hours),bool)
        self.net=None
        self.log_cache=cache/'events.jsonl'

    @staticmethod
    def _memmap(path,shape,*,required=False,dtype='f4'):
        if path.exists():
            a=np.load(path,mmap_mode='r+')
            if a.shape!=shape or a.dtype!=np.dtype(dtype):
                raise ValueError('Cache shape or dtype changed')
            return a
        if required:
            raise ValueError('Committed cache component is missing; refusing empty recreation')
        if shutil.disk_usage(path.parent).free < int(np.prod(shape))*np.dtype(dtype).itemsize+2**30:
            raise ValueError(f'Insufficient free space for immutable FP32 cache: {path}')
        return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

    def indices(self,hours):
        ix=np.searchsorted(self.a['hours'],hours)
        if (ix>=len(self.a['hours'])).any() or not np.array_equal(self.a['hours'][ix],hours):
            raise ValueError('Hour outside B2018 bundle')
        return ix

    def frozen_m0(self,hour):
        from gnss_field.data import ContextBuilder
        from gnss_field.models.mean import MeanField
        from gnss_field.training.fields import predict_patches
        devices=[torch.cuda.current_device()] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            if self.net is None:
                self.net=MeanField().to(self.device).eval().requires_grad_(False)
                self.net.load_state_dict(torch.load(self.runtime['m0'],map_location=self.device,weights_only=False)['ema'])
                self.context_builder=ContextBuilder(self.store)
            context=self.context_builder.build_context([hour],device=self.device)
            return predict_patches(self.net,context,self.store,chunk=4)[0][0,0].cpu().numpy()

    def prepare(self,indices):
        from gnss_field.data.context import cycles
        from gnss_field.models.grid_processor import query_field
        start=time.monotonic(); count=0; profiles=[]
        for index in indices:
            if self.done[index]:
                verify_row([self.features[index],self.m0[index],self.mean[index]],self.row_hashes[index])
                continue
            hour=int(self.a['hours'][index])
            frames=np.array([self.store.context_values([hour-offset])[0] for offset in (0,6,12,18)])
            t=time.monotonic(); m0=self.frozen_m0(hour); m0_seconds=time.monotonic()-t
            t=time.monotonic()
            feature=feature69(self.builder,frames,m0,cycles([hour])[0],self.shape)
            feature_seconds=time.monotonic()-t
            base=m0+self.weight*(feature[59]*self.scale-m0)
            with torch.no_grad():
                mean=query_field(torch.tensor(base[None,None],device=self.device),self.qindices,self.qweights)[0].cpu().numpy()
            valid=self.a['valid'][index]
            if not np.isfinite(feature).all() or not np.isfinite(mean).all():
                raise ValueError('Nonfinite frozen input or base')
            if np.max(np.abs(mean[valid]-self.a['mean'][index,valid]))>.001:
                raise ValueError('Grid-reloaded frozen base differs from archived station values')
            self.features[index]=feature; self.m0[index]=m0; self.mean[index]=mean
            self.row_hashes[index]=row_digest([feature,m0,mean])
            self.features.flush(); self.m0.flush(); self.mean.flush()
            self.row_hashes.flush()
            self.done[index]=True; count+=1
            write_json(self.done_path,dict(done=self.done))
            profiles.append(dict(m0_seconds=m0_seconds,feature_seconds=feature_seconds))
        if count:
            import json
            with self.log_cache.open('a',encoding='utf-8') as out:
                out.write(json.dumps(dict(new_hours=count,complete_hours=int(self.done.sum()),
                    elapsed_seconds=time.monotonic()-start,updated=time.time()))+'\n')
        return profiles

    def batch(self,hours):
        ix=self.indices(hours); self.prepare(ix)
        x=torch.tensor(np.array(self.features[ix]),device=self.device)
        base=torch.tensor(np.array(self.mean[ix]),device=self.device)
        available=torch.tensor(np.isfinite(np.array([self.store.context_values([int(h)-o])[0]
            for h in hours for o in (0,6,12,18)])).all(-1).reshape(len(ix),4,-1).any((1,2)),device=self.device)
        return ix,x,base,available

    def delta(self,model,features,indices,available):
        from gnss_field.models.grid_processor import query_field
        field=model(features,available)*self.scale
        return query_field(field,self.qindices,self.qweights)*self.t['amplitude'][indices,None]

    def noise(self,indices,members,seed=None):
        from gnss_field.models.diffusion import keyed_noise
        b=len(indices); rank=self.t['projection'].shape[1]; rest=self.t['remainder'].shape[1]
        if seed is None:
            latent=torch.randn(b,members,rank,device=self.device)
            remainder=torch.randn(b,members,rest,device=self.device)
            error=torch.randn(b,members,905,device=self.device)[...,self.a['station_ids']]
        else:
            hours=self.a['hours'][indices]
            latent=keyed_noise(hours,members,seed,shape=(rank,),device=self.device)
            remainder=keyed_noise(hours,members,seed+1000,shape=(rest,),device=self.device)
            error=keyed_noise(hours,members,seed+2000,shape=(905,),device=self.device)[...,self.a['station_ids']]
        return (latent@self.t['projection'].T+remainder@self.t['remainder'].T+self.tau*error)*self.t['amplitude'][indices,None,None]
