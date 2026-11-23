"""Input-only full-grid cache plus explicitly separated local B observation roles."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import time
import json
import numpy as np
import torch
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.data import PackedStore
from gnss_field.data.context import cycles
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.models.grid_generation import gaussian_prior_field
from gnss_field.models.diffusion import keyed_noise
from gnss_field.models.grid_processor import query_field
from local_context_features import LocalFeatureBuilder
from grid_pair_data import local_runtime,feature69,row_digest,verify_row

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SOURCE=PROJECT/'results/20261122_local_coverage_expansion_v01'


def monthly_subset(hours):
    hours=np.sort(np.asarray(hours,'i8'));months=hours.astype('datetime64[h]').astype('datetime64[M]')
    return np.array([hours[months==m][0] for m in np.unique(months)],dtype='i8')


def schedule(hours,steps,batch=2):
    rng=np.random.default_rng(42);hours=np.asarray(hours,'i8')
    months=hours.astype('datetime64[h]').astype('datetime64[M]');unique=np.unique(months)
    rows=[]
    while len(rows)<steps*batch:
        parts=[rng.permutation(hours[months==m]).tolist() for m in unique]
        for j in range(max(map(len,parts))):
            for p in parts:
                if j<len(p):rows.append(p[j])
    return np.asarray(rows[:steps*batch]).reshape(steps,batch)


def prepare():
    a=dict(np.load(SOURCE/'data.npz'));lock=read_json(SOURCE/'experiment.lock.json')
    if sha256(SOURCE/'data.npz')!=read_json(SOURCE/'data_complete.json')['sha256']:raise ValueError('Source data changed')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    signature=dict(source_data=sha256(SOURCE/'data.npz'),source_roles=sha256(SOURCE/'experiment.lock.json'),
        feature_builder=sha256(PROJECT/'tools/grid_pair_data.py'),local_builder=sha256(PROJECT/'tools/local_context_features.py'),
        cache_builder=sha256(__file__),schema='full-grid legacy69: compact67+absoluteXY; no new features',C2019=False,D2020=False)
    immutable_json(cache/'contract.json',signature)
    path=cache/'conditions69.npy';shape=(len(a['hours']),69,144,96)
    state=read_json(cache/'cursor.json') if (cache/'cursor.json').exists() else dict(done=[False]*len(a['hours']))
    if any(state['done']) and not path.exists():raise ValueError('Committed cache missing')
    f=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype='f4',shape=shape)
    hashpath=cache/'row_hashes.npy'
    if any(state['done']) and not hashpath.exists():raise ValueError('Committed row digests missing')
    hashes=np.load(hashpath,mmap_mode='r+') if hashpath.exists() else np.lib.format.open_memmap(hashpath,mode='w+',dtype='u1',shape=(len(a['hours']),32))
    store=PackedStore(local_runtime()['data_root'],purpose='context');grid=store.grid;aid=store.ids('A');aq,_=safe_query_geometry(store,aid)
    nodes=np.arange(np.prod(store.shape));xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    maker=LocalFeatureBuilder(xy,store.xy[aid],store.stations.height.to_numpy()[aid],grid['height'],grid['region'],nodes,store.stats,aq,store.query_valid[aid])
    old_m0=np.load(SOURCE/'cache/M0.npy',mmap_mode='r');oldhash=np.load(SOURCE/'cache/row_hashes.npy',mmap_mode='r')
    oldfeatures=np.load(SOURCE/'cache/features.npy',mmap_mode='r');oldinnovation=np.load(SOURCE/'cache/innovation.npy',mmap_mode='r');oldmean=np.load(SOURCE/'cache/mean.npy',mmap_mode='r')
    qi=a['global_nodes'][a['query_indices']];qw=a['query_weights'];runtime=local_runtime();mix=read_json(runtime['parameters'])['weight']
    scale=store.stats['z0_std'];started=time.monotonic();compact_error=mean_error=0.
    frames=[store.context_values(a['hours']-offset) for offset in (0,6,12,18)]
    for i,h in enumerate(a['hours']):
        if state['done'][i]:verify_row([f[i]],hashes[i]);continue
        verify_row([oldfeatures[i],oldinnovation[i],oldmean[i],old_m0[i]],oldhash[i])
        value=feature69(maker,np.array([v[i] for v in frames]),old_m0[i],cycles([h])[0],store.shape)
        err=float(np.max(abs(value[:67].reshape(67,-1)[:,a['global_nodes']].T-a['features'][i])))
        compact_error=max(compact_error,err)
        if err>1e-5:raise ValueError(f'Full-grid conditions differ from compact source: {err}')
        base=old_m0[i]+mix*(value[59]*scale-old_m0[i]);mean=(base.ravel()[qi]*qw).sum(1)
        error=float(np.max(abs(mean-a['mean'][i])));mean_error=max(mean_error,error)
        if error>.001:raise ValueError(f'Frozen mean requery mismatch: {error}')
        f[i]=value;hashes[i]=row_digest([value]);state['done'][i]=True
        if (i+1)%24==0 or all(state['done']):
            f.flush();hashes.flush();write_json(cache/'cursor.json',state)
            print('PREPARE',i+1,len(a['hours']),round(time.monotonic()-started,1),flush=True)
    write_json(cache/'complete.json',dict(complete=True,hours=len(a['hours']),compact67_max_error=compact_error,mean_requery_max_mm=mean_error,
        row_hashes=sha256(hashpath),contract=sha256(cache/'contract.json'),seconds=time.monotonic()-started))


def validate_inputs(cache,source):
    cache=Path(cache);contract=read_json(cache/'contract.json');complete=read_json(cache/'complete.json')
    if sha256(source)!=contract['source_data']:raise ValueError('Source data checksum differs')
    if complete['contract']!=sha256(cache/'contract.json') or complete['row_hashes']!=sha256(cache/'row_hashes.npy'):
        raise ValueError('Input cache contract/digest checksum differs')
    done=read_json(cache/'cursor.json')['done'];features=np.load(cache/'conditions69.npy',mmap_mode='r');hashes=np.load(cache/'row_hashes.npy',mmap_mode='r')
    if not complete['complete'] or not all(done) or len(done)!=len(features) or len(features)!=complete['hours']:
        raise ValueError('Input cache incomplete')
    for row,digest in zip(features,hashes):verify_row([row],digest)


class Data:
    def __init__(self,device='cuda'):
        validate_inputs(ROOT/'cache',SOURCE/'data.npz')
        self.device=device;self.a=dict(np.load(SOURCE/'data.npz'));self.lock=read_json(SOURCE/'experiment.lock.json')
        self.features=np.load(ROOT/'cache/conditions69.npy',mmap_mode='r')
        if not all(read_json(ROOT/'cache/cursor.json')['done']):raise ValueError('Input cache incomplete')
        runtime=local_runtime();sp=dict(np.load(Path(runtime['bundle'])/'spatial.npz'))
        self.phi=torch.tensor(sp['phi'],device=device);self.v=torch.tensor(sp['v'],device=device)
        self.tau=float(read_json(Path(runtime['bundle'])/'manifest.json')['tau_surface_mm'])
        self.qi=torch.tensor(self.a['global_nodes'][self.a['query_indices']],device=device);self.qw=torch.tensor(self.a['query_weights'],device=device)
        self.coords=torch.tensor(np.c_[self.a['xy']/400000,self.a['height']/3000],dtype=torch.float32,device=device)
        self.geometry=self.qw/torch.exp(-self.coords[:,2:3])
        self.m0=np.load(SOURCE/'cache/M0.npy',mmap_mode='r')
        self.settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json')
        self.rank,self.rest=self.v.shape;self.shape=(144,96)
        self.truth=torch.tensor(self.a['truth'],device=device);self.valid=torch.tensor(self.a['valid'],device=device)
        self.scales=[];prior_variance=np.sum(self.a['projection'].astype('f8')**2,1)+np.sum(self.a['remainder'].astype('f8')**2,1)+self.tau**2
        for split in self.lock['splits']:
            ti=self.indices(split['train_hours']);mask=self.a['valid'][ti]&np.array(split['fit'])[None]
            residual=(self.a['truth'][ti]-self.a['mean'][ti])/self.a['amplitude'][ti,None]
            self.scales.append(float(np.sqrt(np.mean((residual**2/prior_variance[None])[mask]))))
        immutable_json(ROOT/'data.lock.json',dict(input_cache=sha256(ROOT/'cache/contract.json'),source_roles=sha256(SOURCE/'experiment.lock.json'),
            input_row_hashes=sha256(ROOT/'cache/row_hashes.npy'),gaussian_scale=self.scales,
            train_hours=self.lock['splits'][0]['train_hours'],inner_hours=monthly_subset(self.lock['splits'][0]['inner_hours']),
            outer_hours=monthly_subset(self.lock['outer_hours']),source_exposure=self.lock['exposure'],tau_once=True,C2019=False,D2020=False))

    def indices(self,hours):
        ix=np.searchsorted(self.a['hours'],hours)
        if (ix>=len(self.a['hours'])).any() or not np.array_equal(self.a['hours'][ix],hours):raise ValueError('Unregistered hour')
        return ix

    def batch(self,hours):
        ix=self.indices(hours);x=torch.tensor(np.array(self.features[ix]),device=self.device)
        return ix,x,torch.tensor(self.a['mean'][ix],device=self.device),torch.tensor(self.a['amplitude'][ix],device=self.device)

    def conditions(self,x):
        return (x.flatten(2)[:,:,self.qi]*self.geometry[None,None]).sum(-1).transpose(1,2)

    def role(self,fold,role):return torch.tensor(self.lock['splits'][fold][role],device=self.device)

    def draw(self,hours,members,seed,fold,scaled=True):
        def noise(shape,offset):return keyed_noise(hours,members,seed+offset,shape=shape,device=self.device)
        latent=noise((self.rank,),0);remainder=noise((self.rest,),1000)
        errors=noise((905,),2000)[...,self.a['station_ids']];white=noise((1,*self.shape),3000)
        prior=gaussian_prior_field(self.phi,self.v,latent,remainder,self.shape)
        s=self.scales[fold] if scaled else 1.
        return prior*s,white,errors*(self.tau*s)

    def sample(self,generator,hours,fold,members,seed,chunk=4,scaled=True,fields=False):
        ix,x,base,amp=self.batch(hours);prior,white,error=self.draw(hours,members,seed,fold,scaled)
        values=[];deltas=[];grids=[]
        for start in range(0,members,chunk):
            k=min(chunk,members-start);p=prior[:,start:start+k].flatten(0,1);w=white[:,start:start+k].flatten(0,1)
            c=x[:,None].expand(-1,k,-1,-1,-1).flatten(0,1)
            y=p if generator is None else generator.sample_residual(c,p,w)
            delta=query_field(y-p,self.qi,self.qw).reshape(len(hours),k,-1)*amp[:,None,None]
            station=query_field(y,self.qi,self.qw).reshape(len(hours),k,-1)
            values.append(base[:,None]+amp[:,None,None]*(station+error[:,start:start+k]))
            deltas.append(delta)
            if fields:grids.append(y.reshape(len(hours),k,1,*self.shape))
        return dict(samples=torch.cat(values,1),delta=torch.cat(deltas,1),base=base,indices=ix,
            fields=torch.cat(grids,1) if fields else None,observation_noise=amp[:,None,None]*error)

    def mean_grid(self,hours):
        ix,x,_,_=self.batch(hours);m0=torch.tensor(np.array(self.m0[ix]),device=self.device)[:,None]
        return m0+self.settings['baseline_weight']*(x[:,59:60]*self.settings['grid_scale']-m0)


if __name__=='__main__':prepare()
