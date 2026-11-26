"""Read-only previous inputs and independent frozen mean, coherent prior and queries."""
from pathlib import Path
import time
import numpy as np
import torch
from grid_pair_data import GridStudy,verify_row
from gnss_field.artifacts import read_json,immutable_json,sha256
from gnss_field.models.grid_processor import query_field
from gnss_field.models.grid_generation import gaussian_prior_field
from gnss_field.models.diffusion import keyed_noise
from gnss_field.training.engine import PauseRequested


def pilot_outer_hours(hours):
    hours=np.asarray(hours,dtype='i8')
    if len(hours)!=2016 or len(np.unique(hours))!=2016:
        raise ValueError('Expected twelve registered complete outer weeks')
    weeks=hours.reshape(12,168)
    if not (np.diff(weeks,axis=1)==1).all(): raise ValueError('Outer week incomplete')
    return weeks[:,np.arange(10,168,21)].ravel()


class ReadOnlyGridStudy(GridStudy):
    def prepare(self,indices):
        for i in indices:
            if not self.done[i]: raise ValueError(f'Frozen source cache missing committed hour {int(self.a["hours"][i])}')
            verify_row([self.features[i],self.m0[i],self.mean[i]],self.row_hashes[i])
        return []


class GenerationData:
    def __init__(self,root,runtime,device='cuda'):
        self.root,self.device,self.runtime=Path(root),device,runtime
        self.root.mkdir(parents=True,exist_ok=True)
        old=Path(runtime['source_root'])
        for name in ('data.lock.json','cache/contract.json','cache/cursor.json','cache/features69.npy','cache/M0.npy','cache/mean.npy','cache/row_hashes.npy'):
            if not (old/name).is_file(): raise ValueError(f'Incomplete frozen source cache: {name}')
        self.source=ReadOnlyGridStudy(old,device=device,runtime=read_json(runtime['source_runtime']))
        self.train_hours=self.source.train_hours; self.inner_hours=self.source.inner_hours
        self.outer_hours=pilot_outer_hours(self.source.outer_hours)
        self.folds=self.source.folds; self.shape=self.source.shape; self.a=self.source.a
        if runtime.get('mean_mode')!='frozen_m0_idw': raise ValueError('This pilot locks the audited M0+IDW mean')
        self.deadline=None
        spatial=dict(np.load(Path(self.source.runtime['bundle'])/'spatial.npz'))
        self.phi=torch.as_tensor(spatial['phi'],device=device)
        self.v=torch.as_tensor(spatial['v'],device=device)
        self.rank=self.v.shape[0]; self.rest=self.v.shape[1]
        self.ids=np.asarray(self.a['station_ids'])
        store=self.source.store
        self.xy=store.xy[self.ids]
        self.coords=torch.tensor(np.c_[self.xy/400000,store.stations.height.to_numpy()[self.ids]/3000],dtype=torch.float32,device=device)
        self.geometry_weights=self.source.qweights/torch.exp(-self.coords[:,2:3])
        self.tau=self.source.tau
        contract=dict(source_data=sha256(self.source.root/'data.lock.json'),source_cache=sha256(self.source.root/'cache/contract.json'),
            mean_mode='frozen_m0_idw',M0=self.source.m0_hash,mixture=sha256(self.source.runtime['parameters']),
            config=sha256(Path(__file__).resolve().parents[1]/'configs/grid_generation_v02.json'),
            train_hours=self.train_hours,inner_hours=self.inner_hours,outer_pilot_hours=self.outer_hours,
            probability='same coherent prior; tau only once at stations; free-grid correction',C2019=False,D2020=False)
        immutable_json(self.root/'data.lock.json',contract)
        for arr in (self.source.features,self.source.m0,self.source.mean,self.source.row_hashes): arr.flags.writeable=False

    def check(self,force=False):
        if not force: return
        if (self.root/'PAUSE').exists() or (self.deadline and time.monotonic()>=self.deadline):
            raise PauseRequested('Generation queue paused at update/hour boundary')

    def batch(self,hours,fold):
        self.check()
        ix,x,base,available=self.source.batch(hours)
        mean=base
        amplitude=self.source.t['amplitude'][ix]
        return ix,x,mean,amplitude,available

    def local_conditions(self,x):
        gathered=x.flatten(2)[:,:,self.source.qindices]
        return (gathered*self.geometry_weights[None,None]).sum(-1).transpose(1,2)

    @torch.no_grad()
    def mean_grid(self,hours,fold):
        ix,x,_,g,available=self.batch(hours,fold)
        m0=torch.tensor(np.array(self.source.m0[ix]),device=self.device)[:,None]
        base=m0+self.source.weight*(x[:,59:60]*self.source.scale-m0)
        return base
    def draw(self,hours,members,seed=None):
        b=len(hours)
        def noise(shape,offset):
            return torch.randn(b,members,*shape,device=self.device) if seed is None else keyed_noise(hours,members,seed+offset,shape=shape,device=self.device)
        latent=noise((self.rank,),0); remainder=noise((self.rest,),1000)
        errors=noise((905,),2000)[...,self.ids]
        white=noise((1,*self.shape),3000)
        prior=gaussian_prior_field(self.phi,self.v,latent,remainder,self.shape)
        return prior,white,errors

    def sample(self,system,hours,fold,members,seed=None,member_chunk=4,return_fields=False):
        ix,x,mean,g,available=self.batch(hours,fold)
        prior,white,errors=self.draw(hours,members,seed)
        outputs=[]; changes=[]; fields=[]
        for start in range(0,members,member_chunk):
            self.check(); end=min(start+member_chunk,members); k=end-start
            p=prior[:,start:end].flatten(0,1); w=white[:,start:end].flatten(0,1)
            c=x[:,None].expand(-1,k,-1,-1,-1).flatten(0,1)
            generated=p if system is None else system.generator.sample_residual(c,p,w)
            delta=(generated-p)*available[:,None,None,None].expand(-1,k,1,1).reshape(-1,1,1,1)
            generated=p+delta
            delta_y=query_field(delta,self.source.qindices,self.source.qweights).reshape(len(hours),k,-1)*g[:,None,None]
            coherent=query_field(generated,self.source.qindices,self.source.qweights).reshape(len(hours),k,-1)
            outputs.append(mean[:,None]+g[:,None,None]*(coherent+self.tau*errors[:,start:end]))
            changes.append(delta_y)
            if return_fields: fields.append(generated.reshape(len(hours),k,1,*self.shape)*g[:,None,None,None,None])
        forecast=torch.cat(outputs,1); delta=torch.cat(changes,1)
        if not torch.isfinite(forecast).all(): raise ValueError('Nonfinite observation forecast')
        return dict(samples=forecast,mean=mean+delta.mean(1),delta=delta,base=mean,
            fields=torch.cat(fields,1) if return_fields else None,observation_noise=g[:,None,None]*self.tau*errors,indices=ix)
