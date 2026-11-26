"""Fold-dependent frozen geometry means with B-training-only probability recalibration."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import time
import numpy as np
import torch
from retrain_common import ROOT,PROJECT,OLD,OPERATORS
from retrain_geometry import GeometryBase
from pilot_data import Data as OldData,SOURCE
from grid_pair_data import local_runtime,row_digest,verify_row
from gnss_field.data import PackedStore
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from local_innovation_components import gaussian_records,fit_gaussian
from gp_ndp import correlation_factor


def prepare():
    a=dict(np.load(SOURCE/'data.npz'));store=PackedStore(local_runtime()['data_root'],purpose='context');geo=GeometryBase(store)
    cache=ROOT/'B_cache';cache.mkdir(exist_ok=True);n=len(a['hours'])
    immutable_json(cache/'lock.json',dict(source=sha256(SOURCE/'data.npz'),mean=sha256(ROOT/'mean.lock.json'),
        conditions=sha256(OLD/'cache/contract.json'),runner=sha256(__file__),folds=2,shape=[144,96]))
    def mmap(name,shape,dtype='f4'):
        path=cache/name
        return np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
    fields=mmap('new_Z0.npy',(n,2,144,96));means=mmap('new_station_mean.npy',(n,2,511));hashes=mmap('row_hashes.npy',(n,32),'u1')
    cursor=read_json(cache/'cursor.json') if (cache/'cursor.json').exists() else dict(done=0)
    for i in range(cursor['done']):verify_row([fields[i],means[i]],hashes[i])
    old_m0=np.load(SOURCE/'cache/M0.npy',mmap_mode='r');qi=a['global_nodes'][a['query_indices']];qw=a['query_weights']
    frames=[store.context_values(a['hours']-offset) for offset in (0,6,12,18)];started=time.monotonic()
    for i in range(cursor['done'],n):
        if (ROOT/'PAUSE').exists():raise RuntimeError('B mean preparation paused')
        f,_=geo.predict(np.array([v[i] for v in frames]),old_m0[i],int(a['hours'][i]))
        fields[i]=f.reshape(2,144,96);means[i]=(f[:,qi]*qw).sum(-1);hashes[i]=row_digest([fields[i],means[i]])
        if (i+1)%24==0 or i+1==n:
            for v in (fields,means,hashes):v.flush()
            cursor['done']=i+1;write_json(cache/'cursor.json',cursor)
            write_json(ROOT/'B_status.json',dict(stage='B_NEW_MEAN_CACHE',hours=i+1,total=n,seconds=time.monotonic()-started))
            print('B_NEW_MEAN',i+1,n,round(time.monotonic()-started,1),flush=True)
    write_json(cache/'complete.json',dict(fields=sha256(cache/'new_Z0.npy'),means=sha256(cache/'new_station_mean.npy'),
        rows=sha256(cache/'row_hashes.npy'),lock=sha256(cache/'lock.json'),hours=n,complete=True))


class Data(OldData):
    def __init__(self,device='cuda'):
        super().__init__(device)
        complete=read_json(ROOT/'B_cache/complete.json')
        for key,name in [('fields','new_Z0.npy'),('means','new_station_mean.npy'),('rows','row_hashes.npy')]:
            if sha256(ROOT/'B_cache'/name)!=complete[key]:raise ValueError('New base cache hash differs')
        self.new_fields=np.load(ROOT/'B_cache/new_Z0.npy',mmap_mode='r');self.new_means=np.load(ROOT/'B_cache/new_station_mean.npy',mmap_mode='r')
        self.fold=0;self.old_mean=self.a['mean'];self.calibration=[];self.factors=[]
        covariance=self.a['projection'].astype('f8')@self.a['projection'].astype('f8').T+self.a['remainder'].astype('f8')@self.a['remainder'].astype('f8').T
        _,groups=np.unique(np.floor(self.a['xy']/200000).astype(int),axis=0,return_inverse=True)
        for fold,split in enumerate(self.lock['splits']):
            ti=self.indices(split['train_hours']);mask=self.a['valid'][ti]&np.array(split['fit'])[None]
            error=self.a['truth'][ti]-self.new_means[ti,fold]
            records=gaussian_records(error,mask,self.a['amplitude'][ti],covariance,groups)
            row=fit_gaussian(records,self.tau,'two_scales');self.calibration.append(row)
            factor=correlation_factor(self.a['projection']*row['field'],self.a['remainder']*row['field'],self.tau*row['nugget'])
            self.factors.append(torch.tensor(factor,device=device))
        immutable_json(ROOT/'probability.lock.json',dict(mean=sha256(ROOT/'mean.lock.json'),B_means=complete['means'],
            roles=sha256(SOURCE/'experiment.lock.json'),calibration=self.calibration,fit='fitted B2018 training576 only;200km group composite NLL',
            C2019_calibration=False,D2020=False,runner=sha256(__file__)))
        self.set_fold(0)

    def set_fold(self,fold):
        self.fold=int(fold);self.a['mean']=self.new_means[:,fold]

    def batch(self,hours):
        ix=self.indices(hours);x=torch.tensor(np.array(self.features[ix]),device=self.device)
        return ix,x,torch.tensor(np.array(self.new_means[ix,self.fold]),device=self.device),torch.tensor(self.a['amplitude'][ix],device=self.device)

    def draw(self,hours,members,seed,fold,scaled=True):
        prior,white,error=super().draw(hours,members,seed,fold,scaled=False)
        row=self.calibration[fold]
        return prior*(row['field'] if scaled else 1),white,error*(row['nugget'] if scaled else 1)

    def sample(self,generator,hours,fold,members,seed,chunk=4,scaled=True,fields=False):
        self.set_fold(fold)
        return super().sample(generator,hours,fold,members,seed,chunk=chunk,scaled=scaled,fields=fields)

    def mean_grid(self,hours):
        return torch.tensor(np.array(self.new_fields[self.indices(hours),self.fold]),device=self.device)[:,None]


if __name__=='__main__':prepare()
