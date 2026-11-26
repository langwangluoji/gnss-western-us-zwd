"""Split/fold-aware adapter; only station identities are sampled during training."""
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from .p14_fusion import FusionData
from .unet_foundation import Features


def validate_window(hours,anchors,length,start,stop):
    a=np.asarray(anchors)
    if not len(a) or not np.array_equal(a,np.arange(a[0],a[-1]+1)):
        raise ValueError('target hours must be consecutive')
    if a[0]-length+1<start or a[-1]>=stop:
        raise ValueError('history/target crosses split boundary')
    if not np.all(np.diff(hours[a[0]-length+1:a[-1]+1])==1):
        raise ValueError('nominal hourly axis has a gap')


def fold_groups(groups,nfolds=5,seed=251125):
    groups=np.asarray(groups);unique,counts=np.unique(groups,return_counts=True)
    rng=np.random.default_rng(seed);order=rng.permutation(len(unique))
    order=order[np.argsort(-counts[order],kind='stable')];loads=np.zeros(nfolds,int);assigned=[[] for _ in loads]
    for j in order:
        f=int(loads.argmin());assigned[f].append(unique[j]);loads[f]+=counts[j]
    return [np.flatnonzero(np.isin(groups,g)) for g in assigned]


class JointData(FusionData):
    def __init__(self,path,length=72,side=16,train_ids=None,fit_stop_year=2019):
        super().__init__(path,length)
        self.side=side;e=self.events
        self.train_ids=np.arange(len(e.geo['xy'])) if train_ids is None else np.asarray(train_ids)
        fit_stop=int(np.searchsorted(e.hours,np.datetime64(f'{fit_stop_year}-01-01','h').astype('int64')))
        # A fresh fold model must not inherit label-derived global moments.
        raw=np.array(e.values[:fit_stop,self.train_ids])
        self.mean=np.nanmean(raw,axis=(0,1)).astype('float32')
        self.std=np.maximum(np.nanstd(raw,axis=(0,1)),.1).astype('float32')
        self.mean_t=torch.as_tensor(self.mean,device='cuda');self.std_t=torch.as_tensor(self.std,device='cuda')
        for f in (self.grid_features,self.train_features,self.dev_features):f.mean=self.mean_t;f.std=self.std_t
        x=torch.linspace(float(self.lo[0]),float(self.hi[0]),side,device='cuda')
        y=torch.linspace(float(self.lo[1]),float(self.hi[1]),side,device='cuda')
        yy,xx=torch.meshgrid(y,x,indexing='ij');self.small_grid=torch.stack([xx.flatten(),yy.flatten()],-1)
        uv=(self.small_grid-self.origin)/25;h,w=self.shape
        uv=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1)[None,None]
        static=self.grid_static.T.reshape(1,8,h,w)
        self.time_static=F.grid_sample(static,uv,padding_mode='border',align_corners=True)[0,:,0].T
        # The small support uses the same projected coordinates and dense terrain.
        self.small_uv=uv
        coast=torch.expm1(self.time_static[:,7]*6).clamp_min(0)
        self.small_static=torch.cat([self.time_static,coast[:,None]/500],-1)
        geo=torch.cat([self.small_grid,self.time_static[:,2:3]*3,self.time_static[:,5:6]],-1)
        namespace=SimpleNamespace(mean=self.mean,std=self.std,a={'geo':self.old_geo})
        self.small_features=Features(namespace,self.time_static[:,:7].cpu().numpy(),geo.cpu().numpy())
        self.set_split(2016,fit_stop_year)

    def set_split(self,start_year,stop_year):
        e=self.events
        self.split_start=int(np.searchsorted(e.hours,np.datetime64(f'{start_year}-01-01','h').astype('int64')))
        self.split_stop=int(np.searchsorted(e.hours,np.datetime64(f'{stop_year}-01-01','h').astype('int64')))
        p=torch.arange(self.split_stop-self.split_start,device='cuda')[:,None]
        raw=self.values[self.split_start:self.split_stop,:,0]
        last=torch.cummax(torch.where(torch.isfinite(raw),p,-1),0).values
        self.station_age=torch.zeros_like(self.values[:,:,0])
        self.station_age[self.split_start:self.split_stop]=(p-last).float()

    def temporal_history(self,raw,hours,c,batch,full_age):
        valid=torch.isfinite(raw);v=raw.clone()
        v[:,:,0]*=torch.exp(self.train_features.cg[c,2]/3)[None]
        v=torch.where(valid,(v-v.new_tensor([0,280,900]))/v.new_tensor([100,20,100]),0)
        v=v.unfold(0,self.length,1).permute(0,3,1,2)
        m=valid.unfold(0,self.length,1).permute(0,3,1,2).float()
        age=full_age.unfold(0,self.length,1).permute(0,2,1)
        d=torch.cdist(self.small_grid,self.train_features.cg[c,:2]);parts=[]
        for radius in (50,150,300):
            w=torch.exp(-.5*(d/radius).square())*(d<=3*radius)
            den=torch.einsum('mc,blcj->blmj',w,m);safe=w.sum(1)[None,None,:,None].clamp_min(1e-8)
            avg=torch.einsum('mc,blcj->blmj',w,v)/den.clamp_min(1e-8)
            ages=torch.einsum('mc,blc->blm',w,age)/safe.squeeze(-1)
            parts.extend([avg,den/safe,ages.log1p()[...,None]])
        hh=hours.unfold(0,self.length,1).double()
        calendar=torch.stack([(hh%24*2*torch.pi/24).sin(),(hh%24*2*torch.pi/24).cos(),
                             (hh*2*torch.pi/(365.25*24)).sin(),(hh*2*torch.pi/(365.25*24)).cos()],-1).float()
        parts.append(calendar[:,:,None].expand(-1,-1,self.side**2,-1))
        return torch.cat(parts,-1)

    @torch.no_grad()
    def batch(self,anchors,context,development=False,temporal=True,features=True):
        validate_window(self.events.hours,anchors,max(self.length,12),self.split_start,self.split_stop)
        if not np.isin(context,self.train_ids).all():raise ValueError('condition contains outer-fold station')
        b=super().batch(anchors,context,development,temporal,features)
        if temporal:
            b['time_static']=self.time_static;b['calendar']=b['history'][:,-1,0,-4:]
            b['fusion_uv']=((self.grid_xy-self.lo)/(self.hi-self.lo)*2-1).reshape(1,*self.shape,2).expand(len(anchors),-1,-1,-1)
        if not development:
            allowed=torch.as_tensor(np.isin(np.arange(len(self.values[0])),self.train_ids),device='cuda')
            b['mask'] &= allowed[None]
        return b

    @torch.no_grad()
    def query_batch(self,anchors,context,batch=None):
        b=dict(self.batch(anchors,context) if batch is None else batch)
        point=[]
        for a in anchors:
            point.append(self.small_features(self.values[a-11:a+1,context],context,int(self.events.hours[a])))
        b['point']=torch.stack(point)
        b['uv']=self.small_uv.expand(len(anchors),-1,-1,-1)
        b['time_uv']=((self.small_grid-self.lo)/(self.hi-self.lo)*2-1)[None].expand(len(anchors),-1,-1)
        b['static']=self.small_static[None].expand(len(anchors),-1,-1)
        return b
