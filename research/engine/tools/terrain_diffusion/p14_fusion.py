"""p14 warm-start carrier + coast/terrain edges + causal temporal residual."""
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree

from .fixed_support import FixedSupportModel, masked_multiscale_summary
from .temporal_encoders import TemporalFoundation
from .temporal_experiment import Events
from .unet_foundation import Features, GEOMETRY
from .data import DEFAULT_DATA, static_geometry
from .coast_features import load_coast, distance_to_coast


def chronological_blocks(start, stop, batch_hours):
    return [np.arange(a,min(a+batch_hours,stop)) for a in range(start,stop,batch_hours)]


class P14Fusion(nn.Module):
    def __init__(self,kind,length=72):
        super().__init__()
        self.kind=kind
        self.base=FixedSupportModel()
        if kind!='p14':
            self.edge=nn.Sequential(nn.Linear(10,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
            self.film=nn.Linear(8,32)
            self.spatial_head=nn.Linear(16,1)
            self.point_head=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
            nn.init.zeros_(self.spatial_head.weight);nn.init.zeros_(self.spatial_head.bias)
            nn.init.zeros_(self.point_head[-1].weight);nn.init.zeros_(self.point_head[-1].bias)
        if kind not in ('p14','features'):
            self.time=TemporalFoundation(kind,length)
            # Temporal output contributes only a spatial field; keep the same
            # point/static correction in every coast-feature arm.
            for p in self.time.query_head.parameters():p.requires_grad_(False)

    def forward(self,b):
        result=self.base(b['point'],b['support'],b['uv'])
        if self.kind=='p14':return result
        features=self.edge(b['edges'])
        weight=b['edge_mask']
        features=(features*weight[...,None]).sum(-2)/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        gamma,beta=self.film(b['grid_static']).chunk(2,-1)
        features=features*(1+gamma[None])+beta[None]
        field=self.spatial_head(features).transpose(1,2).reshape(len(result),1,*b['shape'])
        sampled=F.grid_sample(field,b['uv'],padding_mode='border',align_corners=True).flatten(1)
        result=result+sampled+self.point_head(b['static']).squeeze(-1)
        if self.kind not in ('p14','features'):
            # Existing temporal carrier has six heads; only the first is used
            # for the current-frame residual, not a future forecast in this task.
            correction=self.time(b['history'],b['time_uv'],b['static'],
                                 background=torch.zeros_like(result))[:,0]
            result=result+correction*b.get('time_scale',1.)
        return result


class FusionData:
    def __init__(self,path,length=72):
        import pandas as pd
        self.events=Events(path,max_history=length) # validates all registered hashes
        e=self.events;self.length=length
        self.values=torch.as_tensor(np.array(e.values),device='cuda')
        positions=torch.arange(len(self.values),device='cuda')[:,None]
        last=torch.cummax(torch.where(torch.isfinite(self.values[:,:,0]),positions,-1),dim=0).values
        self.station_age=(positions-last).float()
        del last,positions
        g=dict(np.load(GEOMETRY));self.shape=g['height'].shape
        with np.load(DEFAULT_DATA/'data.npz') as old:
            self.mean=old['mean'];self.std=old['std']
            self.old_geo=old['geo'];self.old_static=old['static']
        namespace=SimpleNamespace(mean=self.mean,std=self.std,a={'geo':self.old_geo})
        st,geo=static_geometry(pd.DataFrame(dict(x=g['x'].ravel(),y=g['y'].ravel(),height=g['height'].ravel())))
        self.grid_features=Features(namespace,st,geo)
        self.train_features=Features(namespace,e.geo['static'],self.old_geo)
        dev_geo=np.c_[e.geo['dev_xy'],e.geo['dev_height']/1000,e.geo['dev_static'][:,5]].astype('float32')
        self.dev_features=Features(namespace,e.geo['dev_static'],dev_geo)
        self.origin=torch.as_tensor(g['origin']/1000+12.5,device='cuda',dtype=torch.float32)
        self.grid_xy=self.grid_features.geo[:,:2]
        coast_path=Path(path)/'p14_grid_coast.npy'
        if not coast_path.exists():
            lines,_=load_coast(Path(path)/'coast_source')
            np.save(coast_path,distance_to_coast(geo[:,:2],lines))
        self.coast=torch.as_tensor(e.geo['coast'],device='cuda')
        self.grid_coast=torch.as_tensor(np.load(coast_path),device='cuda')
        self.grid_static=torch.cat([self.grid_features.static,torch.log1p(self.grid_coast[:,None])/6],-1)
        self.small_grid=torch.as_tensor(e.grid,device='cuda',dtype=torch.float32)
        self.lo=torch.as_tensor(e.lo,device='cuda',dtype=torch.float32)
        self.hi=torch.as_tensor(e.hi,device='cuda',dtype=torch.float32)
        self.hours=torch.as_tensor(e.hours,device='cuda')
        self.mean_t=torch.as_tensor(self.mean,device='cuda');self.std_t=torch.as_tensor(self.std,device='cuda')

    def temporal_history(self,raw,hours,c,batch,full_age):
        # raw contains L+B-1 hours, with fixed context IDs across the block.
        valid=torch.isfinite(raw)
        val=raw.clone()
        val[:,:,0]*=torch.exp(self.train_features.cg[c,2]/3)[None]
        val=torch.where(valid,(val-torch.tensor([0,280,900],device='cuda'))/torch.tensor([100,20,100],device='cuda'),0)
        val=val.unfold(0,self.length,1).permute(0,3,1,2)
        mask=valid.unfold(0,self.length,1).permute(0,3,1,2).float()
        age=full_age.unfold(0,self.length,1).permute(0,2,1)
        d=torch.cdist(self.small_grid,self.train_features.cg[c,:2])
        pieces=[]
        for radius in (50,150,300):
            w=torch.exp(-.5*(d/radius).square())*(d<=3*radius)
            den=torch.einsum('mc,blcj->blmj',w,mask)
            avg=torch.einsum('mc,blcj->blmj',w,val)/den.clamp_min(1e-8)
            coverage=den/w.sum(1)[None,None,:,None].clamp_min(1e-8)
            ages=torch.einsum('mc,blc->blm',w,age)/w.sum(1)[None,None].clamp_min(1e-8)
            pieces.extend([avg,coverage,ages.log1p()[...,None]])
        hh=hours.unfold(0,self.length,1).double()
        t=torch.stack([(hh%24*2*torch.pi/24).sin(),(hh%24*2*torch.pi/24).cos(),
                       (hh*2*torch.pi/(365.25*24)).sin(),(hh*2*torch.pi/(365.25*24)).cos()],-1).float()
        pieces.append(t[:,:,None].expand(-1,-1,64,-1))
        return torch.cat(pieces,-1)

    @torch.no_grad()
    def batch(self,anchors,context,development=False,temporal=True,features=True):
        e=self.events;c=torch.as_tensor(context,device='cuda')
        f=self.dev_features if development else self.train_features
        start,end=int(anchors[0]),int(anchors[-1])
        assert np.array_equal(anchors,np.arange(start,end+1)), 'time must be consecutive'
        raw=self.values[start-self.length+1:end+1,c]
        point=[];support=[];ys=[]
        for a in anchors:
            r=self.values[int(a)-11:int(a)+1,c]
            point.append(f(r,context,int(e.hours[a])))
            # Preserve p14 checkpoint feature units exactly, including its
            # historical height-summary convention; new edge high differences
            # below explicitly use kilometres / 1 km, rather than that helper.
            g=self.grid_features.geo;cg=self.train_features.cg[c]
            s=masked_multiscale_summary(g[:,:2],g[:,2],cg[:,:2],cg[:,2],r[:,:,0],torch.isfinite(r[:,:,0]))
            support.append(s.T.reshape(30,*self.shape))
            if development:
                i=np.searchsorted(e.dev_hours,e.hours[a]);ys.append(e.dev_values[i,:,0])
            else:ys.append(e.values[a,:,0])
        n=len(f.geo);uv=(f.geo[:,:2]-self.origin)/25;h,w=self.shape
        uv=torch.stack([2*uv[:,0]/(w-1)-1,2*uv[:,1]/(h-1)-1],-1)[None,None].expand(len(anchors),-1,-1,-1)
        coast=e.geo['dev_coast' if development else 'coast']
        st=torch.cat([f.static,torch.as_tensor(np.c_[np.log1p(coast)/6,coast/500],device='cuda',dtype=torch.float32)],-1)
        result=dict(point=torch.stack(point),support=torch.stack(support),uv=uv,shape=self.shape,
                    static=st[None].expand(len(anchors),-1,-1),grid_static=self.grid_static,
                    time_uv=((f.geo[:,:2]-self.lo)/(self.hi-self.lo)*2-1)[None].expand(len(anchors),-1,-1),
                    time_scale=100/float(self.std[0]))
        if features:
            dist=torch.cdist(self.grid_xy,self.train_features.cg[c,:2])
            dd,nnidx=dist.topk(min(8,len(c)),largest=False)
            ds=self.grid_features.static[:,None]-self.train_features.static[c[nnidx]]
            edge=torch.cat([ds[:,:,:2]*1000/300,dd[:,:,None].log1p()/6,ds[:,:,2:6],
                ((self.grid_coast[:,None]-self.coast[c[nnidx]])/500)[:,:,None]],-1)
            # 2 xy + log(distance) + height/slopeX/slopeY/rough + coast = 8.
            # Retain 7 geometry channels by combining the two slope components.
            edge=torch.cat([edge[:,:,:4],torch.linalg.norm(edge[:,:,4:6],dim=-1,keepdim=True),edge[:,:,6:]],-1)
            currents=self.values[torch.as_tensor(anchors,device='cuda')[:,None],c[None]]
            norm=torch.where(torch.isfinite(currents),(currents-self.mean_t)/self.std_t,0)
            observed=norm[:,nnidx]
            result['edges']=torch.cat([edge[None].expand(len(anchors),-1,-1,-1),observed],-1)
            result['edge_mask']=torch.isfinite(currents[:,:,0])[:,nnidx].float()/dd.clamp_min(5)[None].square()
        if temporal:
            result['history']=self.temporal_history(raw,self.hours[start-self.length+1:end+1],c,len(anchors),
                self.station_age[start-self.length+1:end+1,c])
        y=torch.as_tensor(np.stack(ys),device='cuda')
        mask=torch.isfinite(y)
        if not development:
            hidden=~np.isin(e.geo['groups'],e.geo['groups'][context])
            mask &= torch.as_tensor(hidden,device='cuda')[None]
        result['target']=torch.nan_to_num((y-self.mean_t[0])/self.std_t[0]);result['mask']=mask
        return result
