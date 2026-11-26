"""European coordinate/data adapter for the original p14+GRU carrier."""
import sys
import numpy as np
import torch
from torch import nn
from .data import ENGINE

sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.p14_fusion import P14Fusion


class EuropeBase(nn.Module):
    def __init__(self):
        super().__init__();self.carrier=P14Fusion('gru',16)

    def forward(self,b):
        # Known spatial interpolation prior plus the original trainable carrier.
        return b['prior']+self.carrier(b)*100


class FieldData:
    def __init__(self,a,device='cuda'):
        self.a=a;self.device=device
        self.values=torch.tensor(a['values'],device=device)
        self.hours=torch.tensor(a['hours'],device=device,dtype=torch.float64)
        self.static=torch.tensor(a['static'],device=device)
        self.grid_static=torch.tensor(a['grid_static'],device=device)
        self.xy=self.static[:,:2]*1000;self.grid_xy=self.grid_static[:,:2]*1000
        self.lo=torch.tensor(a['lo'],device=device,dtype=torch.float32)
        self.hi=torch.tensor(a['hi'],device=device,dtype=torch.float32)
        self.shape=tuple(a['grid_shape']);self.length=16
        positions=torch.arange(len(self.values),device=device)[:,None,None]
        last=torch.cummax(torch.where(torch.isfinite(self.values),positions,-100000),0).values
        self.ages=((positions-last)*6).float().clamp_max(10000)
        # 8x8 carrier retains the actual original GRU topology.
        h,w=self.shape;iy=torch.linspace(0,h-1,8,device=device).round().long();ix=torch.linspace(0,w-1,8,device=device).round().long()
        self.small_index=(iy[:,None]*w+ix[None]).flatten()
        self.all_static=torch.cat([self.static,self.grid_static])

    def pooled(self,r,valid,qstatic,cstatic):
        distance=torch.cdist(qstatic[:,:2]*1000,cstatic[:,:2]*1000)
        norm=(torch.nan_to_num(r)-r.new_tensor([0,280,900]))/r.new_tensor([100,20,100])
        norm=torch.where(valid,norm,0)
        common=norm[...,0]*torch.exp(cstatic[:,2])[None,None]
        pieces=[];weather=[];summaries=[]
        for radius in (50,150,300):
            weight=torch.exp(-.5*(distance/radius).square())*(distance<=3*radius)
            den=torch.einsum('qc,blcv->blqv',weight,valid.float());safe=den.clamp_min(1e-8)
            avg=torch.einsum('qc,blcv->blqv',weight,norm)/safe
            var=(torch.einsum('qc,blc->blq',weight,norm[...,0].square())/safe[...,0]-avg[...,0].square()).clamp_min(0)
            tm=torch.einsum('qc,blc->blq',weight,common)/safe[...,0]
            ts=(torch.einsum('qc,blc->blq',weight,common.square())/safe[...,0]-tm.square()).clamp_min(0).sqrt()
            factor=torch.exp(-qstatic[:,2])[None,None]
            pieces.append(torch.stack([avg[...,0],var.sqrt(),den[...,0].log1p(),tm*factor,ts*factor,den[...,0].log1p(),avg[...,1],avg[...,2]],-1))
            weather.extend([avg[:,-1,:,1],avg[:,-1,:,2],den[:,-1,:,1].log1p(),den[:,-1,:,2].log1p()])
            summaries.append((weight,avg,den))
        return torch.cat(pieces,-1),torch.stack(weather,-1),distance,summaries

    @torch.no_grad()
    def batch(self,anchors,context,query):
        anchors=np.asarray(anchors)
        if not np.all(np.diff(anchors)==1):raise ValueError('training origins must be consecutive')
        if anchors[0]<15:raise ValueError('history before dataset')
        at=torch.as_tensor(anchors,device=self.device);c=torch.as_tensor(context,device=self.device);q=torch.as_tensor(query,device=self.device)
        index=at[:,None]-torch.arange(15,-1,-1,device=self.device)[None]
        r=self.values[index[:,:,None],c[None,None]];valid=torch.isfinite(r)
        qs=self.static[q];cs=self.static[c];B=len(at);Q=len(q)
        history,weather,distance,_=self.pooled(r,valid,qs,cs)
        phase=qs[:,:2,None]*1000/r.new_tensor([50,100,200,400])*2*torch.pi
        fourier=torch.cat([phase.sin(),phase.cos()],-1).reshape(Q,16)
        hours=self.hours[at]
        clock=torch.stack([(hours%24*2*torch.pi/24).sin(),(hours%24*2*torch.pi/24).cos(),(hours*2*torch.pi/(365.25*24)).sin(),(hours*2*torch.pi/(365.25*24)).cos()],-1).float()
        near=distance[None].masked_fill(~valid[:,-1,:,0][:,None],float('inf')).amin(-1).clamp_max(600)
        count=((distance[None]<150)&valid[:,-1,:,0][:,None]).sum(-1).float().log1p()
        point=torch.cat([qs[None,:,:7].expand(B,-1,-1),fourier[None].expand(B,-1,-1),clock[:,None].expand(-1,Q,-1),weather,near[...,None]/100,count[...,None],history[:,-1],history.mean(1),history[:,-1]-history[:,0]],-1)
        # Coarse fixed support summarizes recent available observations.
        g=self.grid_static;gd=torch.cdist(self.grid_xy,cs[:,:2]*1000)
        available=valid[:,-12:,:,0].any(1)
        positions=torch.arange(12,device=self.device)[None,:,None]
        last=torch.where(valid[:,-12:,:,0],positions,-1).amax(1)
        latest=r[:,-12:,:,0].gather(1,last.clamp_min(0)[:,None]).squeeze(1)/100
        latest=torch.where(available,latest,0)
        support=[]
        dh=(cs[None,:,2]-g[:,None,2])*3000
        gn=gd[None].masked_fill(~available[:,None],float('inf')).amin(-1).clamp_max(600)
        for radius in (50,150,300):
            w=(torch.exp(-.5*(gd/radius).square())*(gd<=3*radius))[None]*available[:,None]
            den=w.sum(-1);safe=den.clamp_min(1e-8)
            avg=(w*latest[:,None]).sum(-1)/safe
            sd=((w*(latest[:,None]-avg[...,None]).square()).sum(-1)/safe).clamp_min(0).sqrt()
            transported=latest[:,None]*torch.exp(dh[None]/3000)
            ta=(w*transported).sum(-1)/safe;cnt=(w>0).sum(-1).float()
            hd=(w*dh.abs()).sum(-1)/safe;hs=((w*(dh.abs()-hd[...,None]).square()).sum(-1)/safe).clamp_min(0).sqrt()
            support.append(torch.stack([avg,sd,den.log1p(),ta,gn/100,cnt.log1p(),hd/1000,hs/1000,(den>0).float(),(cnt>0).float()],-1))
        support=torch.cat(support,-1).transpose(1,2).reshape(B,30,*self.shape)
        uv=((qs[:,:2]*1000-self.lo)/(self.hi-self.lo)*2-1)[None,None].expand(B,1,-1,-1)
        # Coast/terrain condition edges retain direction and height differences.
        dd,nnidx=gd.topk(min(8,len(c)),largest=False);delta=g[:,None]-cs[nnidx]
        edge=torch.cat([delta[...,:2]*1000/300,dd[...,None].log1p()/6,delta[...,2:3]*3,delta[...,3:5].norm(dim=-1,keepdim=True),delta[...,5:6],delta[...,8:9]],-1)
        norm=torch.where(valid[:,-1],(r[:,-1]-r.new_tensor([0,280,900]))/r.new_tensor([100,20,100]),0)
        edges=torch.cat([edge[None].expand(B,-1,-1,-1),norm[:,nnidx]],-1)
        edge_mask=valid[:,-1,:,0][:,nnidx].float()/dd.clamp_min(5)[None].square()
        # Original GRU's multiscale25 channels at64 support locations.
        small=g[self.small_index];sdist=torch.cdist(small[:,:2]*1000,cs[:,:2]*1000)
        normal=torch.where(valid,(r-r.new_tensor([0,280,900]))/r.new_tensor([100,20,100]),0)
        normal[...,0]*=torch.exp(cs[:,2])[None,None]
        ages=self.ages[index[:,:,None],c[None,None],0]
        pieces=[]
        for radius in (50,150,300):
            w=torch.exp(-.5*(sdist/radius).square())*(sdist<=3*radius)
            den=torch.einsum('mc,blcv->blmv',w,valid.float())
            avg=torch.einsum('mc,blcv->blmv',w,normal)/den.clamp_min(1e-8)
            cov=den/w.sum(-1)[None,None,:,None].clamp_min(1e-8)
            age=torch.einsum('mc,blc->blm',w,ages)/w.sum(-1)[None,None].clamp_min(1e-8)
            pieces.extend([avg,cov,age.log1p()[...,None]])
        hh=self.hours[index]
        times=torch.stack([(hh%24*2*torch.pi/24).sin(),(hh%24*2*torch.pi/24).cos(),(hh*2*torch.pi/(365.25*24)).sin(),(hh*2*torch.pi/(365.25*24)).cos()],-1).float()
        pieces.append(times[:,:,None].expand(-1,-1,64,-1))
        target=self.values[at[:,None],q[None],0]
        return dict(point=point,support=support,uv=uv,shape=self.shape,static=qs[None].expand(B,-1,-1),grid_static=g[:,:8],history=torch.cat(pieces,-1),time_uv=uv[:,0],time_scale=1.,edges=edges,edge_mask=edge_mask,prior=point[:,:,52]*100,target=target,mask=torch.isfinite(target),nearest=near)
