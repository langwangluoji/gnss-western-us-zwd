"""v12 pointwise foundations, full-point residual diffusion, spatial-fold features."""
import math
import numpy as np
import torch
from torch import nn
from .model import GeoAttention, schedule, recover


def spatial_folds(geo,groups):
    blocks=np.floor(np.asarray(geo)[:,:2]/150).astype(int)
    fold=(blocks[:,0]*73856093+blocks[:,1]*19349663)%3
    for group in np.unique(groups):
        idx=np.flatnonzero(groups==group);fold[idx]=fold[idx[0]]
    return fold


def masked_mse(pred,target,mask):
    return (pred[mask]-target[mask]).square().mean()


def common_shift(samples,correction):
    return samples+correction


def flat_features(static,history):
    return torch.cat([static,history[:,:,-1],history.mean(-2),history[:,:,-1]-history[:,:,0]],-1)


class Foundation(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='mlp':
            self.net=nn.Sequential(nn.Linear(113,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
        elif kind=='cnn_lstm':
            self.conv=nn.Sequential(nn.Conv1d(8,32,3,padding=1),nn.SiLU(),nn.Conv1d(32,32,3,padding=1),nn.SiLU())
            self.lstm=nn.LSTM(96,64,batch_first=True)
            self.net=nn.Sequential(nn.Linear(105,96),nn.SiLU(),nn.Linear(96,1))
        else:raise ValueError(kind)
    def forward(self,static,history):
        if self.kind=='mlp':return self.net(flat_features(static,history)).squeeze(-1)
        b,n,l,_=history.shape;x=history.reshape(b*n*l,3,8).transpose(1,2)
        x=self.conv(x).flatten(1).reshape(b*n,l,96);_,(h,_)=self.lstm(x)
        return self.net(torch.cat([static,h[-1].reshape(b,n,64)],-1)).squeeze(-1)


class GaussianResidual(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(114,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,2))
    def forward(self,f):
        p=self.net(f);return p[...,0],p[...,1].clamp(-5,2).exp()


class ResidualDenoiser(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(115,64);self.time=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,64))
        self.graph=nn.ModuleList([GeoAttention(64,16,300) for _ in range(3)])
        self.norm=nn.ModuleList([nn.LayerNorm(64) for _ in range(3)])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64)) for _ in range(3)])
        self.head=nn.Linear(64,1);self.register_buffer('ab',schedule())
    def forward(self,x,t,f,g):
        phase=t[:,None]*torch.exp(-math.log(10000)*torch.arange(16,device=x.device)/16)
        z=self.input(torch.cat([f,x[...,None]],-1))+self.time(torch.cat([phase.sin(),phase.cos()],-1))[:,None]
        for graph,norm,ff in zip(self.graph,self.norm,self.ff):
            h=norm(z);z=z+graph(h,h,g,g);z=z+ff(norm(z))
        return self.head(z).squeeze(-1)
    def sample(self,f,g,noise,steps=20):
        x=noise;times=torch.linspace(100,1,steps,device=x.device).round().long()
        for i,t in enumerate(times):
            v=self(x,t.expand(len(x)),f,g);clean,eps=recover(x,v,self.ab[t]);nxt=times[i+1] if i+1<len(times) else 0
            x=self.ab[nxt].sqrt()*clean+(1-self.ab[nxt]).sqrt()*eps
        return x


def features_from_batch(d,mean,std):
    # Each scale has raw mean/std/support, height-adjusted mean/std/support,T,P.
    c=d['context'][0].numpy();cg=d['cgeo'][0].numpy();qg=d['qgeo'][0].numpy()
    dist=np.linalg.norm(qg[:,None,:2]-cg[None,:,:2],axis=-1);dh=cg[None,:,2]-qg[:,None,2]
    items=[]
    for radius in (50,150,300):
        weights=np.exp(-.5*(dist/radius)**2)*(dist<=3*radius);frames=[]
        for t in range(12):
            valid=c[t,:,3]>0;w=weights*valid[None];den=w.sum(-1);wn=w/np.maximum(den[:,None],1e-8)
            raw=np.where(valid,c[t,:,0],0)[None,:];transport=((raw*std+mean)*np.exp(dh/3)-mean)/std
            vals=[]
            for value in (raw,transport):
                avg=(wn*value).sum(-1);sd=np.sqrt(np.maximum((wn*(value-avg[:,None])**2).sum(-1),0));vals.extend([avg,sd,np.log1p(den)])
            for j in (1,2):
                validw=c[t,:,j+3]>0;w=weights*validw[None];vals.append((w*c[t,None,:,j]).sum(-1)/np.maximum(w.sum(-1),1e-8))
            frames.append(np.stack(vals,-1))
        items.append(np.stack(frames,1))
    return d['query'][0].numpy(),np.concatenate(items,-1).astype('f4')


def spatial_context_query(data,rng,row,t,pool,hole,query_pool=None,nq=128):
    pool=np.asarray(pool);valid=pool[np.isfinite(data.a['train'][row,t,pool,0])]
    if query_pool is None:
        if hole:
            center=data.a['geo'][rng.choice(valid),:2];distance=np.linalg.norm(data.a['geo'][pool,:2]-center,axis=1)
            c=pool[np.argsort(distance)[-max(82,len(pool)//2):]];c=rng.choice(c,82,False)
        else:c=rng.choice(pool,82,False)
        eligible=valid[~np.isin(data.a['groups'][valid],data.a['groups'][c])]
    else:
        c=rng.choice(pool,82,False);eligible=np.asarray(query_pool)
        eligible=eligible[np.isfinite(data.a['train'][row,t,eligible,0])]
    q=rng.choice(eligible,min(nq,len(eligible)),False)
    # Cache fixed query dimension by padding with masked copies, never by duplicating loss weight.
    return c,q
