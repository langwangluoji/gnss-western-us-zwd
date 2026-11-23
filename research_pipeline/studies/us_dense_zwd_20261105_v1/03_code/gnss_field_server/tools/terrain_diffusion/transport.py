"""Condition-only observation summaries; direct ZWD outputs remain unchanged."""
import math
import torch
from torch import nn
from .model import DiffusionModel


class ObservationSummary(nn.Module):
    def __init__(self,variant,mean,std):
        super().__init__();self.variant=variant
        self.register_buffer('mean',torch.tensor(float(mean)));self.register_buffer('std',torch.tensor(float(std)))
        if variant=='transport':
            self.scale_logit=nn.Parameter(torch.tensor(math.log(2/3)))
            self.edge_value=nn.Sequential(nn.Linear(6,16),nn.SiLU(),nn.Linear(16,1))
            nn.init.zeros_(self.edge_value[-1].weight);nn.init.zeros_(self.edge_value[-1].bias)

    def forward(self,d):
        cg,qg=d['cgeo'],d['qgeo'];dist=torch.cdist(qg[...,:2],cg[...,:2])
        dh=cg[:,None,:,2]-qg[:,:,None,2];rough=cg[:,None,:,3]-qg[:,:,None,3]
        features=[]
        for offset in (-1,-4,-7):
            context=d['context'][:,offset];valid=context[...,3]>0
            raw=torch.where(valid,context[...,0]*self.std+self.mean,0.)[:,None,:].expand_as(dist)
            if self.variant=='transport':
                scale=1+5*self.scale_logit.sigmoid()
                transformed=raw*torch.exp(dh/scale)
                weather=context[:,None,:,1:3].expand(*dist.shape,2)
                edge=torch.cat([dist[...,None]/300,dh[...,None],rough[...,None],weather,(raw/self.std)[...,None]],-1)
                value=(transformed-self.mean)/self.std+.1*self.edge_value(edge).squeeze(-1).tanh()
            else:value=(raw-self.mean)/self.std
            value=torch.where(valid[:,None,:],value,0.)
            for radius in (50,150,300):
                w=torch.exp(-.5*(dist/radius).square())*(dist<=3*radius)*valid[:,None,:]
                den=w.sum(-1);norm=w/den.clamp_min(1e-8)[...,None]
                avg=(norm*value).sum(-1)
                variance=(norm*(value-avg[...,None]).square()).sum(-1)
                sd=variance.clamp_min(1e-8).sqrt()*(den>0)
                features.extend([avg,sd,den.log1p()])
        return torch.stack(features,-1)


class TransportModel(DiffusionModel):
    def __init__(self,variant,mean,std,width=64,blocks=4):
        super().__init__(width,blocks);self.variant=variant
        if variant!='base':
            self.summary=ObservationSummary(variant,mean,std)
            self.query=nn.Sequential(nn.Linear(69,width),nn.SiLU(),nn.Linear(width,width))

    def features(self,x,step,d,encoded=None):
        if self.variant!='base':
            d=dict(d);summary=d['observation_features'] if 'observation_features' in d else self.summary(d)
            d['query']=torch.cat([d['query'],summary],-1)
        return super().features(x,step,d,encoded)

    def sample(self,d,noise,steps=20,checkpoint=False):
        if self.variant!='base':
            d=dict(d);d['observation_features']=self.summary(d)
        return super().sample(d,noise,steps,checkpoint)


def transfer_conditioner(model,state):
    prefixes=('context.','spatial.','temporal.','summary.','query.')
    model.load_state_dict({k:v for k,v in state.items() if k.startswith(prefixes)},strict=False)
