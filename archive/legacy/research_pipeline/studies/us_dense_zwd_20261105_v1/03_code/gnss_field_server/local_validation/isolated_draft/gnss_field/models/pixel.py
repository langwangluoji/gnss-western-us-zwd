"""Faithful/joint Gaussian residual field with standalone pure backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from .legacy_core import ResidualModel

class ResidualField(ResidualModel):
    def __init__(self):
        super().__init__('det')
        self.output=nn.Sequential(nn.GroupNorm(8,32),nn.SiLU())
        self.mean_head=nn.Conv2d(32,1,3,padding=1)
        nn.init.zeros_(self.mean_head.weight);nn.init.zeros_(self.mean_head.bias)
        self.variance_head=nn.Sequential(nn.Linear(35,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias,.5413248546)

    def encode_condition(self,b):
        stations=b['stations'].masked_fill(b['padding'][...,None],0)
        tokens=self.station_encoder(stations)+self.amplitude(stations[...,:5])
        tokens=torch.cat([tokens,self.null_station.expand(len(stations),-1,-1)],1)
        positions=torch.cat([stations[...,10:13],stations.new_zeros(len(stations),1,3)],1)
        pad=torch.cat([b['padding'],~b['padding'].all(1,keepdim=True)],1)
        coarse=F.interpolate(self.coarse_encoder(b['coarse'].flatten(1,2)),size=(24,24),mode='bilinear',align_corners=False)
        maps=torch.cat([b['mu_condition'],b['baseline_condition'],b['maps'],coarse,b['domain']],1)
        geometry=b['maps'][:,[1,2,0]]
        return dict(maps=maps,domain=b['domain'],
            attn12=self.attn12.prepare(tokens,positions,pad,F.interpolate(geometry,size=(12,12),mode='bilinear',align_corners=False)),
            attn6=self.attn6.prepare(tokens,positions,pad,F.interpolate(geometry,size=(6,6),mode='bilinear',align_corners=False)))

    def forward(self,b):
        cond=self.encode_condition(b);zero=torch.zeros_like(b['domain'])
        features=self._backbone(zero,zero.new_zeros(len(zero)),cond)
        return self.mean_head(features),features

    def standard_deviation(self,features,geometry,scale,faithful):
        if faithful:features=features.detach()
        return (F.softplus(self.variance_head(torch.cat([features,geometry],-1)).squeeze(-1))+1e-4)*scale

def gaussian_losses(mean,target,variance,scale,faithful):
    e=(mean-target)/scale;q=variance/(scale*scale)
    mse=.5*e.square().mean()
    nll=.5*(q.log()+(e.detach() if faithful else e).square()/q).mean()
    return mse,nll
