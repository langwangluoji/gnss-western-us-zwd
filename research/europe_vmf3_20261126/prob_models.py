"""European query conditions on the existing projective-GP / v-flow models."""
import torch
from torch import nn

from .features import HERE
from tools.terrain_diffusion.native_weather_gp import WeatherProjectiveGP
from tools.terrain_diffusion.native_process_models import LocalTrajectoryVelocity,AffineTrajectoryVelocity
from tools.terrain_diffusion.native_weather_models import NativeGaussian


class EuropeWeatherGP(WeatherProjectiveGP):
    def __init__(self,copula):
        super().__init__(265,copula)
        self.node[0]=nn.Linear(218+90+2,98)

    def state(self,c,summary,static,query_ids):
        if c.shape[-1]!=218 or summary.shape!=(len(c),90):raise ValueError('European218/90condition required')
        B,Q,_=c.shape
        if len(static)!=Q or len(query_ids)!=Q or len(torch.unique(query_ids))!=Q:raise ValueError('unique query identity required')
        condition=torch.cat([c,summary[:,None].expand(-1,Q,-1)],-1)
        lead=torch.arange(7,device=c.device,dtype=c.dtype)/6
        clock=torch.stack([lead,lead.square()],-1)
        output=self.node(torch.cat([condition[:,:,None].expand(-1,-1,7,-1),clock[None,None].expand(B,Q,-1,-1)],-1))
        base=static[:,:3].double()*static.new_tensor([1000/300,1000/300,2],dtype=torch.float64)
        psi=base[None,:,None]+3*torch.tanh(output[...,:3].double()/3)
        rho=self.rho_min+(self.rho_max-self.rho_min)*torch.sigmoid(self.rho_logit+output[...,3].double())
        return psi,rho,self.time_correlation(summary)


def gaussian():
    model=NativeGaussian(218,64)
    nn.init.zeros_(model.net[-1].weight);nn.init.zeros_(model.net[-1].bias)
    return model


def process(mode):
    if mode=='local':return LocalTrajectoryVelocity(308,111)
    if mode=='affine':return AffineTrajectoryVelocity(308,102)
    raise ValueError('registered local or affine process required')
