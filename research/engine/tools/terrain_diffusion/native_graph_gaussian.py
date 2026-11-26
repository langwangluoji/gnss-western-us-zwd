"""Conditional affine velocity over the entire query graph and seven-hour vector."""
import math
import torch
from torch import nn
from .native_weather_models import NativeDiffusion
from .native_process_models import clock_features


class GraphAffineVelocity(NativeDiffusion):
    def __init__(self,condition_dim=175,width=79):
        super().__init__(condition_dim,width,'joint')
        for name in ('input','local','filters','norm','output'):delattr(self,name)
        self.net=nn.Sequential(nn.Linear(condition_dim+5,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,308))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.alpha[-1]=0

    def denoise(self,x,c,k,U,eigen):
        clock=clock_features(k)[:,None].expand(-1,c.shape[1],-1)
        coefficients=self.net(torch.cat([c,clock],-1)).reshape(*c.shape[:2],2,154)
        h=x
        bands=[torch.ones_like(eigen),torch.exp(-eigen),torch.exp(-4*eigen)]
        for level in range(2):
            bias=coefficients[:,:,level,:7]
            matrix=coefficients[:,:,level,7:].reshape(*c.shape[:2],3,7,7)/math.sqrt(7)
            frequency=torch.einsum('qm,bqh->bmh',U,h)
            filtered=torch.stack([torch.einsum('qm,bmh->bqh',U,frequency*band[None,:,None]) for band in bands],2)
            mapped=bias+torch.einsum('bqlij,bqlj->bqi',matrix,filtered)
            h=x+mapped if level==0 else mapped
        return h
