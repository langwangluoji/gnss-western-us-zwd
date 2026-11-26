"""Capacity-matched local nonlinear and conditional-Gaussian trajectory flows."""
import math
import torch
from torch import nn
from .native_weather_models import NativeDiffusion
from .native_process_ops import heun0


def clock_features(k):
    return torch.stack([k,k.sin(),k.cos(),(k*math.pi).sin(),(k*math.pi).cos()],-1)


class LocalTrajectoryVelocity(NativeDiffusion):
    def __init__(self,condition_dim=175,width=135):
        super().__init__(condition_dim,width)
        for name in ('input','local','filters','norm','output'):delattr(self,name)
        self.net=nn.Sequential(nn.Linear(condition_dim+12,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,7))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.alpha[-1]=0

    def denoise(self,x,c,k,U=None,eigen=None):
        clock=clock_features(k)[:,None].expand(-1,x.shape[1],-1)
        return self.net(torch.cat([x,c,clock],-1))


class AffineTrajectoryVelocity(NativeDiffusion):
    def __init__(self,condition_dim=175,width=124):
        super().__init__(condition_dim,width)
        for name in ('input','local','filters','norm','output'):delattr(self,name)
        self.net=nn.Sequential(nn.Linear(condition_dim+5,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,56))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.alpha[-1]=0

    def coefficients(self,c,k):
        clock=clock_features(k)[:,None].expand(-1,c.shape[1],-1)
        coefficient=self.net(torch.cat([c,clock],-1))
        return coefficient[...,:7],coefficient[...,7:].reshape(*c.shape[:2],7,7)/math.sqrt(7)

    def denoise(self,x,c,k,U=None,eigen=None):
        bias,matrix=self.coefficients(c,k)
        return bias+torch.einsum('bqij,bqj->bqi',matrix,x)

    @torch.no_grad()
    def affine_map(self,c,steps=20,solver='heun0'):
        # Propagate zero and the seven basis vectors through the same discrete map.
        # This computes its exact Gaussian mean / linear factor, not a MC estimate.
        B,Q,_=c.shape
        basis=torch.cat([torch.zeros(1,7,device=c.device),torch.eye(7,device=c.device)])
        initial=basis[:,None,None].expand(-1,B,Q,-1).reshape(8*B,Q,7)
        condition=c.repeat(8,1,1)
        result=(heun0(self,condition,initial,steps=steps) if solver=='heun0' else
            self.sample(condition,initial,None,None,steps,solver)).reshape(8,B,Q,7)
        mean=result[0];matrix=(result[1:]-mean[None]).permute(1,2,3,0)
        return mean,matrix


def make_process_model(mode,condition_dim=175):
    if mode=='local':return LocalTrajectoryVelocity(condition_dim)
    if mode=='affine':return AffineTrajectoryVelocity(condition_dim)
    if mode=='full':
        model=NativeDiffusion(condition_dim,64,'joint');model.alpha[-1]=0;return model
    raise ValueError(mode)
