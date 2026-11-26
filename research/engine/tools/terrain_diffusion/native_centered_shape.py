"""Conditional Hermite shape/scale controls with exact zero residual mean."""
import math
import torch
from torch import nn


class CenteredShape(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ('scale','shape','both'):raise ValueError(mode)
        self.mode=mode;width={'scale':114,'shape':112,'both':110}[mode]
        outputs={'scale':7,'shape':14,'both':21}[mode]
        self.net=nn.Sequential(nn.Linear(265,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,outputs))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def coefficients(self,c):
        raw=self.net(c);zero=torch.zeros_like(raw[...,:7]);logscale=zero
        if self.mode=='scale':logscale=raw;b=zero;d=zero
        elif self.mode=='shape':b,d=raw.chunk(2,-1)
        else:logscale,b,d=raw.chunk(3,-1)
        b=b.tanh();d=d.tanh();a=(1+b.square()+d.square()).rsqrt()
        return a,a*b,a*d,logscale.clamp(-math.log(4),math.log(4)).exp()

    def transform(self,c,eta):
        a,b,d,mult=self.coefficients(c)
        return mult*(a*eta+b*(eta.square()-1)/math.sqrt(2)+d*(eta.pow(3)-3*eta)/math.sqrt(6))

    def moments(self,c):
        _,_,_,std=self.coefficients(c)
        return torch.zeros_like(std),std


def fair_crps_torch(samples,truth):
    K=len(samples)
    if K<2:raise ValueError('fair CRPS requires at least2 IID members')
    ordered=samples.sort(dim=0).values
    weight=(2*torch.arange(K,device=samples.device,dtype=samples.dtype)-K+1).reshape((K,)+(1,)*(samples.ndim-1))
    return (samples-truth[None]).abs().mean(0)-(ordered*weight).sum(0)/(K*(K-1))


def masked_loss(model,target,c,gs,members=16):
    valid=torch.isfinite(target);count=int(valid.sum())
    if not count:return next(model.parameters()).sum()*0,0
    eta=torch.randn(members,*target.shape,device=target.device,dtype=target.dtype)
    samples=gs[None]*model.transform(c,eta)
    score=fair_crps_torch(samples,torch.where(valid,target,0))
    return score[valid].mean(),count
