"""Weather-only Gaussian scale and covariance-matched Student degrees of freedom."""
import math
import torch
from torch import nn
from .native_student_process import log_density


class WeatherCalibrator(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ('scale','tail'):raise ValueError('unknown calibration mode')
        self.mode=mode
        self.net=nn.Sequential(nn.Linear(90,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        if mode=='tail':nn.init.constant_(self.net[-1].bias,math.log(17/980))

    def forward(self,weather):
        if weather.ndim!=2 or weather.shape[-1]!=90:raise ValueError('only registered90-dimensional history summary allowed')
        raw=self.net(weather).squeeze(-1)
        return raw.tanh().exp() if self.mode=='scale' else 3+997*raw.sigmoid()


def loss(radius,logdet,dimension,scale2,nu=None):
    valid=dimension>0;count=int(valid.sum())
    if not count:return scale2.sum()*0,0
    s=scale2.double()[valid];d=dimension[valid].double()
    if not torch.isfinite(s).all() or (s<=0).any():raise ValueError('positive finite conditional variance scale required')
    v=None if nu is None else nu[valid]
    value=-log_density(radius[valid]/s,logdet[valid]+d*s.log(),d,v)/d
    return value.mean(),count


def transform_ensemble(samples,mean,scale2,nu=None,seed=261128):
    if samples.ndim!=4 or samples.shape[1:]!=mean.shape or scale2.shape!=(len(mean),):
        raise ValueError('members x origins x queries x leads and origin-only scales required')
    if not torch.isfinite(scale2).all() or (scale2<=0).any():raise ValueError('positive finite variance scales required')
    if nu is None and torch.equal(scale2,torch.ones_like(scale2)):return samples.clone()
    factor=scale2.sqrt()[None,:,None,None].to(samples.dtype)
    if nu is not None:
        v=nu.to(samples.device).double()
        if v.shape!=scale2.shape or not torch.isfinite(v).all() or (v<=2).any():raise ValueError('finite covariance requires origin-only nu>2')
        torch.manual_seed(seed)
        u=2*torch.distributions.Gamma(v/2,torch.ones_like(v)).sample((len(samples),))
        factor=factor*((v[None]-2)/u).sqrt().to(samples.dtype)[:,:,None,None]
    return mean[None]+(samples-mean[None])*factor
