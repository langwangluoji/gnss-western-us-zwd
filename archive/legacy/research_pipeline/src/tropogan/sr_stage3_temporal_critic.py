"""Conditional same-station temporal residual critic; no dense pseudo-truth."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import functional as F


class TemporalResidualCritic(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        sn=lambda a,b:spectral_norm(nn.Linear(a,b),n_power_iterations=1)
        # Two endpoints: legal23 + xy2; elapsed time1; residuals2; rate1.
        self.net=nn.Sequential(sn(54,width),nn.LeakyReLU(.2),sn(width,width),nn.LeakyReLU(.2),sn(width,1))

    def forward(self,a,b,samples):
        m=samples['valid'];dt=samples['dt']
        if not m.any() or not torch.isfinite(dt).all() or not (dt>0).all():
            raise ValueError('Positive elapsed time and observed temporal pairs required')
        ca,cb=samples['cond_a'],samples['cond_b']
        if ca.shape!=(*m.shape,23) or cb.shape!=ca.shape:
            raise ValueError('Temporal critic requires 23 legal channels per endpoint')
        # Conditions must not carry gradients or any target-derived residuals.
        if ca.requires_grad or cb.requires_grad:
            raise ValueError('Legal context must be a data-only tensor')
        ba=ca[...,5]*100*torch.exp(-ca[...,10]/3)
        bb=cb[...,5]*100*torch.exp(-cb[...,10]/3)
        ra=(a-ba)/10;rb=(b-bb)/10
        age=(dt[:,None]/24).expand_as(a)
        rate=(rb-ra)*6/dt[:,None]
        feature=torch.cat([ca,cb,samples['xy_a'],samples['xy_b'],age[...,None],
            ra[...,None],rb[...,None],rate[...,None]],dim=-1)
        if not torch.isfinite(feature[m]).all():raise ValueError('Nonfinite valid temporal feature')
        score=self.net(feature[m]).squeeze(-1)
        return score

    @contextmanager
    def frozen_for_generator(self):
        modes=[(m,m.training) for m in self.modules()]
        params=[(p,p.requires_grad) for p in self.parameters()]
        self.eval();self.requires_grad_(False)
        try:yield self
        finally:
            for m,mode in modes:m.training=mode
            for p,req in params:p.requires_grad_(req)


def discriminator_loss(real,fake):
    return F.relu(1-real).mean()+F.relu(1+fake).mean()


def generator_loss(fake):return -fake.mean()
