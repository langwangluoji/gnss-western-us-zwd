"""Free-grid residual generators sharing a frozen coherent Gaussian starting point."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .grid_processor import ResidualBlock
from .diffusion import heun_sample,edm_coefficients


def gaussian_prior_field(phi,v,latent,remainder,shape):
    if phi.ndim!=2 or v.ndim!=2 or phi.shape[1]!=v.shape[1]:
        raise ValueError('Frozen spatial factor dimensions differ')
    coefficients=latent@v+remainder-(remainder@v.T)@v
    return (coefficients@phi.T).reshape(*latent.shape[:2],1,*shape)


class ConditionalUNet(nn.Module):
    def __init__(self,widths=(32,64,128,128),timed=False):
        super().__init__(); self.timed=timed
        self.encoders=nn.ModuleList([ResidualBlock(71,widths[0])]+
            [ResidualBlock(w,w) for w in widths[1:]])
        self.down=nn.ModuleList([nn.Conv2d(widths[i],widths[i+1],3,stride=2,padding=1) for i in range(3)])
        self.decoders=nn.ModuleList([ResidualBlock(widths[i+1]+widths[i],widths[i]) for i in (2,1,0)])
        if timed:
            self.register_buffer('frequencies',torch.exp(-torch.arange(16,dtype=torch.float32)*.5))
            self.time_embedding=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128))
            self.film=nn.ModuleList([nn.Linear(128,2*w) for w in (*widths,*reversed(widths[:-1]))])
        self.output=nn.Conv2d(widths[0],1,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,condition,prior,state,sigma=None):
        if condition.ndim!=4 or condition.shape[1]!=69 or prior.shape!=state.shape:
            raise ValueError('Expected spatial conditions69 and matching one-channel fields')
        x=torch.cat([condition,prior,state],1); embedding=None
        if self.timed:
            sigma=torch.as_tensor(sigma,device=x.device,dtype=x.dtype).reshape(-1)
            if sigma.numel()==1: sigma=sigma.expand(len(x))
            if sigma.shape!=(len(x),) or (sigma<=0).any(): raise ValueError('Invalid noise levels')
            phase=(sigma.log()/4)[:,None]*self.frequencies[None]
            embedding=self.time_embedding(torch.cat([phase.sin(),phase.cos()],1))
        def mod(value,index):
            if embedding is None: return value
            scale,shift=self.film[index](embedding).chunk(2,1)
            return value*(1+.1*torch.tanh(scale[:,:,None,None]))+.1*shift[:,:,None,None]
        x=mod(self.encoders[0](x),0); levels=[x]
        for i,(down,block) in enumerate(zip(self.down,self.encoders[1:]),1):
            x=mod(block(down(x)),i); levels.append(x)
        for i,(block,skip) in enumerate(zip(self.decoders,reversed(levels[:-1])),4):
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
            x=mod(block(torch.cat([x,skip],1)),i)
        return self.output(x)


class ResidualGenerator(nn.Module):
    def __init__(self,kind,widths=(32,64,128,128),scale_mm=10.,steps=18,recompute=True):
        super().__init__()
        if kind not in ('stochastic','gan','diffusion'): raise ValueError('Unknown generator kind')
        self.kind,self.scale_mm,self.steps,self.recompute=kind,float(scale_mm),steps,recompute
        self.net=ConditionalUNet(widths,timed=kind=='diffusion')
        with torch.no_grad():
            gain=heun_sample(lambda x,s:x/(1+s*s),torch.ones(1,1,dtype=torch.float64),steps=steps)[0,0]
        self.register_buffer('gaussian_gain',gain)

    def denoise(self,state,sigma,condition,prior):
        sigma=torch.as_tensor(sigma,device=state.device,dtype=state.dtype)
        skip,out,inp,_,_=edm_coefficients(sigma,sigma_data=1.)
        return skip*state+out*self.net(condition,prior/self.scale_mm,state*inp,sigma)

    def sample_residual(self,condition,prior,white):
        if not torch.isfinite(condition).all() or not torch.isfinite(prior).all() or not torch.isfinite(white).all():
            raise ValueError('Nonfinite generator inputs')
        if self.kind!='diffusion':
            return prior+self.scale_mm*self.net(condition,prior/self.scale_mm,white)
        def denoise(x,s):
            if self.recompute and torch.is_grad_enabled():
                return checkpoint(self.denoise,x,s,condition,prior,use_reentrant=False)
            return self.denoise(x,s,condition,prior)
        sample=heun_sample(denoise,white,steps=self.steps)
        change=sample/self.gaussian_gain.to(dtype=sample.dtype)-white
        return prior+self.scale_mm*change


class ObservationCritic(nn.Module):
    """Shared point encoder with invariant masked moments; no grid truth required."""
    def __init__(self,width=64):
        super().__init__()
        self.points=nn.Sequential(nn.Linear(73,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,residual,coordinates,condition,mask):
        if residual.ndim!=2 or mask.shape!=residual.shape or condition.shape!=(*residual.shape,69):
            raise ValueError('Critic station dimensions differ')
        if (mask.sum(1)==0).any(): raise ValueError('Empty critic target population')
        if coordinates.ndim==2: coordinates=coordinates[None].expand(len(residual),-1,-1)
        values=torch.cat([residual[:,:,None],coordinates,condition],-1)
        if not torch.isfinite(values[mask]).all(): raise ValueError('Nonfinite active critic input')
        values=torch.where(mask[:,:,None],values,0.)
        h=self.points(values)*mask[:,:,None]
        count=mask.sum(1).clamp_min(1)[:,None]
        average=h.sum(1)/count; square=h.square().sum(1)/count
        return self.readout(torch.cat([average,square],1)).squeeze(-1)


class GenerationSystem(nn.Module):
    def __init__(self,kind,**kwargs):
        super().__init__(); self.generator=ResidualGenerator(kind,**kwargs)
        self.critic=ObservationCritic() if kind=='gan' else None


def critic_gradient_penalty(critic,real,fake,coordinates,condition,mask):
    alpha=torch.rand(len(real),1,device=real.device,dtype=real.dtype)
    mixed=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    score=critic(mixed,coordinates,condition,mask)
    gradient=torch.autograd.grad(score.sum(),mixed,create_graph=True)[0]*mask
    return (gradient.norm(dim=1)-1).square().mean()
