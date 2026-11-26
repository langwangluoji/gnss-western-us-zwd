"""Projective GP-plus-latent fields, fitted with joint proper sample scores."""
import math
import torch
from torch import nn
from .native_weather import dct_matrix


def clock(theta,batch,device,dtype):
    t=torch.full((batch,),theta,device=device,dtype=dtype)
    return torch.stack([t/(math.pi/2),t.sin(),t.cos(),(2*t).sin(),(2*t).cos()],-1)


class QueryHead(nn.Module):
    def __init__(self,modes):
        super().__init__();self.modes=modes
        self.net=nn.Sequential(nn.Linear(267,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,modes+1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,c):
        lead=torch.arange(7,device=c.device,dtype=c.dtype)/6
        position=torch.stack([lead,lead.square()],-1)
        out=self.net(torch.cat([c[:,:,None].expand(-1,-1,7,-1),position[None,None].expand(*c.shape[:2],-1,-1)],-1))
        return (.5*torch.tanh(out[...,0])).exp(),.5*torch.tanh(out[...,1:])/math.sqrt(self.modes)


class AffineLatentVelocity(nn.Module):
    def __init__(self,modes):
        super().__init__();self.modes=modes
        self.net=nn.Sequential(nn.Linear(95,84),nn.SiLU(),nn.Linear(84,84),nn.SiLU(),nn.Linear(84,3*modes*modes))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        adjacency=torch.zeros(7,7)
        for i in range(7):
            for j in [i-1,i+1]:
                if 0<=j<7:adjacency[i,j]=1
        adjacency/=adjacency.sum(-1,keepdim=True)
        self.register_buffer('bands',torch.stack([torch.eye(7),adjacency,adjacency@adjacency]))

    def forward(self,x,c,theta):
        features=torch.cat([c,clock(theta,len(c),c.device,c.dtype)],-1)
        matrix=self.net(features).reshape(len(c),3,self.modes,self.modes)/math.sqrt(self.modes)
        values=torch.einsum('bdh,ljh->bldj',x,self.bands)
        return torch.einsum('blij,bljh->bih',matrix,values)


class OddLatentVelocity(nn.Module):
    def __init__(self,modes):
        super().__init__();width=64
        self.input=nn.Linear(modes+97,width)
        self.convolution=nn.ModuleList([nn.Conv1d(width,width,3,padding=1) for _ in range(2)])
        self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.output=nn.Linear(width,modes)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def net(self,x,c,theta):
        position=torch.arange(7,device=x.device,dtype=x.dtype)/6
        p=torch.stack([position,position.square()],-1)[None].expand(len(x),-1,-1)
        global_features=torch.cat([c,clock(theta,len(c),c.device,c.dtype)],-1)
        h=self.input(torch.cat([x.transpose(1,2),global_features[:,None].expand(-1,7,-1),p],-1))
        for convolution,norm in zip(self.convolution,self.norm):
            h=norm(h+torch.nn.functional.silu(convolution(h.transpose(1,2)).transpose(1,2)))
        return self.output(h).transpose(1,2)

    def forward(self,x,c,theta):return .5*(self.net(x,c,theta)-self.net(-x,c,theta))


class ProjectiveLatentFlow(nn.Module):
    def __init__(self,family='odd',domain='time',modes=8,steps=4):
        super().__init__()
        if family not in ('affine','odd') or domain not in ('time','frequency'):raise ValueError('unknown latent flow')
        self.family=family;self.domain=domain;self.steps=steps;self.modes=modes
        self.query=QueryHead(modes)
        self.flow=(AffineLatentVelocity(modes) if family=='affine' else OddLatentVelocity(modes))
        self.register_buffer('D',dct_matrix(7))

    def latent(self,noise,c):
        K,B,D,H=noise.shape
        x=noise.reshape(K*B,D,H);condition=c.repeat(K,1)
        original=x
        if self.domain=='frequency':x=x@self.D.T
        initial=x
        delta=-(math.pi/2)/self.steps
        for step in range(self.steps):
            theta=math.pi/2+step*delta;v=self.flow(x,condition,theta)
            proposal=x+delta*v
            x=x+.5*delta*(v+self.flow(proposal,condition,theta+delta))
        if self.domain=='frequency':x=original+(x-initial)@self.D
        return x.reshape(K,B,D,H)

    def forward(self,condition,global_condition,mean,std,gp_path,latent_noise):
        scale,phi=self.query(condition);g=self.latent(latent_noise,global_condition)
        low_rank=torch.einsum('bqhd,kbdh->kbqh',phi,g)
        return mean[None]+std[None]*(scale[None]*gp_path+low_rank)


def fair_crps(samples,target):
    K=len(samples)
    if K<2:raise ValueError('fair scoring requires at least2 IID members')
    ordered=samples.sort(0).values
    coefficient=2*torch.arange(1,K+1,device=samples.device,dtype=samples.dtype)-K-1
    spread=(ordered*coefficient.reshape(K,*([1]*(samples.ndim-1)))).sum(0)/(K*(K-1))
    return (samples-target[None]).abs().mean(0)-spread


def masked_mean(values,valid):
    return torch.where(valid,values,0).sum()/valid.sum().clamp_min(1)


def joint_score(samples,truth,mean,std,correlation,pairs):
    K,B,Q,H=samples.shape;valid=torch.isfinite(truth);safe=torch.where(valid,truth,0)
    marginal=masked_mean(fair_crps(samples,safe)/std.clamp_min(1),valid)
    index=torch.arange(Q*H,device=truth.device).reshape(Q,H)
    covariance=correlation[:,index[:,1:],index[:,:-1]]
    ds=(std[...,1:]**2+std[...,:-1]**2-2*covariance*std[...,1:]*std[...,:-1]).clamp_min(1).sqrt()
    dv=valid[...,1:]&valid[...,:-1]
    dy=torch.where(dv,truth[...,1:]-truth[...,:-1],0)
    increment=masked_mean(fair_crps(samples.diff(dim=-1),dy)/ds,dv)
    if len(pairs):
        a,b=pairs.unbind(-1);pv=valid[:,a]&valid[:,b]
        target=torch.where(pv,truth[:,a]-truth[:,b],0)
        covariance=correlation[:,index[a],index[b]]
        scale=(std[:,a]**2+std[:,b]**2-2*covariance*std[:,a]*std[:,b]).clamp_min(1).sqrt()
        pair=masked_mean(fair_crps(samples[:,:,a]-samples[:,:,b],target)/scale,pv)
    else:pair=samples.sum()*0
    normalized=(samples-mean[None])/std[None]
    target=torch.where(valid,(safe-mean)/std,0)
    difference=torch.where(valid[None],normalized-target[None],0).flatten(2)
    first=torch.linalg.vector_norm(difference,dim=-1).mean(0)
    members=torch.where(valid[None],normalized,0).permute(1,0,2,3).flatten(2)
    distance=torch.linalg.vector_norm(members[:,:,None]-members[:,None,:],dim=-1)
    energy=(first-distance.sum((1,2))/(2*K*(K-1)))/valid.flatten(1).sum(-1).clamp_min(1).sqrt()
    energy=masked_mean(energy,valid.flatten(1).any(-1))
    terms=dict(marginal=marginal,increment=increment,pair=pair,energy=energy)
    return marginal+.25*increment+.25*pair+.1*energy,terms
