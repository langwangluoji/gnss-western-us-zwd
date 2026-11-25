"""One innovation state/noise chain, optionally denoised in space and graph spectra."""
import math
import torch
from torch import nn
from .masked_transformer_fusion import CausalSpatialEncoder


class TokenBlock(nn.Module):
    def __init__(self,dimension,condition_dim,width=128):
        super().__init__();self.dimension=dimension;self.groups=math.ceil(dimension/8)
        self.input=nn.Linear(8,width);self.position=nn.Parameter(torch.randn(1,self.groups,width)*.02)
        self.physical_memory=condition_dim>=1027
        memory_width=64 if self.physical_memory else condition_dim
        global_dim=condition_dim-1024 if self.physical_memory else 0
        self.condition=nn.Linear(global_dim+3,width)
        self.memory=nn.Linear(memory_width,width)
        self.memory_position=nn.Parameter(torch.randn(1,16 if self.physical_memory else 1,width)*.02)
        self.cross_layers=nn.ModuleList([nn.TransformerDecoderLayer(width,4,2*width,dropout=0,batch_first=True,norm_first=True) for _ in range(2)])
        self.output=nn.Linear(width,8)

    def forward(self,x,c,k):
        B=len(x);x=nn.functional.pad(x,(0,self.groups*8-self.dimension)).reshape(B,self.groups,8)
        clock=torch.stack([k,(k*math.pi).sin(),(k*math.pi).cos()],-1)
        if self.physical_memory:
            # ProbabilityCondition flattens channels first, then4x4 locations.
            memory=c[:,:1024].reshape(B,64,16).transpose(1,2);glob=c[:,1024:]
        else:memory=c[:,None];glob=c[:,:0]
        memory=self.memory(memory)+self.memory_position
        step=self.condition(torch.cat([glob,clock],-1))[:,None]
        h=self.input(x)+self.position
        for layer in self.cross_layers:h=layer(h+step,memory)
        return self.output(h).flatten(1)[:,:self.dimension]


class InnovationDiffusion(nn.Module):
    def __init__(self,dimension,condition_dim,domain='spatial',U=None,width=128,state_to_space=None):
        super().__init__()
        if domain not in ('spatial','graph','dual'):raise ValueError(domain)
        self.domain=domain;self.dimension=dimension
        U=torch.eye(dimension) if U is None else torch.as_tensor(U,dtype=torch.float32)
        self.register_buffer('U',U)
        M=torch.eye(dimension) if state_to_space is None else torch.as_tensor(state_to_space,dtype=torch.float32)
        self.register_buffer('state_to_space',M)
        self.register_buffer('space_to_state',torch.linalg.inv(M))
        if domain!='graph':self.space=TokenBlock(dimension,condition_dim,width)
        if domain!='spatial':self.spectrum=TokenBlock(dimension,condition_dim,width)
        if domain=='dual':self.head=nn.Linear(2*dimension,dimension)
        t=torch.linspace(0,1,101);a=torch.cos((t+.008)/1.008*math.pi/2).square();a=(a/a[0]).clamp_min(1e-5)
        self.register_buffer('alpha',a)

    def forward(self,x,c,k):
        physical=x@self.state_to_space
        if self.domain!='graph':s=self.space(physical,c,k)@self.space_to_state
        if self.domain!='spatial':f=self.spectrum(physical@self.U,c,k)@self.U.T@self.space_to_state
        if self.domain=='dual':return self.head(torch.cat([s,f],-1))
        return s if self.domain=='spatial' else f

    def loss(self,x,c):
        k=torch.randint(1,101,(len(x),),device=x.device);eps=torch.randn_like(x)
        a=self.alpha[k,None];noisy=a.sqrt()*x+(1-a).sqrt()*eps
        return (self(noisy,c,k/100)-eps).square().mean()

    @torch.no_grad()
    def sample(self,c,noise,steps=20):
        x=noise.clone();schedule=torch.linspace(100,0,steps+1,device=x.device).round().long()
        for k,j in zip(schedule[:-1],schedule[1:]):
            eps=self(x,c,torch.full((len(x),),float(k)/100,device=x.device))
            a,b=self.alpha[k],self.alpha[j]
            clean=(x-(1-a).sqrt()*eps)/a.sqrt()
            x=b.sqrt()*clean+(1-b).sqrt()*eps
        return x


class ProbabilityCondition(nn.Module):
    """Shared across OOF models: observed 25ch + frozen physical mean, static8."""
    def __init__(self,side=16,width=64,spectral=False):
        super().__init__();self.spectral=spectral
        self.encoder=CausalSpatialEncoder(channels=26,width=width,side=side)
        self.output_dim=width*16+(16 if spectral else 0)+3

    def forward(self,history,static,clock,spectrum=None):
        h=self.encoder(history,static)
        out=nn.functional.adaptive_avg_pool2d(h,4).flatten(1)
        if self.spectral:
            if spectrum is None:raise ValueError('past-only spectrum required')
            out=torch.cat([out,spectrum],-1)
        return torch.cat([out,clock],-1)


class ConditionalGaussian(nn.Module):
    def __init__(self,dimension,condition_dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(condition_dim,128),nn.SiLU(),nn.Linear(128,2*dimension))

    def forward(self,c):
        mean,logstd=self.net(c).chunk(2,-1);return mean,logstd.clamp(-4,3)

    def loss(self,x,c):
        mean,logstd=self(c);return (.5*((x-mean)*(-logstd).exp()).square()+logstd).mean()

    def sample(self,c,noise,steps=20):
        mean,logstd=self(c);return mean+logstd.exp()*noise


class LatentField(nn.Module):
    """Training-label set encoder and a query decoder linear in canonical z."""
    def __init__(self,dimension=128,condition_width=64):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,128),nn.SiLU())
        self.encode_head=nn.Linear(256,dimension)
        self.basis=nn.Sequential(nn.Linear(10+condition_width,128),nn.SiLU(),nn.Linear(128,dimension))

    def encode(self,u,static,mask):
        if not mask.any(-1).all():raise ValueError('label set must have at least one valid point')
        s=static[None].expand(len(u),-1,-1)
        h=self.point(torch.cat([torch.nan_to_num(u)[...,None],s],-1))
        mean=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum=h.masked_fill(~mask[...,None],-torch.inf).max(1).values
        return self.encode_head(torch.cat([mean,maximum],-1))

    def decode(self,z,query_features):
        return torch.einsum('bqd,bd->bq',self.basis(query_features),z)/math.sqrt(z.shape[-1])
