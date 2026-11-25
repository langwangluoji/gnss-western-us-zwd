"""Full temporal-frequency tokens with graph spectra and local band interaction."""
import math
import torch
from torch import nn
from .native_weather_models import NativeDiffusion,NativeGaussian


class NativeFrequencyTokens(NativeDiffusion):
    def __init__(self,condition_dim=64,width=64):
        super().__init__(condition_dim,width,domain='graph')
        # Replace the seven-channel node MLP by shared one-coefficient tokens.
        # All seven coefficients survive every layer; no bottleneck projection.
        self.input=nn.Linear(1+condition_dim+5+2,width)
        self.output=nn.Linear(width,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.time_mix=nn.ModuleList([nn.Conv1d(width,width,3,padding=1,groups=width) for _ in range(2)])

    def denoise(self,x,c,k,U,eigen):
        B,Q,H=x.shape;frequency=x@self.D.T
        clock=torch.stack([k,k.sin(),k.cos(),(k*math.pi).sin(),(k*math.pi).cos()],-1)
        index=torch.arange(H,device=x.device).float()/max(H-1,1)
        band=torch.stack([index,index.square()],-1)
        h=self.input(torch.cat([frequency[:,:,:,None],c[:,:,None].expand(-1,-1,H,-1),
            clock[:,None,None].expand(-1,Q,H,-1),band[None,None].expand(B,Q,-1,-1)],-1))
        for local,filters,norm,time_mix in zip(self.local,self.filters,self.norm,self.time_mix):
            graph_frequency=torch.einsum('qm,bqhw->bmhw',U,h)
            responses=[torch.ones_like(eigen),torch.exp(-eigen),torch.exp(-4*eigen)]
            spatial=sum(layer(torch.einsum('qm,bmhw->bqhw',U,graph_frequency*response[None,:,None,None]))
                for layer,response in zip(filters,responses))
            W=h.shape[-1]
            temporal=time_mix(h.reshape(B*Q,H,W).transpose(1,2)).transpose(1,2).reshape(B,Q,H,W)
            h=norm(h+torch.nn.functional.silu(local(h)+spatial+temporal))
        return self.output(h).squeeze(-1)@self.D


def probability_model(domain):
    if domain=='gaussian':return NativeGaussian()
    if domain=='frequency_tokens':return NativeFrequencyTokens()
    return NativeDiffusion(domain=domain)
