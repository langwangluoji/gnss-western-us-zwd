"""Equivalent full-spectrum evaluation with commuting bias-free linear filters."""
import math
import torch
from .native_frequency_tokens import NativeFrequencyTokens


class CommutedFrequencyTokens(NativeFrequencyTokens):
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
            # U and channel-linear W commute; each eigen-response is scalar.
            # All graph modes survive. Original parameter names/shapes are intact.
            filtered=sum(layer(graph_frequency)*response[None,:,None,None]
                for layer,response in zip(filters,responses))
            spatial=torch.einsum('qm,bmhw->bqhw',U,filtered)
            W=h.shape[-1]
            temporal=time_mix(h.reshape(B*Q,H,W).transpose(1,2)).transpose(1,2).reshape(B,Q,H,W)
            h=norm(h+torch.nn.functional.silu(local(h)+spatial+temporal))
        return self.output(h).squeeze(-1)@self.D
