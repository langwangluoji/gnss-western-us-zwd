"""Versioned mask normalization fix; the immutable v1 source is retained."""
import math
import torch
from .native_directed_lag import DirectedLagReadout,NativeDirectedLag
from .native_mean_fusion_v2 import NativeMeanFusionV2


def independent_value_pool(score,values,valid):
    """Normalize each variable on its own support, including distant-only data."""
    results=[]
    for variable in range(valid.shape[-1]):
        mask=valid[...,variable]
        channel=score.masked_fill(~mask[:,None,None,:,:,None],float('-inf'))
        maximum=channel.amax((3,4),keepdim=True)
        present=mask.any((1,2))[:,None,None,None,None,None]
        maximum=torch.where(present,maximum,0)
        weight=(channel-maximum).exp()
        numeric=torch.where(mask[:,None],values[...,variable],0)
        numerator=torch.einsum('bqhcld,bqcl->bqhd',weight,numeric)
        # With any support, the largest legal weight is exactly1. Empty support
        # has numerator0 and denominator0 and returns zero with finite gradients.
        results.append(numerator/weight.sum((3,4)).clamp_min(1))
    return torch.stack(results,dim=-1)


class DirectedLagReadoutV2(DirectedLagReadout):
    def forward(self,h,states,history,context_static,query_static):
        B,Q,W=h.shape;C=history.shape[2];K=len(self.lags);D=W//self.heads
        positions=history.shape[1]-1-self.lags
        if int(positions.min())<0:raise ValueError('history shorter than registered lags')
        now=history[:,positions];previous=history[:,(positions-1).clamp_min(0)]
        available=now[...,3:6]>0;old=previous[...,3:6]>0
        paired=available&old&(positions>0)[None,:,None,None]
        valid=torch.stack([available[...,0],paired[...,0],paired[...,1],paired[...,2]],-1).permute(0,2,1,3)
        numeric=torch.stack([now[...,10]*10,(now[...,0]-previous[...,0])*100,
            (now[...,1]-previous[...,1])*100,(now[...,2]-previous[...,2])*500],-1).permute(0,2,1,3)
        numeric=torch.where(valid,numeric,0)
        selected=(states[:,:,positions] if self.mode=='joint' else states[:,:,-1:].expand(-1,-1,K,-1))
        keys=self.key(selected).reshape(B,C,K,self.heads,D)
        query=self.query(h[:,:,None]+self.horizon.weight[None,None]).reshape(B,Q,7,self.heads,D)
        score=torch.einsum('bqhdr,bcldr->bqhcld',query,keys)/math.sqrt(D)
        delta=query_static[:,None]-context_static[None]
        xy=delta[...,:2]*1000/300;distance=xy.square().sum(-1).sqrt()*300
        geo=torch.cat([xy,distance.log1p()[...,None]/6,delta[...,2:3]*3,
            delta[...,8:9],delta[...,3:6]],-1)
        score=score+self.edge_scores(geo)[None]-.5*(distance[None,:,None,:,None,None]/self.radii).square()
        height=torch.exp(context_static[None,:,2]-query_static[:,None,2])
        factor=torch.stack([height,height,torch.ones_like(height),torch.ones_like(height)],-1)
        values=numeric[:,None]*factor[None,:,:,None]
        pooled=independent_value_pool(score,values,valid)
        raw=(pooled*self.readout[None,None]).sum((-1,-2))
        return self.limits*torch.tanh(raw/self.limits)


class NativeDirectedLagV2(NativeDirectedLag):
    def __init__(self,mode='joint',kind='transformer',width=64,length=72):
        # Identical parameter creation order to v1 and the matched H carrier.
        NativeMeanFusionV2.__init__(self,'H',kind,width,length)
        self.transport=DirectedLagReadoutV2(mode,width,length)
        self.disabled_transport=False
