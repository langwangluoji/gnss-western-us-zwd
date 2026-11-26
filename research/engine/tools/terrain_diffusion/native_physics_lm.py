"""Causal station-state supervision and explicit innovation/increment value paths."""
import math
import torch
from torch import nn
from .native_weather_models import NativeMean


def causal_station_loss(prediction,history):
    raw=history[:,:,:,:3].permute(0,2,1,3)
    valid=(history[:,:,:,3:6]>0).permute(0,2,1,3)
    ratio=raw.new_tensor([10.,10.,50.]) # cache normalization / (10mm,2K,2hPa)
    total=prediction.sum()*0;count=0;L=raw.shape[2]
    for lead in range(1,min(7,L)):
        mask=valid[:,:,lead:]&valid[:,:,:-lead]
        target=(raw[:,:,lead:]-raw[:,:,:-lead])*ratio
        error=prediction[:,:,:L-lead,lead-1]-target
        total=total+error[mask].square().sum();count+=int(mask.sum())
    return total/max(count,1),count


class NativePhysicsLM(NativeMean):
    def __init__(self,kind='transformer',width=64,length=72,physics=True):
        if kind=='weather':raise ValueError('station temporal encoder required')
        super().__init__(kind,width,length,weather=True);self.physics=physics
        self.node_head=nn.Linear(width,18)
        self.register_buffer('lags',torch.tensor([l for l in (0,1,3,6,12,24) if l<length]))
        self.register_buffer('radii',torch.tensor([50.,100.,200.,400.]))
        self.current_gain=nn.Linear(width,len(self.lags));self.future_gain=nn.Linear(width,6)
        self.head_mix=nn.Linear(width,24)
        for layer in (self.node_head,self.current_gain,self.future_gain,self.head_mix):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    @staticmethod
    def weights(score,valid):
        # score[B,Q,C,head], valid[B,C,lag]. No target availability is used.
        value=score[:,:,:,None].masked_fill(~valid[:,None,:,:,None],-1e4).softmax(2)
        value=value*valid[:,None,:,:,None]
        return value/value.sum(2,keepdim=True).clamp_min(1e-8)

    def forward(self,condition,history,context_static,context_ids,query_static,return_aux=False):
        query=self.query(condition);all_states=self.time(history,context_static,context_ids)
        states=all_states[:,:,-1];B,Q,W=query.shape;C=len(context_ids);D=W//4
        xy=(query_static[:,None,:2]-context_static[None,:,:2])*1000/300
        distance=xy.square().sum(-1).sqrt()*300
        edge=torch.cat([xy,distance.log1p()[...,None]/6,
            (query_static[:,None,2:3]-context_static[None,:,2:3])*3,
            query_static[:,None,8:9]-context_static[None,:,8:9],
            query_static[:,None,3:4]-context_static[None,:,3:4],
            query_static[:,None,5:6]-context_static[None,:,5:6]],-1)
        encoded=self.edge(edge);bias,values=encoded[:,:,:4],encoded[:,:,4:]
        score=torch.einsum('bqhd,bchd->bqch',query.reshape(B,Q,4,D),self.key(states).reshape(B,C,4,D))/math.sqrt(D)
        score=score+bias[None]-.5*(distance[None,:,:,None]/self.radii).square()
        any_history=(history[:,:,:,3].sum(1)>0)[:,:,None]
        weight=self.weights(score,any_history).squeeze(3)
        value=self.value(states)[:,None]+values[None]
        message=(weight[...,None]*value.reshape(B,Q,C,4,D)).sum(2).reshape(B,Q,W)
        h=self.fusion(torch.cat([query,message],-1));raw=self.head(h)
        if return_aux:
            auxiliary=self.node_head(all_states).reshape(B,C,history.shape[1],6,3)
            terminal=auxiliary[:,:,-1]
        else:terminal=self.node_head(states).reshape(B,C,6,3)
        if self.physics:
            positions=history.shape[1]-1-self.lags
            observed=(history[:,positions,:,3]>0).permute(0,2,1)
            errors=(history[:,positions,:,10]*10).permute(0,2,1)
            physical_weight=self.weights(score,observed)
            height=torch.exp(context_static[None,:,2]-query_static[:,None,2])
            perlag=torch.einsum('bqckd,bck,qc->bqkd',physical_weight,errors,height).mean(-1)
            current=(torch.tanh(self.current_gain(h))*perlag).sum(-1)/math.sqrt(len(self.lags))
            perhead=torch.einsum('bqcd,bch,qc->bqhd',physical_weight[:,:,:,0],terminal[:,:,:,0]*10,height)
            mixture=self.head_mix(h).reshape(B,Q,6,4).softmax(-1)
            future=(perhead*mixture).sum(-1)*(1+.5*torch.tanh(self.future_gain(h)))
            raw=raw+torch.cat([current[:,:,None],current[:,:,None]+future],-1)
        correction=self.limits*torch.tanh(raw/self.limits)
        return (correction,h,auxiliary) if return_aux else (correction,h)


def mean_model(kind,length=72,variant='baseline',weather=True):
    if variant=='baseline':return NativeMean(kind,64,length,weather)
    return NativePhysicsLM(kind,64,length,physics=variant in ('physics','physics_aux'))
