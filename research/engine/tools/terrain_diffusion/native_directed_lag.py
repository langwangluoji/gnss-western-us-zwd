"""Independent H carrier plus query/lead/station/lag numeric transport."""
import math
import torch
from torch import nn
from .native_mean_fusion_v2 import NativeMeanFusionV2


class DirectedLagReadout(nn.Module):
    def __init__(self,mode='joint',width=64,length=72):
        super().__init__()
        if mode not in ('factorized','joint'):raise ValueError('factorized or joint transport required')
        self.mode=mode;self.heads=4;self.width=width
        self.query=nn.Linear(width,width);self.key=nn.Linear(width,width)
        self.horizon=nn.Embedding(7,width)
        self.edge=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,4))
        self.readout=nn.Parameter(torch.zeros(7,4,4))
        self.register_buffer('lags',torch.tensor([i for i in (0,1,3,6,12,24) if i<length]),persistent=False)
        self.register_buffer('radii',torch.tensor([50.,100.,200.,400.]),persistent=False)
        self.register_buffer('limits',torch.tensor([3.,6.,6.,6.,6.,6.,6.]),persistent=False)

    def edge_scores(self,geo):
        Q,C,_=geo.shape;K=len(self.lags);device=geo.device
        lag=(self.lags/24).to(geo.dtype).view(1,1,1,K,1)
        lead=torch.arange(7,device=device,dtype=geo.dtype).view(1,7,1,1,1)/6
        if self.mode=='joint':
            features=torch.cat([geo[:,None,:,None].expand(Q,7,C,K,8),
                lag.expand(Q,7,C,K,1),lead.expand(Q,7,C,K,1)],-1)
            return self.edge(features)
        geography=torch.cat([geo[:,None,:,None].expand(Q,7,C,1,8),
            geo.new_zeros(Q,7,C,1,1),lead.expand(Q,7,C,1,1)],-1)
        temporal=torch.cat([geo.new_zeros(1,7,1,K,8),lag.expand(1,7,1,K,1),lead.expand(1,7,1,K,1)],-1)
        origin=torch.cat([geo.new_zeros(1,7,1,1,9),lead],-1)
        return self.edge(geography)+self.edge(temporal)-self.edge(origin)

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
        token=valid.any(-1)[:,None,None,:,:,None]
        score=score.masked_fill(~token,-1e4)
        weight=(score-score.amax((3,4),keepdim=True)).exp()*token
        denominator=torch.einsum('bqhcld,bclv->bqhdv',weight,valid.to(weight.dtype))
        height=torch.exp(context_static[None,:,2]-query_static[:,None,2])
        factor=torch.stack([height,height,torch.ones_like(height),torch.ones_like(height)],-1)
        values=numeric[:,None]*factor[None,:,:,None]
        numerator=torch.einsum('bqhcld,bqclv->bqhdv',weight,values)
        pooled=numerator/denominator.clamp_min(1e-8)
        raw=(pooled*self.readout[None,None]).sum((-1,-2))
        return self.limits*torch.tanh(raw/self.limits)


class NativeDirectedLag(NativeMeanFusionV2):
    def __init__(self,mode='joint',kind='transformer',width=64,length=72):
        super().__init__('H',kind,width,length)
        self.transport=DirectedLagReadout(mode,width,length)
        self.disabled_transport=False

    def new_parameter_names(self):
        return super().new_parameter_names()|{n for n,_ in self.named_parameters() if n.startswith('transport.')}

    def forward(self,condition,history,context_static,context_ids,query_static,return_aux=False):
        # Immutable H computation reproduced below; only the final bounded path is added.
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
        positions=history.shape[1]-1-self.lags
        observed=(history[:,positions,:,3]>0).permute(0,2,1)
        errors=(history[:,positions,:,10]*10).permute(0,2,1)
        physical_weight=self.weights(score,observed)
        height=torch.exp(context_static[None,:,2]-query_static[:,None,2])
        lag_heads=torch.einsum('bqckd,bck,qc->bqkd',physical_weight,errors,height)
        perlag=lag_heads.mean(-1)
        current=(torch.tanh(self.current_gain(h))*perlag).sum(-1)/math.sqrt(len(self.lags))
        centered=lag_heads-perlag[:,:,:,None]
        gains=torch.tanh(self.current_residual(h)).reshape(B,Q,len(self.lags),4)
        current=current+(gains*centered).sum((-1,-2))/math.sqrt(len(self.lags))
        perhead=torch.einsum('bqcd,bch,qc->bqhd',physical_weight[:,:,:,0],terminal[:,:,:,0]*10,height)
        mixture=self.head_mix(h).reshape(B,Q,6,4).softmax(-1)
        future=(perhead*mixture).sum(-1)*(1+.5*torch.tanh(self.future_gain(h)))
        raw=raw+torch.cat([current[:,:,None],current[:,:,None]+future],-1)
        correction=self.limits*torch.tanh(raw/self.limits)
        if not self.disabled_transport:correction=correction+self.transport(h,all_states,history,context_static,query_static)
        return (correction,h,auxiliary) if return_aux else (correction,h)
