"""Independent, exactly warm-started current-head or same-time spatial mean arms."""
import math
import torch
from torch import nn
from .native_physics_lm import NativePhysicsLM
from .native_weather_models import StationEncoder


def neighbor_indices(context_static,context_ids,k=8):
    """Geometry-only neighbors, excluding self; fixed IDs break distance ties."""
    count=len(context_ids)
    order=torch.argsort(context_ids,stable=True)
    xy=context_static[:,:2]
    distance=(xy[:,None]-xy[None]).square().sum(-1)
    distance.fill_diagonal_(torch.inf)
    ranked=torch.argsort(distance[:,order],dim=-1,stable=True)
    return order[ranked[:,:min(k,max(count-1,0))]]


class SameTimeSpatial(nn.Module):
    def __init__(self,width,bottleneck=16,neighbors=8):
        super().__init__();self.neighbors=neighbors
        self.query=nn.Linear(width,bottleneck,bias=False)
        self.key=nn.Linear(width,bottleneck,bias=False)
        self.value=nn.Linear(width,bottleneck,bias=False)
        self.edge=nn.Sequential(nn.Linear(7,bottleneck),nn.SiLU(),nn.Linear(bottleneck,1))
        self.output=nn.Linear(bottleneck,width,bias=False)
        self.gate=nn.Parameter(torch.zeros(()))

    def forward(self,clean_states,history,context_static,context_ids):
        neighbors=neighbor_indices(context_static,context_ids,self.neighbors)
        if not neighbors.shape[-1]:return torch.zeros_like(clean_states)
        center=context_static[:,None];source=context_static[neighbors]
        xy=(center[...,:2]-source[...,:2])*1000/300
        distance=xy.square().sum(-1).sqrt()*300
        edge=torch.cat([xy,distance.log1p()[...,None]/6,(center[...,2:3]-source[...,2:3])*3,
            center[...,8:9]-source[...,8:9],center[...,3:4]-source[...,3:4],
            center[...,5:6]-source[...,5:6]],-1)
        q=self.query(clean_states);keys=self.key(clean_states)[:,:,neighbors]
        score=(q[:,:,:,None]*keys).sum(-1)/math.sqrt(q.shape[-1])
        score=score+self.edge(edge).squeeze(-1)[None,None]-.5*(distance[None,None]/300).square()
        valid=(history[:,:,:,3:6]>0).any(-1)[:,:,neighbors]
        weight=score.masked_fill(~valid,-1e4).softmax(-1)*valid
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        value=self.value(clean_states)
        message=(weight[...,None]*(value[:,:,neighbors]-value[:,:,:,None])).sum(-2)
        return torch.tanh(self.gate)*self.output(torch.nn.functional.silu(message))


class SpatialStationEncoder(StationEncoder):
    def __init__(self,kind,width=64,length=72):
        super().__init__(kind,width,length);self.spatial=SameTimeSpatial(width)

    def forward(self,history,context_static,context_ids):
        B,L,C,_=history.shape
        if L>self.length:raise ValueError('longer history than registered encoder')
        st=context_static[None,None].expand(B,L,-1,-1)
        identity=self.identity(context_ids)[None,None].expand(B,L,-1,-1)
        h=self.input(torch.cat([history,st,identity],-1))
        # The old encoder receives the original cache exactly. The new message
        # receives sanitized variable values; a missing GNSS also masks innovations.
        clean=history.clone()
        clean[:,:,:,:3]=torch.where(history[:,:,:,3:6]>0,history[:,:,:,:3],0)
        clean[:,:,:,9:11]=torch.where(history[:,:,:,3:4]>0,history[:,:,:,9:11],0)
        clean_states=self.input(torch.cat([clean,st,identity],-1))
        h=h+self.spatial(clean_states,history,context_static,context_ids)
        h=h.permute(0,2,1,3).reshape(B*C,L,self.width)+self.position[-L:][None]
        if self.kind=='transformer':
            mask=torch.ones(L,L,device=h.device,dtype=torch.bool).triu(1)
            h=self.encoder(h,mask=mask)
        else:h=self.encoder(h)[0]
        return h.reshape(B,C,L,self.width)


class NativeMeanFusionV2(NativePhysicsLM):
    def __init__(self,arm='C',kind='transformer',width=64,length=72):
        if arm not in ('C','H','S'):raise ValueError('arm must be C, H or S; no combined arm')
        super().__init__(kind,width,length,physics=True);self.arm=arm
        if arm=='H':
            self.current_residual=nn.Linear(width,len(self.lags)*4)
            nn.init.zeros_(self.current_residual.weight);nn.init.zeros_(self.current_residual.bias)
        elif arm=='S':self.time=SpatialStationEncoder(kind,width,length)

    def new_parameter_names(self):
        prefixes=('current_residual.',) if self.arm=='H' else ('time.spatial.',) if self.arm=='S' else ()
        return {name for name,_ in self.named_parameters() if name.startswith(prefixes)}

    def forward(self,condition,history,context_static,context_ids,query_static,return_aux=False):
        if self.arm!='H':
            return super().forward(condition,history,context_static,context_ids,query_static,return_aux)
        # Frozen source forward copied once; only the centered current residual is new.
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
        return (correction,h,auxiliary) if return_aux else (correction,h)


def load_source_state(model,state):
    """Load every source key strictly, allowing only registered new parameters."""
    expected=set(model.state_dict())-model.new_parameter_names()
    if set(state)!=expected:
        raise ValueError(f'source keys differ: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}')
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)!=model.new_parameter_names() or result.unexpected_keys:
        raise ValueError('source load changed unregistered parameters')
