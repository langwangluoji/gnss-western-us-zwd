"""Existing RAW_CURRENT M0 and matched faithful/joint Gaussian residuals."""
import torch
from torch import nn
from torch.nn import functional as F
from architecture_search import SearchGenerator
from residual_edm_v1.model import ResidualModel

class MeanField(SearchGenerator):
    def __init__(self):super().__init__('query_film',False)

    def forward(self,b):
        stations=b['stations'];pad=b['padding']
        tokens=stations.new_zeros(*stations.shape[:2],67)
        tokens[...,:5]=stations[...,:5];tokens[...,5]=1;tokens[...,64:]=stations[...,10:13]
        tokens=tokens.masked_fill(pad[...,None],0)
        # Explicit empty token only visible for samples with no input station.
        empty=tokens.new_zeros(len(tokens),1,67)
        tokens=torch.cat([tokens,empty],1);pad=torch.cat([pad,~pad.all(1,keepdim=True)],1)
        maps=b['maps'];cycles=b['coarse'][:,0,9:13,0,0]
        pos=F.interpolate(maps[:,[1,2,0]],size=(6,6),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        queries=torch.cat([pos,cycles[:,None].expand(-1,36,-1)],-1)
        addition=self.token_features(tokens,pad,queries)
        valid=(~pad).float();pooled=(tokens[:,:,:64]*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        film=.1*torch.tanh(self.film_head(pooled).reshape(-1,4,16))
        finepos=maps[:,[1,2,0]].flatten(2).transpose(1,2)
        q=torch.cat([finepos,cycles[:,None].expand(-1,576,-1)],-1)
        encoded=self.mae.embedding(tokens[:,:,:64])+self.mae.position(tokens[:,:,64:])
        query=self.mae.query(q)
        distance=torch.cdist(q[:,:,:2],tokens[:,:,64:66])/.25
        mask=(-distance).masked_fill(pad[:,None,:],float('-inf')).repeat_interleave(4,dim=0)
        context,_=self.mae.attention(query,encoded,encoded,attn_mask=mask,need_weights=False)
        fine=torch.tanh(self.query_gate)*self.query_head(self.mae.norm(query+context)).transpose(1,2).reshape(-1,1,24,24)
        output=self.core(b['coarse'],b['baseline']/20,prior_only=True,sample=False,
            encoder_addition=addition,use_latent=False,fine_addition=fine,film=film)
        return output['field_mm']*20,None

class ResidualField(ResidualModel):
    def __init__(self):
        super().__init__('det')
        self.output=nn.Sequential(nn.GroupNorm(8,32),nn.SiLU())
        self.mean_head=nn.Conv2d(32,1,3,padding=1)
        nn.init.zeros_(self.mean_head.weight);nn.init.zeros_(self.mean_head.bias)
        self.variance_head=nn.Sequential(nn.Linear(35,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.variance_head[-1].weight)
        nn.init.constant_(self.variance_head[-1].bias,.5413248546)

    def encode_condition(self,b):
        stations=b['stations'].masked_fill(b['padding'][...,None],0)
        tokens=self.station_encoder(stations)+self.amplitude(stations[...,:5])
        tokens=torch.cat([tokens,self.null_station.expand(len(stations),-1,-1)],1)
        positions=torch.cat([stations[...,10:13],stations.new_zeros(len(stations),1,3)],1)
        pad=torch.cat([b['padding'],~b['padding'].all(1,keepdim=True)],1)
        coarse=F.interpolate(self.coarse_encoder(b['coarse'].flatten(1,2)),size=(24,24),mode='bilinear',align_corners=False)
        maps=torch.cat([b['mu_condition'],b['baseline_condition'],b['maps'],coarse,b['domain']],1)
        geometry=b['maps'][:,[1,2,0]]
        return dict(maps=maps,domain=b['domain'],
            attn12=self.attn12.prepare(tokens,positions,pad,F.interpolate(geometry,size=(12,12),mode='bilinear',align_corners=False)),
            attn6=self.attn6.prepare(tokens,positions,pad,F.interpolate(geometry,size=(6,6),mode='bilinear',align_corners=False)))

    def forward(self,b):
        cond=self.encode_condition(b);zero=torch.zeros_like(b['domain'])
        features=self._backbone(zero,zero.new_zeros(len(zero)),cond)
        return self.mean_head(features),features

    def standard_deviation(self,features,geometry,scale,faithful):
        if faithful:features=features.detach()
        return (F.softplus(self.variance_head(torch.cat([features,geometry],-1)).squeeze(-1))+1e-4)*scale

def gaussian_losses(mean,target,variance,scale,faithful):
    e=(mean-target)/scale;q=variance/(scale*scale)
    mse=.5*e.square().mean()
    nll=.5*(q.log()+(e.detach() if faithful else e).square()/q).mean()
    return mse,nll
