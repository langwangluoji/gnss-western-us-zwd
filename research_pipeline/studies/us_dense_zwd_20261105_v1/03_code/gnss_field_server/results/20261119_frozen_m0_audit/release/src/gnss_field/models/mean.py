"""Checkpoint-compatible 24x24 mean field, in millimetres.

The patch baseline is the original 6x6 field.
"""
import torch
from torch.nn import functional as F
from .legacy_core import SearchGenerator

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
