"""Permutation-invariant GNSS neighbor interpolation with learned corrections.

The model receives reference stations only. Constructing the target-excluded,
same-epoch reference set is the data builder's audited responsibility.
"""
import torch
from torch import nn


class NeighborGraph(nn.Module):
    def __init__(self, context_dim, neighbor_dim=9, width=64, correction_scale=20.0):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(context_dim,width),nn.SiLU(),
                                   nn.Linear(width,32),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(neighbor_dim+32,width),nn.SiLU(),
                                nn.Linear(width,32),nn.SiLU(),nn.Linear(32,2))
        self.global_correction=nn.Linear(32,1)
        self.correction_scale=float(correction_scale)
        # Starts at the explicit inverse-square distance baseline.
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)
        nn.init.zeros_(self.global_correction.weight)
        nn.init.zeros_(self.global_correction.bias)

    def forward(self, context, neighbor_features, neighbor_zwd_mm, distance_km, valid):
        if context.ndim!=2 or neighbor_features.ndim!=3:
            raise ValueError("Expected context [B,C], neighbors [B,K,F]")
        valid=valid.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Rows without reference observations require an explicit external fallback")
        if not torch.isfinite(context).all() or not torch.isfinite(neighbor_features).all():
            raise ValueError("Preprocess/impute permitted inputs before model evaluation")
        h=self.context(context)
        repeated=h[:,None,:].expand(-1,neighbor_features.shape[1],-1)
        edge=self.edge(torch.cat([repeated,neighbor_features],dim=-1))
        logits=-2*torch.log(distance_km.clamp_min(1.0)) + edge[...,0]
        logits=logits.masked_fill(~valid,-torch.inf)
        weights=torch.softmax(logits,dim=1)
        corrected=neighbor_zwd_mm+self.correction_scale*edge[...,1]
        prediction=(weights*corrected).sum(dim=1)
        prediction=prediction+self.correction_scale*self.global_correction(h).squeeze(-1)
        return prediction,weights
