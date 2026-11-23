"""One differentiable field publication and observation operator."""
import numpy as np
import torch
from torch.nn import functional as F

def mosaic(patches,origins,shape):
    n,c,h,w=patches.shape
    axis=torch.arange(h,device=patches.device,dtype=patches.dtype)
    window=torch.sin(torch.pi*(axis+.5)/h).square()
    weight=window[:,None]*window[None,:]
    numerator=patches.new_zeros(c,*shape)
    denominator=patches.new_zeros(*shape)
    for patch,(y,x) in zip(patches,origins):
        numerator[:,y:y+h,x:x+w]=numerator[:,y:y+h,x:x+w]+patch*weight
        denominator[y:y+h,x:x+w]+=weight
    if (denominator<=0).any(): raise ValueError('Uncovered mosaic grid')
    return numerator/denominator

def query_grid(field,xy):
    h,w=field.shape[-2:]
    if ((xy<0).any() or (xy[:,0]>w-1).any() or (xy[:,1]>h-1).any()):
        raise ValueError('Query outside grid centers')
    x0=xy[:,0].floor().long().clamp(max=w-2);y0=xy[:,1].floor().long().clamp(max=h-2)
    dx=xy[:,0]-x0;dy=xy[:,1]-y0
    return (field[:,y0,x0]*(1-dx)*(1-dy)+field[:,y0,x0+1]*dx*(1-dy)
            +field[:,y0+1,x0]*(1-dx)*dy+field[:,y0+1,x0+1]*dx*dy).T

def curvature(field,support,scale):
    m=field[0]
    lap=m[1:-1,:-2]+m[1:-1,2:]+m[:-2,1:-1]+m[2:,1:-1]-4*m[1:-1,1:-1]
    good=support[1:-1,:-2]&support[1:-1,2:]&support[:-2,1:-1]&support[2:,1:-1]&support[1:-1,1:-1]
    return (lap[good]/scale).square().mean() if good.any() else m.sum()*0
