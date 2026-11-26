"""V2 spike: use the published grid observation operator during training too."""
import numpy as np
import torch
from scipy.ndimage import uniform_filter
import sequence_mechanism_spike as core
from gnss_field.artifacts import write_json, sha256

OriginalMean, OriginalRandom, original_encode = core.MeanModel, core.RandomField, core.encode_inputs
STORE = core.PackedStore(core.DATA, purpose='context')
GRID = STORE.grid
Z = GRID['height']; GY,GX=np.gradient(Z,25000.,25000.)
ROUGH=np.sqrt(np.maximum(uniform_filter(Z*Z,5)-uniform_filter(Z,5)**2,0))
TERRAIN=np.stack([Z/3000,GX/.1,GY/.1,ROUGH/500],-1).reshape(-1,4)
XY=np.c_[GRID['x'].ravel(),GRID['y'].ravel()]
_,QI,QW,_=core.query_table(STORE)


def grid_inputs(a,raw,hours,xy,terrain):
    assert np.array_equal(xy,a['xy']), 'This training adapter expects the locked station order'
    nodes,inverse=np.unique(QI,return_inverse=True)
    x,base,_=original_encode(a,raw,hours,XY[nodes],TERRAIN[nodes])
    _,_,support=original_encode(a,raw,hours,xy,terrain)
    index=inverse.reshape(-1,4)
    x=x[:,:,index]
    weights=np.broadcast_to(QW, x.shape[:-1])[...,None].astype('f4')
    return np.concatenate([x,weights],-1), (base[:,:,index]*QW).sum(-1).astype('f4'),support


class GridMean(OriginalMean):
    def __init__(self,dim,temporal):super().__init__(dim-1,temporal)

    def forward(self,x):
        b,t,n,c,f=x.shape
        r,h=super().forward(x[...,:-1].reshape(b,t,n*c,f-1))
        w=x[...,-1]
        r=(r.reshape(b,t,n,c)*w).sum(-1)
        h=torch.cat([h.reshape(b,t,n,c,32),w[...,None]],-1)
        return r,h


class GridRandom(OriginalRandom):
    def forward(self,h,noise):
        b,t,n,c,f=h.shape
        r=super().forward(h[...,:-1].reshape(b,t,n*c,32),noise)
        return (r.reshape(b,noise.shape[1],t,n,c)*h[...,-1][:,None]).sum(-1)


if __name__=='__main__':
    core.ROOT=core.PROJECT/'results/20261123_sequence_spike_v02_grid'
    core.ROOT.mkdir(exist_ok=True)
    core.MeanModel=GridMean; core.RandomField=GridRandom; core.encode_inputs=grid_inputs
    core.main()
    write_json(core.ROOT/'operator_receipt.json',dict(adapter_sha256=sha256(__file__),
        core_sha256=sha256(core.__file__),training_and_evaluation='144x96 Z0 grid, bilinear query, exp(-H/3000)',
        initial_v1='direct coordinate prototype retained separately; never compare as same operator'))
