"""Query-local trajectory transport coupled through a frozen weather GP."""
import torch
from .native_weather_gp import weather_summary
from .native_process_models import LocalTrajectoryVelocity,AffineTrajectoryVelocity
from .native_process_ops import heun0


def condition265(c175,history):
    summary=weather_summary(history)
    return torch.cat([c175,summary[:,None].expand(-1,c175.shape[1],-1)],-1)


def make_model(mode):
    if mode=='local':return LocalTrajectoryVelocity(265,111)
    if mode=='affine':return AffineTrajectoryVelocity(265,102)
    raise ValueError('weather GP transport supports local or affine')


def trajectory_factors(blocks):
    matrix=blocks.double()
    if matrix.shape[-2:]!=(7,7) or not torch.allclose(matrix,matrix.transpose(-1,-2),atol=1e-8,rtol=1e-8):
        raise ValueError('symmetric7x7 trajectory correlations required')
    eigen,U=torch.linalg.eigh((matrix+matrix.transpose(-1,-2))/2)
    if not torch.isfinite(eigen).all() or eigen.min()<=0:raise ValueError('trajectory correlation must be positive definite')
    return (U*eigen.sqrt()[...,None,:])@U.transpose(-1,-2),(U*eigen.rsqrt()[...,None,:])@U.transpose(-1,-2)


def row_transform(x,factor):
    return torch.einsum('...qj,...qji->...qi',x.double(),factor.double())


def masked_loss(model,u,c,inverse):
    eligible=torch.isfinite(u).all(-1);patterns,index=torch.unique(eligible,dim=0,return_inverse=True)
    loss=next(model.parameters()).sum()*0;count=0;used_origins=0
    for group,pattern in enumerate(patterns):
        if not pattern.any():continue
        origins=torch.nonzero(index==group).flatten();queries=torch.nonzero(pattern).flatten()
        white=row_transform(u[origins][:,queries],inverse[origins][:,queries]).float()
        weight=len(origins)*len(queries)
        loss=loss+weight*model.loss(white,c[origins][:,queries],None,None)
        count+=weight;used_origins+=len(origins)
    return loss/max(count,1),count,used_origins


@torch.no_grad()
def transform_gp_path(model,c,path,root,steps=20,solver='heun0'):
    inverse=torch.linalg.inv(root.double())
    initial=row_transform(path,inverse).float()
    result=(heun0(model,c,initial,steps=steps) if solver=='heun0' else
        model.sample(c,initial,None,None,steps,solver))
    return row_transform(result,root)


@torch.no_grad()
def affine_marginals(model,c,root,mu,std,steps=20,solver='heun0'):
    m,M=model.affine_map(c,steps,solver)
    mean=mu.double()+std.double()*row_transform(m,root)
    factor=std.double()[...,None]*(root.double()@M.double())
    return mean,factor.square().sum(-1).sqrt()
