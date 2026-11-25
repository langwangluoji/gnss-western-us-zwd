"""Canonical Gaussian-process factors, masked whitening and exact clean endpoint."""
import torch
from .native_copula import spatial_correlation
from .native_weather import graph_basis


def principal_factors(matrix):
    K=matrix.double()
    if not torch.allclose(K,K.T,atol=1e-6,rtol=1e-6):raise ValueError('correlation must be symmetric')
    K=(K+K.T)/2;eigen,U=torch.linalg.eigh(K)
    if not torch.isfinite(eigen).all() or eigen.min()<=0:raise ValueError('correlation must be positive definite')
    root=(U*eigen.sqrt()[None])@U.T;inverse=(U*eigen.rsqrt()[None])@U.T
    return root.float(),inverse.float(),eigen


def space_factors(static,copula,query_ids=None):
    if query_ids is not None and len(torch.unique(query_ids))!=len(query_ids):raise ValueError('unique query IDs required')
    return principal_factors(spatial_correlation(static,float(copula['spatial_rho'])))


def time_factors(copula,device=None):
    return principal_factors(torch.tensor(copula['time_correlation'],device=device,dtype=torch.float64))


def whiten(x,spatial_inverse,temporal_inverse):
    if spatial_inverse is not None:x=torch.einsum('qr,...rh->...qh',spatial_inverse,x)
    return x@temporal_inverse


def color(x,spatial_root,temporal_root):
    if spatial_root is not None:x=torch.einsum('qr,...rh->...qh',spatial_root,x)
    return x@temporal_root


@torch.no_grad()
def heun0(model,c,noise,U=None,eigen=None,steps=20):
    if not 1<=steps<=100:raise ValueError('Heun0 steps must be within1..100')
    x=noise.clone();schedule=torch.linspace(100,0,steps+1).round().long().tolist()
    theta=model.alpha.sqrt().acos()
    for k,j in zip(schedule[:-1],schedule[1:]):
        first=model.denoise(x,c,torch.full((len(x),),k/100,device=x.device),U,eigen)
        delta=theta[j]-theta[k];proposal=x+delta*first
        second=(torch.zeros_like(first) if j==0 else
            model.denoise(proposal,c,torch.full((len(x),),j/100,device=x.device),U,eigen))
        x=x+.5*delta*(first+second)
    return x


def masked_process_loss(model,u,c,static,copula,mode,query_ids=None,temporal_inverse=None):
    if mode not in ('full','local','affine'):raise ValueError(mode)
    eligible=torch.isfinite(u).all(-1)
    patterns,inverse=torch.unique(eligible,dim=0,return_inverse=True)
    if temporal_inverse is None:temporal_inverse=time_factors(copula,u.device)[1]
    loss=next(model.parameters()).sum()*0;count=0;used_origins=0
    for group,pattern in enumerate(patterns):
        if not pattern.any():continue
        origins=torch.nonzero(inverse==group).flatten();queries=torch.nonzero(pattern).flatten()
        x=u[origins][:,queries];condition=c[origins][:,queries];st=static[queries]
        if mode=='full':
            _,spatial_inverse,_=space_factors(st,copula,query_ids[queries] if query_ids is not None else None)
            eigen,U=graph_basis(st[:,:2]*1000,st[:,2]*3000)
        else:spatial_inverse=None;U=None;eigen=None
        target=whiten(x,spatial_inverse,temporal_inverse)
        weight=len(origins)*len(queries)
        loss=loss+model.loss(target,condition,U,eigen)*weight
        count+=weight;used_origins+=len(origins)
    return loss/max(count,1),count,used_origins


@torch.no_grad()
def generate_process(model,mode,c,eta,static,copula,steps=20,solver='heun0',query_ids=None,spatial_noise=None):
    if mode not in ('reference','full','local','affine'):raise ValueError(mode)
    As,_,_=space_factors(static,copula,query_ids);At,_,_=time_factors(copula,eta.device)
    if mode=='full':
        if spatial_noise is not None:raise ValueError('full whitening starts from white eta, not precolored noise')
        eigen,U=graph_basis(static[:,:2]*1000,static[:,2]*3000);initial=eta
    else:
        U=None;eigen=None;initial=spatial_noise if spatial_noise is not None else torch.einsum('qr,brh->bqh',As,eta)
    if mode=='reference':return color(initial,None,At)
    result=(heun0(model,c,initial,U,eigen,steps) if solver=='heun0' else
        model.sample(c,initial,U,eigen,steps,solver))
    return color(result,As if mode=='full' else None,At)
