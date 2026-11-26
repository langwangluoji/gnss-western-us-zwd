"""Full-field Gaussian/Student controls with matched mean and covariance."""
import math
import torch


def log_density(radius,logdet,dimension,nu=None):
    q=torch.as_tensor(radius).double()
    ld=torch.as_tensor(logdet,device=q.device,dtype=q.dtype)
    d=torch.as_tensor(dimension,device=q.device,dtype=q.dtype)
    if not torch.isfinite(q).all() or not torch.isfinite(ld).all() or (q<0).any() or (d<=0).any():
        raise ValueError('density requires nonempty observed coordinates and finite radius/logdet')
    if nu is None:return -.5*(d*math.log(2*math.pi)+ld+q)
    v=torch.as_tensor(nu,device=q.device,dtype=q.dtype)
    if not torch.isfinite(v).all() or (v<=2).any():raise ValueError('matched finite covariance requires nu>2')
    return (torch.lgamma((v+d)/2)-torch.lgamma(v/2)-.5*d*torch.log((v-2)*math.pi)-.5*ld
        -.5*(v+d)*torch.log1p(q/(v-2)))


def masked_radius(error,covariance):
    error=error.double();covariance=covariance.double()
    if error.ndim!=2 or covariance.shape!=(len(error),error.shape[1],error.shape[1]):
        raise ValueError('batched error and matching full covariance required')
    if not torch.isfinite(covariance).all():raise ValueError('generated covariance must remain finite')
    valid=torch.isfinite(error);dimension=valid.sum(-1)
    radius=error.new_full((len(error),),float('nan'));logdet=radius.clone()
    for i in range(len(error)):
        index=torch.nonzero(valid[i]).flatten()
        if not len(index):continue
        C=covariance[i][index][:,index];L=torch.linalg.cholesky(C)
        value=torch.linalg.solve_triangular(L,error[i,index,None],upper=False)
        radius[i]=value.square().sum();logdet[i]=2*L.diag().log().sum()
    return radius,logdet,dimension


@torch.no_grad()
def affine_covariance(model,b,residual):
    if model.family!='affine':raise ValueError('analytic covariance requires the affine Gaussian source')
    scale,phi=model.query(b['condition']);B,Q,H,D=phi.shape
    eye=torch.eye(D*H,device=scale.device).reshape(D*H,1,D,H).expand(-1,B,-1,-1)
    F=model.latent(eye,b['global_condition']).flatten(2).permute(1,2,0).double()
    P=(phi[...,None]*torch.eye(H,device=phi.device)[None,None,:,None,:]).reshape(B,Q*H,D*H).double()
    a=scale.flatten(1).double();M=(a[...,None]*b['basis'].double()+P)@F
    C=a[:,:,None]*residual*a[:,None,:]+M@M.transpose(-1,-2)
    sigma=b['std'].flatten(1).double();C=sigma[:,:,None]*C*sigma[:,None,:]
    return .5*(C+C.transpose(-1,-2))
