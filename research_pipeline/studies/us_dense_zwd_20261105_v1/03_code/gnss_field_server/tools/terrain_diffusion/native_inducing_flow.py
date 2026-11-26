"""Coupled inducing innovations; retain the complete residual Gaussian process."""
import torch
from .native_latent_flow import ProjectiveLatentFlow


def anchor_locations(train_static, count=8):
    points=train_static[:,:3].double()*train_static.new_tensor([1000/300,1000/300,2],dtype=torch.float64)
    points=torch.unique(points,dim=0)
    if not 0<count<=len(points):raise ValueError('insufficient unique geometry for anchors')
    chosen=[int((points-points.mean(0)).square().sum(-1).argmax())]
    distance=(points-points[chosen[0]]).square().sum(-1)
    for _ in range(count-1):
        chosen.append(int(distance.argmax()))
        distance=torch.minimum(distance,(points-points[chosen[-1]]).square().sum(-1))
    return points[chosen]


def inducing_decomposition(gp,c175,history,static,query_ids,anchors):
    psi,rho,time=gp.state(c175,history,static,query_ids)
    B,Q,H,_=psi.shape;A=len(anchors);N=Q*H
    anchor_points=anchors.to(device=psi.device,dtype=psi.dtype)[None,:,None].expand(B,A,H,3).reshape(B,A*H,3)
    points=torch.cat([psi.reshape(B,N,3),anchor_points],1)
    lead=torch.arange(H,device=psi.device).repeat(Q+A)
    global_kernel=gp.spatial_kernel(points,lead)*time[:,lead[:,None],lead[None,:]]
    Kaa=global_kernel[:,N:,N:]
    values,vectors=torch.linalg.eigh(Kaa)
    if not torch.isfinite(values).all() or (values<=0).any():
        raise ValueError('anchor covariance must be strictly positive definite; no clipping')
    inverse_root=(vectors*values.rsqrt()[:,None,:])@vectors.transpose(-1,-2)
    Kqa=rho.reshape(B,N,1).sqrt()*global_kernel[:,:N,N:]
    W=Kqa@inverse_root
    R=gp.correlation(c175,history,static,query_ids)
    residual=R-W@W.transpose(-1,-2)
    residual=.5*(residual+residual.transpose(-1,-2))
    torch.linalg.cholesky(residual)
    return R,W,residual


class InducingLatentFlow(ProjectiveLatentFlow):
    def forward(self,condition,global_condition,mean,std,gp_path,latent_noise,basis):
        scale,phi=self.query(condition);g=self.latent(latent_noise,global_condition)
        change=torch.einsum('bnd,kbd->kbn',basis,(g-latent_noise).flatten(2)).reshape_as(gp_path)
        low_rank=torch.einsum('bqhd,kbdh->kbqh',phi,g)
        return mean[None]+std[None]*(scale[None]*(gp_path+change)+low_rank)
