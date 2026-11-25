"""Condition-space model errors, causal history and small analytic correction heads."""
import torch


def innovation_fields(errors,valid,qxy,cxy,qheight,cheight):
    distance=torch.cdist(qxy.float(),cxy.float(),compute_mode='donot_use_mm_for_euclid_dist')
    errors=torch.where(valid,errors.float(),0)*torch.exp(cheight.float()[None]/3000)
    fields=[]
    for radius in (50,150,300):
        weight=torch.exp(-.5*(distance/radius).square())[None]*valid[:,None]
        mean=(weight*errors[:,None]).sum(-1)/weight.sum(-1).clamp_min(1e-8)
        fields.append(mean*torch.exp(-qheight.float()[None]/3000))
    return torch.stack(fields,-1)


def causal_features(fields,positions,hours):
    parts=[]
    for lag in (0,1,3):
        index=positions-lag
        parts.append(fields[index.clamp_min(0)]*(index>=0)[:,None,None])
    base=torch.cat(parts,-1)/10
    phase=hours.double()*(2*torch.pi/(365.25*24))
    return torch.cat([base,base*phase.sin().float()[:,None,None],base*phase.cos().float()[:,None,None]],-1)


def normal_equations(features,target,mask,weights):
    x=features.double();y=target.double();w=weights.double()
    a=x[mask];b=y[mask];wt=w[mask];den=wt.sum()
    matrix=a.T@(a*wt[:,None])/den;rhs=a.T@(b*wt)/den
    pair=mask[1:]&mask[:-1]
    if pair.any():
        dx=x.diff(dim=0)[pair];dy=y.diff(dim=0)[pair];dw=((w[1:]+w[:-1])/2)[pair];den=dw.sum()
        matrix=matrix+.1*dx.T@(dx*dw[:,None])/den
        rhs=rhs+.1*dx.T@(dy*dw)/den
    return matrix.cpu(),rhs.cpu()


def solve_ridge(matrix,rhs,ridge=.01):
    return torch.linalg.solve(matrix+ridge*torch.diag(matrix.diag().clamp_min(1e-10)),rhs)
