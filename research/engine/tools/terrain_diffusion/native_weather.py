"""Native query weather/history features and lossless orthogonal transforms."""
import math
import numpy as np
import torch


def graph_basis(xy,height):
    distance=torch.cdist(xy.float(),xy.float(),compute_mode='donot_use_mm_for_euclid_dist')
    dh=(height[:,None]-height[None])/1500
    weight=torch.exp(-.5*(distance/300).square()-.5*dh.square())
    weight.fill_diagonal_(0)
    degree=weight.sum(-1).clamp_min(1e-8).rsqrt()
    lap=torch.eye(len(xy),device=xy.device)-degree[:,None]*weight*degree[None]
    eigen,U=torch.linalg.eigh(lap)
    return eigen,U


def dct_matrix(length,device=None):
    k=torch.arange(length,device=device).float()[:,None]
    j=torch.arange(length,device=device).float()[None]
    result=torch.cos(math.pi/length*(j+.5)*k)*math.sqrt(2/length)
    result[0]/=math.sqrt(2)
    return result


def full_spectrum(x,U,inverse=False):
    D=dct_matrix(x.shape[-1],x.device)
    if inverse:return torch.einsum('qm,...mh->...qh',U,x)@D
    return torch.einsum('qm,...qh->...mh',U,x@D.T)


def weather_fields(raw,qxy,cxy,qheight,cheight):
    valid=torch.isfinite(raw)
    values=torch.nan_to_num(raw).clone()
    values[:,:,0]*=torch.exp(cheight[None]/3000)
    values=(values-values.new_tensor([0,280,900]))/values.new_tensor([100,20,100])
    values=torch.where(valid,values,0)
    distance=torch.cdist(qxy.float(),cxy.float(),compute_mode='donot_use_mm_for_euclid_dist')
    parts=[]
    for radius in (50,150,300):
        w=torch.exp(-.5*(distance/radius).square())
        den=torch.einsum('qc,tcv->tqv',w,valid.float())
        avg=torch.einsum('qc,tcv->tqv',w,values)/den.clamp_min(1e-8)
        avg[:,:,0]*=torch.exp(-qheight[None]/3000)
        parts.extend([avg,den/w.sum(-1)[None,:,None].clamp_min(1e-8)])
    return torch.cat(parts,-1)


def history_features(innovation,background,weather,hours,positions,static,nearest):
    if int(positions.min())<24:raise ValueError('requires24h of earlier states')
    history=torch.cat([innovation[positions-lag,:,:3] for lag in (0,1,3,6,12,24)],-1)
    current=background[positions]
    levels=torch.stack([current/100,*[(current-background[positions-lag])/10 for lag in (1,3,6,12,24)]],-1)
    met=torch.cat([weather[positions-lag] for lag in (0,3,12,24)],-1)
    phase=hours[positions].double()
    clock=torch.stack([(phase%24*2*math.pi/24).sin(),(phase%24*2*math.pi/24).cos(),
        (phase*2*math.pi/(365.25*24)).sin(),(phase*2*math.pi/(365.25*24)).cos()],-1).float()
    distance=torch.stack([nearest.clamp_max(3000).log1p()/6,(nearest<50).float()],-1)
    return torch.cat([history,levels,met,static[None].expand(len(positions),-1,-1),
        clock[:,None].expand(-1,len(static),-1),distance],-1)


def validate_origin(origins,last_dynamic,start,stop,history=72,horizon=6):
    origins=np.asarray(origins)
    if not len(origins) or not np.all(np.diff(origins)==1):raise ValueError('origins must be consecutive')
    if np.any(np.asarray(last_dynamic)>origins):raise ValueError('future dynamic condition')
    if origins[0]-history+1<start or origins[-1]+horizon>=stop:raise ValueError('window crosses split')
