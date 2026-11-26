"""Fixed spatial coordinates and invertible, training-fitted residual transforms."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.fft import dct, idct


class AnchorField:
    def __init__(self,xy,neighbors=8):
        self.xy=np.asarray(xy,dtype=float)
        self.tri=Delaunay(self.xy);self.tree=cKDTree(self.xy)
        d,j=self.tree.query(self.xy,k=min(neighbors+1,len(xy)))
        scale=max(float(np.median(d[:,1:])),1e-6)
        W=np.zeros((len(xy),len(xy)))
        for i in range(len(xy)):W[i,j[i,1:]]=np.exp(-.5*(d[i,1:]/scale)**2)
        W=np.maximum(W,W.T);self.L=np.diag(W.sum(1))-W
        self.eigenvalues,self.U=np.linalg.eigh(self.L)

    def readout(self,q,return_outside=False):
        q=np.asarray(q,dtype=float);simplex=self.tri.find_simplex(q)
        H=np.zeros((len(q),len(self.xy)));inside=simplex>=0
        trans=self.tri.transform[simplex[inside]]
        b=np.einsum('nij,nj->ni',trans[:,:2],q[inside]-trans[:,2])
        weights=np.c_[b,1-b.sum(1)]
        H[np.flatnonzero(inside)[:,None],self.tri.simplices[simplex[inside]]]=weights
        outside=~inside;H[np.flatnonzero(outside),self.tree.query(q[outside])[1]]=1
        return (H,outside) if return_outside else H

    def encode(self,values,H,ridge=.01,smooth=.01):
        if ridge<=0 or smooth<0:raise ValueError('ridge must be positive; smooth nonnegative')
        values=np.asarray(values);H=np.asarray(H)
        result=[];penalty=ridge*np.eye(H.shape[1])+smooth*self.L
        for row in values.reshape(-1,values.shape[-1]):
            ok=np.isfinite(row)
            if not ok.any():raise ValueError('no labels for anchor encoding')
            h=H[ok];result.append(np.linalg.solve(h.T@h+penalty,h.T@row[ok]))
        return np.asarray(result).reshape(*values.shape[:-1],H.shape[1])


@dataclass
class InnovationTransform:
    bias:np.ndarray
    scale:np.ndarray
    rho:np.ndarray
    innovation_bias:np.ndarray
    innovation_scale:np.ndarray

    @classmethod
    def fit(cls,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=3 or not np.isfinite(x).all():raise ValueError('finite [event,time,channel] training states required')
        bias=x.mean((0,1));scale=np.maximum(x.std((0,1)),1e-4);w=(x-bias)/scale
        prev=np.concatenate([np.zeros_like(w[:,:1]),w[:,:-1]],1)
        rho=np.clip((prev*w).sum((0,1))/np.maximum((prev*prev).sum((0,1)),1e-8),-.95,.95)
        e=w-rho*prev
        return cls(bias,scale,rho,e.mean((0,1)),np.maximum(e.std((0,1)),1e-4))

    def forward(self,x):
        w=(x-self.bias)/self.scale
        prev=np.concatenate([np.zeros_like(w[...,:1,:]),w[...,:-1,:]],-2)
        return (w-self.rho*prev-self.innovation_bias)/self.innovation_scale

    def inverse(self,v):
        e=v*self.innovation_scale+self.innovation_bias;state=np.zeros_like(e[...,0,:]);out=[]
        for i in range(e.shape[-2]):
            state=self.rho*state+e[...,i,:];out.append(state.copy())
        return np.stack(out,-2)*self.scale+self.bias

    def arrays(self):return {k:getattr(self,k) for k in self.__dataclass_fields__}


def history_spectrum(values,origin,length=72,periods=(6.,12.,24.,48.)):
    """Past-only LS harmonic coefficients + power (phase retained), masks explicit."""
    if origin-length+1<0:raise ValueError('insufficient history')
    x=np.asarray(values[origin-length+1:origin+1],float)
    x=x.reshape(length,-1);t=np.arange(length,dtype=float)-length+1
    result=[]
    for period in periods:
        basis=np.c_[np.ones(length),np.cos(2*np.pi*t/period),np.sin(2*np.pi*t/period)]
        row=[]
        for y in x.T:
            ok=np.isfinite(y)
            c=np.linalg.lstsq(basis[ok],y[ok],rcond=None)[0][1:] if ok.sum()>=4 else np.zeros(2)
            row.append([*c,float(c@c),ok.mean()])
        result.append(row)
    return np.asarray(result).transpose(1,0,2).reshape(x.shape[1],-1).astype('float32')


def joint_transform(x,U,inverse=False):
    if inverse:return idct(x@U.T,type=2,axis=-2,norm='ortho')
    return dct(x,type=2,axis=-2,norm='ortho')@U


def require_forecast_origin(origins,last_dynamic,last_mean_source):
    if np.any(np.asarray(last_dynamic)>origins) or np.any(np.asarray(last_mean_source)>origins):
        raise ValueError('future conditions or means read observations after forecast origin')
