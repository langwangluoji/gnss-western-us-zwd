"""Weather-conditioned PSD space/time kernels with fixed Gaussian marginals."""
import math
import torch
from torch import nn


def weather_summary(history):
    if history.ndim!=4 or history.shape[-1]!=15 or history.shape[1]<72:
        raise ValueError('weather summary requires at least72 observed15-channel station states')
    clean=history.clone()
    clean[...,:3]=torch.where(history[...,3:6]>0,history[...,:3],0)
    clean[...,9:11]=torch.where(history[...,3:4]>0,history[...,9:11],0)
    return torch.cat([value for n in (1,24,72) for value in
        (clean[:,-n:].mean((1,2)),clean[:,-n:].std((1,2),unbiased=False))],-1)


def gaussian_nll(u,correlation):
    root=torch.linalg.cholesky(correlation.double());x=u.double()
    solved=torch.cholesky_solve(x[...,None],root).squeeze(-1)
    return (.5*((x*solved).sum(-1)+2*root.diagonal(dim1=-2,dim2=-1).log().sum(-1))/x.shape[-1]).mean()


class WeatherProjectiveGP(nn.Module):
    def __init__(self,view,copula):
        super().__init__()
        if view not in (175,265):raise ValueError('GP condition view must be175 or265')
        rho=float(copula['spatial_rho']);self.rho_min=1e-5;self.rho_max=.99
        if not self.rho_min<rho<self.rho_max:raise ValueError('reference rho must lie inside registered model bounds')
        self.view=view;rows,cols=torch.tril_indices(7,7)
        self.register_buffer('rows',rows);self.register_buffer('cols',cols)
        T=torch.tensor(copula['time_correlation'],dtype=torch.float64)
        if not torch.allclose(T,T.T,atol=1e-8,rtol=1e-8):raise ValueError('reference time correlation must be symmetric')
        if not torch.allclose(T.diag(),torch.ones(7,dtype=T.dtype),atol=1e-6,rtol=0):raise ValueError('reference time marginal variance must be1')
        unshrunk=(T-.05*torch.eye(7,dtype=T.dtype))/.95
        self.register_buffer('L0',torch.linalg.cholesky(unshrunk))
        p=(rho-self.rho_min)/(self.rho_max-self.rho_min)
        self.register_buffer('rho_logit',torch.tensor(math.log(p/(1-p)),dtype=torch.float64))
        width=137 if view==175 else 98
        self.node=nn.Sequential(nn.Linear(view+2,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,4))
        nn.init.zeros_(self.node[-1].weight);nn.init.zeros_(self.node[-1].bias)
        if view==175:self.time=nn.Parameter(torch.zeros(28))
        else:
            self.time=nn.Sequential(nn.Linear(90,64),nn.SiLU(),nn.Linear(64,28))
            nn.init.zeros_(self.time[-1].weight);nn.init.zeros_(self.time[-1].bias)

    def time_correlation(self,summary):
        B=len(summary)
        delta=(self.time[None].expand(B,-1) if self.view==175 else self.time(summary)).double()
        base=self.L0[self.rows,self.cols]
        values=torch.where((self.rows==self.cols)[None],base[None]*delta.clamp(-1,1).exp(),base[None]+delta)
        L=torch.zeros(B,7,7,device=summary.device,dtype=torch.float64)
        L[:,self.rows,self.cols]=values
        covariance=L@L.transpose(-1,-2);std=covariance.diagonal(dim1=-2,dim2=-1).sqrt()
        correlation=covariance/std[:,:,None]/std[:,None,:]
        return .95*correlation+.05*torch.eye(7,device=summary.device,dtype=torch.float64)

    def state(self,c175,history,static,query_ids):
        if c175.ndim!=3 or c175.shape[-1]!=175:raise ValueError('frozen S175 query condition required')
        B,Q,_=c175.shape
        if len(static)!=Q or len(query_ids)!=Q or len(torch.unique(query_ids))!=Q:
            raise ValueError('static geometry and unique query IDs required')
        summary=weather_summary(history)
        if len(summary)!=B:raise ValueError('history and query origins differ')
        condition=c175 if self.view==175 else torch.cat([c175,summary[:,None].expand(-1,Q,-1)],-1)
        lead=torch.arange(7,device=c175.device,dtype=c175.dtype)/6
        clock=torch.stack([lead,lead.square()],-1)
        output=self.node(torch.cat([condition[:,:,None].expand(-1,-1,7,-1),clock[None,None].expand(B,Q,-1,-1)],-1))
        base=static[:,:3].double()*static.new_tensor([1000/300,1000/300,2],dtype=torch.float64)
        psi=base[None,:,None]+3*torch.tanh(output[...,:3].double()/3)
        rho=self.rho_min+(self.rho_max-self.rho_min)*torch.sigmoid(self.rho_logit+output[...,3].double())
        return psi,rho,self.time_correlation(summary)

    def correlation(self,c175,history,static,query_ids):
        psi,rho,time=self.state(c175,history,static,query_ids);B,Q,H,_=psi.shape
        flat=psi.reshape(B,Q*H,3)
        distance=torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist')
        global_kernel=(-.5*distance.square()).exp()
        rho=rho.reshape(B,-1);common=rho.sqrt()[:,:,None]*rho.sqrt()[:,None,:]
        local=(1-rho).sqrt()[:,:,None]*(1-rho).sqrt()[:,None,:]
        ids=query_ids.repeat_interleave(7);lead=torch.arange(7,device=psi.device).repeat(Q)
        time=time[:,lead[:,None],lead[None,:]]
        return (common*global_kernel+local*(ids[:,None]==ids[None,:])[None])*time

    def query_time_blocks(self,c175,history,static,query_ids):
        psi,rho,time=self.state(c175,history,static,query_ids);B,Q,H,_=psi.shape
        flat=psi.reshape(B*Q,H,3)
        distance=torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist').reshape(B,Q,H,H)
        common=rho.sqrt()[...,None]*rho.sqrt()[:,:,None,:]
        local=(1-rho).sqrt()[...,None]*(1-rho).sqrt()[:,:,None,:]
        return (common*(-.5*distance.square()).exp()+local)*time[:,None]
