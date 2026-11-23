"""Isolated architecture spikes. No changes to production model definitions."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gnss_field.models.grid_generation import ConditionalUNet,ResidualGenerator,ObservationCritic
from gnss_field.models.diffusion import edm_coefficients


class MPConv(nn.Module):
    def __init__(self,inputs,outputs,kernel=3,stride=1):
        super().__init__();self.stride=stride;self.padding=kernel//2
        self.weight=nn.Parameter(torch.randn(outputs,inputs,kernel,kernel))

    def forward(self,x):
        w=self.weight/self.weight.square().sum((1,2,3),keepdim=True).add(1e-8).sqrt()
        return F.conv2d(x,w,stride=self.stride,padding=self.padding)


def mp_sum(a,b):
    return (a+b)/math.sqrt(2)


def mp_cat(a,b):
    total=a.shape[1]+b.shape[1]
    return torch.cat([a*math.sqrt(total/(2*a.shape[1])),b*math.sqrt(total/(2*b.shape[1]))],1)


class MPBlock(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.first=MPConv(inputs,outputs);self.second=MPConv(outputs,outputs)
        self.skip=nn.Identity() if inputs==outputs else MPConv(inputs,outputs,1)

    def forward(self,x):
        return mp_sum(self.skip(x),self.second(F.silu(self.first(F.silu(x)/.596))/.596))


class MPOutput(nn.Module):
    def __init__(self,width):
        super().__init__();self.conv=MPConv(width,1,1);self.gain=nn.Parameter(torch.zeros(()))

    def forward(self,x):return self.gain*self.conv(x)


class ConditionAttention(nn.Module):
    def __init__(self,width,dim=64,heads=4):
        super().__init__();self.heads=heads;self.dim=dim
        self.q=nn.Conv2d(width,dim,1);self.kv=nn.Conv2d(69,2*dim,1);self.out=nn.Conv2d(dim,width,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,condition):
        b,_,h,w=x.shape
        def tokens(z):return z.flatten(2).transpose(1,2).reshape(b,h*w,self.heads,self.dim//self.heads).transpose(1,2)
        k,v=self.kv(F.adaptive_avg_pool2d(condition,(h,w))).chunk(2,1)
        y=F.scaled_dot_product_attention(tokens(self.q(x)),tokens(k),tokens(v))
        y=y.transpose(1,2).reshape(b,h*w,self.dim).transpose(1,2).reshape(b,self.dim,h,w)
        return x+self.out(y)


class PilotUNet(ConditionalUNet):
    def __init__(self,variant,widths,timed):
        super().__init__(widths,timed);self.variant=variant
        if variant=='attention':self.attention=ConditionAttention(widths[-1])
        if variant=='noise':
            self.noise_gain=nn.Parameter(torch.zeros(2))
            self.noise_projection=nn.ModuleList([nn.Conv2d(1,widths[3],1),nn.Conv2d(1,widths[2],1)])
        if variant=='mp':
            self.encoders=nn.ModuleList([MPBlock(71,widths[0])]+[MPBlock(w,w) for w in widths[1:]])
            self.down=nn.ModuleList([MPConv(widths[i],widths[i+1],3,2) for i in range(3)])
            self.decoders=nn.ModuleList([MPBlock(widths[i+1]+widths[i],widths[i]) for i in (2,1,0)])
            self.output=MPOutput(widths[0])

    def forward(self,condition,prior,state,sigma=None):
        x=torch.cat([condition,prior,state],1);embedding=None
        if self.timed:
            sigma=torch.as_tensor(sigma,device=x.device,dtype=x.dtype).reshape(-1)
            if sigma.numel()==1:sigma=sigma.expand(len(x))
            if sigma.shape!=(len(x),) or (sigma<=0).any():raise ValueError('Invalid sigma')
            phase=(sigma.log()/4)[:,None]*self.frequencies[None]
            embedding=self.time_embedding(torch.cat([phase.sin(),phase.cos()],1))
        def mod(value,index):
            if embedding is None:return value
            scale,shift=self.film[index](embedding).chunk(2,1)
            return value*(1+.1*scale.tanh()[:,:,None,None])+.1*shift[:,:,None,None]
        x=mod(self.encoders[0](x),0);levels=[x]
        for i,(down,block) in enumerate(zip(self.down,self.encoders[1:]),1):
            x=mod(block(down(x)),i);levels.append(x)
        if self.variant=='attention':x=self.attention(x,condition)
        if self.variant=='noise':x=x+self.noise_gain[0]*self.noise_projection[0](F.adaptive_avg_pool2d(state,x.shape[-2:]))
        for i,(block,skip) in enumerate(zip(self.decoders,reversed(levels[:-1])),4):
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
            x=mod(block(mp_cat(x,skip) if self.variant=='mp' else torch.cat([x,skip],1)),i)
            if self.variant=='noise' and i==4:
                x=x+self.noise_gain[1]*self.noise_projection[1](F.adaptive_avg_pool2d(state,x.shape[-2:]))
        return self.output(x)


def make_generator(kind,variant='plain',widths=(32,64,128,128),steps=18,recompute=True):
    model=ResidualGenerator(kind,widths=widths,steps=steps,recompute=recompute)
    if variant!='plain':
        # Copy shared paths so attention/noise comparisons share baseline initialization.
        original=model.net.state_dict();model.net=PilotUNet(variant,widths,kind=='diffusion')
        shared=original if variant!='mp' else {k:v for k,v in original.items() if k.startswith(('time_embedding.','film.','frequencies'))}
        model.net.load_state_dict(shared,strict=False)
    return model


class PairCritic(ObservationCritic):
    def __init__(self,width=64,max_edges=96):
        super().__init__(width);self.max_edges=max_edges;self._geometry=None
        self.pair=nn.Sequential(nn.Linear(143,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.relation=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,1))

    def edges(self,coordinates):
        xy=coordinates.detach().cpu().numpy()
        key=xy.tobytes()
        if self._geometry is not None and self._geometry[0]==key:return self._geometry[1]
        # Canonical endpoint ordering and lexicographic tie-breaking preserve station-order invariance.
        order=np.lexsort((xy[:,2],xy[:,1],xy[:,0]));i,j=np.triu_indices(len(xy),1);i,j=order[i],order[j]
        distance=np.linalg.norm(xy[i,:2]-xy[j,:2],axis=1)*400
        rows=[]
        for lo,hi in ((0,50),(50,150),(150,300)):
            keep=np.flatnonzero((distance>=lo)&(distance<hi))
            if len(keep):
                s=np.lexsort((xy[j[keep],2],xy[j[keep],1],xy[j[keep],0],xy[i[keep],2],xy[i[keep],1],xy[i[keep],0],distance[keep]));keep=keep[s]
            rows.append((torch.tensor(i[keep],device=coordinates.device),torch.tensor(j[keep],device=coordinates.device)))
        self._geometry=(key,rows);return rows

    def forward(self,residual,coordinates,condition,mask):
        global_score=super().forward(residual,coordinates,condition,mask)
        if coordinates.ndim!=2:raise ValueError('Pilot pair critic expects fixed physical station geometry')
        r=torch.where(mask,residual,0.);c=torch.where(mask[:,:,None],condition,0.);rows=[]
        for b in range(len(r)):
            bins=[]
            for i,j in self.edges(coordinates):
                use=(mask[b,i]&mask[b,j]).nonzero().flatten()[:self.max_edges];ii,jj=i[use],j[use]
                if not len(use):bins.append(r.new_zeros(self.pair[0].out_features));continue
                delta=coordinates[ii]-coordinates[jj];difference=r[b,ii]-r[b,jj];average=(r[b,ii]+r[b,jj])/2
                f=torch.cat([difference[:,None],average[:,None],delta,c[b,ii],c[b,jj]],-1)
                rev=torch.cat([-difference[:,None],average[:,None],-delta,c[b,jj],c[b,ii]],-1)
                bins.append(((self.pair(f)+self.pair(rev))/2).mean(0))
            rows.append(torch.cat(bins))
        return global_score+self.relation(torch.stack(rows)).squeeze(-1)


class SetDenoiser(nn.Module):
    """NDP-inspired finite-set observation denoiser; no continuous process claim."""
    def __init__(self,attention=True,width=64):
        super().__init__();self.attention=attention
        self.input=nn.Linear(89,width)
        if attention:
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,batch_first=True,norm_first=True) for _ in range(2)])
        else:
            self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width)) for _ in range(2)])
        self.output=nn.Linear(width,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('frequencies',torch.exp(-torch.arange(8,dtype=torch.float32)*.5))

    def denoise(self,state,sigma,condition,coordinates,mask):
        if (mask.sum(1)==0).any():raise ValueError('No valid set tokens')
        state=torch.where(mask,state,0.);condition=torch.where(mask[:,:,None],condition,0.)
        if not torch.isfinite(state).all() or not torch.isfinite(condition).all():raise ValueError('Nonfinite active set input')
        sigma=torch.as_tensor(sigma,device=state.device,dtype=state.dtype).reshape(-1)
        if sigma.numel()==1:sigma=sigma.expand(len(state))
        skip,out,inp,noise,_=edm_coefficients(sigma,1.)
        phase=noise[:,None]*self.frequencies[None];emb=torch.cat([phase.sin(),phase.cos()],-1)[:,None].expand(-1,state.shape[1],-1)
        xy=coordinates[None].expand(len(state),-1,-1) if coordinates.ndim==2 else coordinates
        x=self.input(torch.cat([(state*inp[:,None])[:,:,None],condition,xy,emb],-1))
        for block in self.blocks:
            x=block(x,src_key_padding_mask=~mask) if self.attention else x+block(x)
        value=skip[:,None]*state+out[:,None]*self.output(F.silu(x)).squeeze(-1)
        return torch.where(mask,value,0.)
