"""T1--T5 exact-window causal encoders and multiscale p14 residual fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .p14_fusion import P14Fusion
from .fixed_support import _Conv


def resample_support(field,physical_uv):
    return F.grid_sample(field,physical_uv,padding_mode='border',align_corners=True)


class CausalSpatialEncoder(nn.Module):
    def __init__(self,channels=25,width=64,side=16,layers=2,spatial=True):
        super().__init__();self.side=side;self.width=width
        self.project=nn.Linear(channels,width);self.static=nn.Linear(8,width)
        self.space=nn.ModuleList([nn.Conv2d(width,width,3,padding=1) if spatial else nn.Identity() for _ in range(layers)])
        self.time=nn.ModuleList([nn.TransformerEncoderLayer(width,4,2*width,dropout=0,batch_first=True,norm_first=True) for _ in range(layers)])

    def forward(self,x,static,all_times=False):
        B,T,M,C=x.shape
        if M!=self.side**2:raise ValueError('history support does not match encoder')
        h=self.project(x)+self.static(static)[None,None]
        pos=torch.arange(T,device=x.device,dtype=x.dtype)[:,None]
        freq=torch.exp(torch.arange(0,self.width,2,device=x.device,dtype=x.dtype)*(-math.log(10000)/self.width))
        pe=torch.zeros(T,self.width,device=x.device,dtype=x.dtype);pe[:,0::2]=(pos*freq).sin();pe[:,1::2]=(pos*freq).cos()
        h=h+pe[None,:,None]
        mask=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
        for space,time in zip(self.space,self.time):
            s=h.permute(0,1,3,2).reshape(B*T,self.width,self.side,self.side)
            s=space(s).reshape(B,T,self.width,M).permute(0,1,3,2)
            if not isinstance(space,nn.Identity):h=h+s
            h=time(h.permute(0,2,1,3).reshape(B*M,T,self.width),src_mask=mask)
            h=h.reshape(B,M,T,self.width).permute(0,2,1,3)
        if all_times:return h
        return h[:,-1].transpose(1,2).reshape(B,self.width,self.side,self.side)


class MaskedP14Fusion(nn.Module):
    def __init__(self,kind='T5',length=72,side=16,width=64):
        super().__init__()
        if kind not in ('T0','T1','T2','T3','T4','T5'):raise ValueError(kind)
        self.kind=kind;self.length=length;self.side=side
        self.reference=P14Fusion('features',length)
        if kind=='T0':
            self.legacy=P14Fusion('transformer',length).time
            return
        self.encoder=CausalSpatialEncoder(width=width,side=side,spatial=kind in ('T4','T5'))
        if kind=='T1':self.head=nn.Conv2d(width,1,1)
        else:
            self.t0=nn.Conv2d(width,16,1);self.t1=nn.Conv2d(width,32,1)
            self.a0=_Conv(40,16);self.a1=_Conv(72,32);self.decode=_Conv(48,16);self.head=nn.Conv2d(16,1,1)
            if kind in ('T3','T4','T5'):
                self.gate=nn.Sequential(nn.Conv2d(42,16,1),nn.SiLU(),nn.Conv2d(16,1,1))
                nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-2)
            if kind=='T5':
                self.film=nn.Conv2d(8,32,1);nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,b):
        result=self.reference(b)
        if self.kind=='T0':
            return result+self.legacy(b['history'],b['time_uv'],b['static'],background=torch.zeros_like(result))[:,0]*b.get('time_scale',1.)
        h=self.encoder(b['history'],b['time_static'])
        if self.kind=='T1':field=self.head(h)
        else:
            f0=self.reference.base.proj(b['support']);f1=self.reference.base.down(F.avg_pool2d(f0,2))
            B,_,H,W=f0.shape
            static=b['grid_static'].T.reshape(1,8,H,W).expand(B,-1,-1,-1)
            uv=b['fusion_uv']
            low_uv=F.avg_pool2d(uv.permute(0,3,1,2),2).permute(0,2,3,1)
            a0=self.a0(torch.cat([f0,resample_support(self.t0(h),uv),static],1))
            a1=self.a1(torch.cat([f1,resample_support(self.t1(h),low_uv),F.avg_pool2d(static,2)],1))
            up=F.interpolate(a1,size=(H,W),mode='bilinear',align_corners=False)
            r=self.decode(torch.cat([a0,up],1))
            if self.kind=='T5':
                scale,shift=self.film(static).chunk(2,1);r=r*(1+scale)+shift
            field=self.head(r)
            if hasattr(self,'gate'):
                calendar=b['calendar'][:,:,None,None].expand(-1,-1,H,W)
                field=field*self.gate(torch.cat([b['support'],static,calendar],1)).sigmoid()
        uv=b['time_uv'][:,None] if self.kind=='T1' else b['uv']
        correction=F.grid_sample(field,uv,padding_mode='border',align_corners=True).flatten(1)
        return result+correction
