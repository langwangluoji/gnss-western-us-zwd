"""Causal geometric conditional diffusion of surface ZWD."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as recompute


def schedule(n=100):
    t=torch.arange(n+1,dtype=torch.float64)/n
    a=torch.cos((t+.008)/1.008*math.pi/2).square();a=a/a[0]
    beta=(1-a[1:]/a[:-1]).clamp(max=.999)
    return torch.cat([torch.ones(1,dtype=torch.float64),(1-beta).cumprod(0)]).float()


def v_target(x,noise,a):return a.sqrt()*noise-(1-a).sqrt()*x


def recover(x,v,a):return a.sqrt()*x-(1-a).sqrt()*v,(1-a).sqrt()*x+a.sqrt()*v


class GeoAttention(nn.Module):
    def __init__(self,width,k,radius):
        super().__init__();self.k=k;self.radius=radius;self.heads=4;self.d=width//4
        self.q=nn.Linear(width,width);self.kv=nn.Linear(width,width*2);self.out=nn.Linear(width,width)
        self.edge=nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,4))

    def forward(self,x,y,g,h,global_tokens=False):
        b,n,w=x.shape;m=y.shape[1]
        delta=g[:,:,None,:]-h[:,None,:,:]
        dist=delta[...,:2].square().sum(-1).sqrt()
        edge=torch.cat([delta[...,:2]/300,delta[...,2:3],delta[...,3:4],dist[...,None]/300],-1)
        # Always keep nearest support, even outside radius, to avoid empty softmax.
        rank=dist.argsort(-1).argsort(-1)
        allowed=((rank<self.k)&(dist<=self.radius))|(rank==0)
        q=self.q(x).view(b,n,4,self.d).transpose(1,2)
        k,v=self.kv(y).chunk(2,-1)
        k=k.view(b,m,4,self.d).transpose(1,2);v=v.view(b,m,4,self.d).transpose(1,2)
        score=q@k.transpose(-1,-2)/math.sqrt(self.d)+self.edge(edge).permute(0,3,1,2)
        score=score.masked_fill(~allowed[:,None],-torch.inf)
        if global_tokens:
            kg=k.mean(2,keepdim=True);vg=v.mean(2,keepdim=True)
            # Two invariant summaries: mean and featurewise maximum.
            kg=torch.cat([kg,k.amax(2,keepdim=True)],2);vg=torch.cat([vg,v.amax(2,keepdim=True)],2)
            score=torch.cat([score,q@kg.transpose(-1,-2)/math.sqrt(self.d)],-1);v=torch.cat([v,vg],2)
        return self.out((score.softmax(-1)@v).transpose(1,2).reshape(b,n,w))


class Block(nn.Module):
    def __init__(self,w):
        super().__init__();self.cross=GeoAttention(w,16,600);self.graph=GeoAttention(w,12,300)
        self.norm=nn.ModuleList([nn.LayerNorm(w) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(w,w*2),nn.GELU(),nn.Linear(w*2,w))
        self.mod=nn.Linear(w,w*2)

    def forward(self,x,c,qg,cg,step):
        scale,shift=self.mod(step).chunk(2,-1)
        z=self.norm[0](x)*(1+scale[:,None])+shift[:,None]
        x=x+self.cross(z,c,qg,cg,True)
        z=self.norm[1](x);x=x+self.graph(z,z,qg,qg)
        return x+self.ff(self.norm[2](x))


class DiffusionModel(nn.Module):
    def __init__(self,width=64,blocks=4):
        super().__init__();self.width=width
        self.context=nn.Sequential(nn.Linear(38,width),nn.SiLU(),nn.Linear(width,width))
        self.spatial=nn.ModuleList([GeoAttention(width,12,600) for _ in range(2)])
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.query=nn.Sequential(nn.Linear(42,width),nn.SiLU(),nn.Linear(width,width))
        self.history=nn.GRU(3,width,batch_first=True);self.fuse=nn.Linear(width*2,width)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([Block(width) for _ in range(blocks)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        self.scale=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        self.register_buffer('ab',schedule())

    def encode(self,context,cgeo):
        b,l,n,_=context.shape;z=self.context(context).reshape(b*l,n,self.width)
        g=cgeo[:,None].expand(b,l,n,4).reshape(b*l,n,4)
        for layer in self.spatial:z=z+layer(z,z,g,g)
        z=z.reshape(b,l,n,self.width).transpose(1,2).reshape(b*n,l,self.width)
        causal=torch.ones(l,l,dtype=torch.bool,device=z.device).triu(1)
        z=self.temporal(z,mask=causal)
        return z.reshape(b,n,l,self.width).transpose(1,2)

    def features(self,x,step,d,encoded=None):
        c=self.encode(d['context'],d['cgeo']) if encoded is None else encoded
        b,n=x.shape;hist=d['history'].permute(0,2,1,3).reshape(b*n,6,3)
        _,h=self.history(hist);h=h[-1].reshape(b,n,self.width)
        z=self.fuse(torch.cat([self.query(torch.cat([x[...,None],d['query']],-1)),h],-1))
        freq=torch.exp(-math.log(10000)*torch.arange(self.width//2,device=x.device)/(self.width//2))
        phase=step[:,None]*freq[None];s=self.time(torch.cat([phase.sin(),phase.cos()],-1))
        for block in self.blocks:z=block(z,c[:,-1],d['qgeo'],d['cgeo'],s)
        return z

    def forward(self,x,step,d,encoded=None):return self.head(self.features(x,step,d,encoded)).squeeze(-1)

    def baseline(self,d):
        z=self.features(torch.zeros_like(d['target']),torch.zeros(d['target'].shape[0],device=d['target'].device),d)
        return self.head(z).squeeze(-1),self.scale(z).squeeze(-1).clamp(-5,3).exp()

    def sample(self,d,noise,steps=20,checkpoint=False):
        # Freeze the dictionary binding: caller advances history after sampling;
        # checkpoint recomputation must see this hour's original history.
        d=dict(d)
        c=self.encode(d['context'],d['cgeo']);x=noise
        times=torch.linspace(100,1,steps,device=x.device).round().long()
        for i,t in enumerate(times):
            step=t.expand(x.shape[0])
            if checkpoint and torch.is_grad_enabled():
                v=recompute(lambda a,b,e:self(a,b,d,e),x,step,c,use_reentrant=False)
            else:v=self(x,step,d,c)
            clean,eps=recover(x,v,self.ab[t]);nxt=times[i+1] if i+1<len(times) else 0
            x=self.ab[nxt].sqrt()*clean+(1-self.ab[nxt]).sqrt()*eps
        return x
