"""Station-error temporal encoders, terrain queries and native graph v diffusion."""
import math
import torch
from torch import nn
from .native_weather import dct_matrix


class StationEncoder(nn.Module):
    def __init__(self,kind,width=64,length=72):
        super().__init__();self.kind=kind;self.width=width;self.length=length
        self.identity=nn.Embedding(410,8);self.input=nn.Linear(32,width)
        if kind=='transformer':
            layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        elif kind in ('gru','lstm'):
            self.encoder=(nn.GRU if kind=='gru' else nn.LSTM)(width,width,2,batch_first=True)
        else:raise ValueError(kind)
        position=torch.arange(length)[:,None].float()
        frequency=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000)/width))
        p=torch.zeros(length,width);p[:,0::2]=torch.sin(position*frequency);p[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position',p)

    def forward(self,history,context_static,context_ids):
        B,L,C,_=history.shape
        if L>self.length:raise ValueError('longer history than registered encoder')
        st=context_static[None,None].expand(B,L,-1,-1)
        identity=self.identity(context_ids)[None,None].expand(B,L,-1,-1)
        h=self.input(torch.cat([history,st,identity],-1)).permute(0,2,1,3).reshape(B*C,L,self.width)
        h=h+self.position[-L:][None]
        if self.kind=='transformer':
            mask=torch.ones(L,L,device=h.device,dtype=torch.bool).triu(1)
            h=self.encoder(h,mask=mask)
        else:h=self.encoder(h)[0]
        return h.reshape(B,C,L,self.width)


class NativeMean(nn.Module):
    def __init__(self,kind='transformer',width=64,length=72,weather=True):
        super().__init__();self.kind=kind;self.weather=weather
        self.query=nn.Sequential(nn.Linear(111,width),nn.SiLU(),nn.Linear(width,width))
        if kind!='weather':
            self.time=StationEncoder(kind,width,length)
            self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
            self.edge=nn.Sequential(nn.Linear(7,32),nn.SiLU(),nn.Linear(32,4+width))
            self.fusion=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.LayerNorm(width))
        self.head=nn.Linear(width,7);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('limits',torch.tensor([5.,30.,30.,30.,30.,30.,30.]))

    def forward(self,condition,history,context_static,context_ids,query_static):
        if not self.weather:
            condition=condition.clone();condition[:,:,24:96]=0
        query=self.query(condition);h=query
        if self.kind!='weather':
            states=self.time(history,context_static,context_ids)[:,:,-1]
            xy=(query_static[:,None,:2]-context_static[None,:,:2])*1000/300
            distance=xy.square().sum(-1).sqrt()*300
            edge=torch.cat([xy,distance.log1p()[...,None]/6,
                (query_static[:,None,2:3]-context_static[None,:,2:3])*3,
                query_static[:,None,8:9]-context_static[None,:,8:9],
                query_static[:,None,3:4]-context_static[None,:,3:4],
                query_static[:,None,5:6]-context_static[None,:,5:6]],-1)
            encoded=self.edge(edge);bias,values=encoded[:,:,:4],encoded[:,:,4:]
            B,Q,W=query.shape;C=len(context_ids);D=W//4
            score=torch.einsum('bqhd,bchd->bqch',query.reshape(B,Q,4,D),self.key(states).reshape(B,C,4,D))/math.sqrt(D)
            score=score+bias[None]-torch.log1p(distance[None,:,:,None]/50)
            available=(history[:,:,:,3].sum(1)>0)
            score=score.masked_fill(~available[:,None,:,None],-1e4)
            weight=score.softmax(2)*available[:,None,:,None]
            weight=weight/weight.sum(2,keepdim=True).clamp_min(1e-8)
            value=self.value(states)[:,None]+values[None]
            message=(weight[...,None]*value.reshape(B,Q,C,4,D)).sum(2).reshape(B,Q,W)
            h=self.fusion(torch.cat([query,message],-1))
        raw=self.head(h)
        return self.limits*torch.tanh(raw/self.limits),h


class NativeDiffusion(nn.Module):
    def __init__(self,condition_dim=64,width=64,domain='spatial'):
        super().__init__()
        if domain not in ('spatial','graph','joint'):raise ValueError(domain)
        self.domain=domain
        self.input=nn.Linear(7+condition_dim+5,width)
        self.local=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        self.filters=nn.ModuleList([nn.ModuleList([nn.Linear(width,width,bias=False) for _ in range(3)]) for _ in range(2)])
        self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.output=nn.Linear(width,7)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        t=torch.linspace(0,1,101);a=torch.cos((t+.008)/1.008*math.pi/2).square()
        self.register_buffer('alpha',(a/a[0]).clamp_min(1e-5))
        self.register_buffer('D',dct_matrix(7))

    def denoise(self,x,c,k,U,eigen):
        if self.domain=='joint':x=x@self.D.T
        clock=torch.stack([k,k.sin(),k.cos(),(k*math.pi).sin(),(k*math.pi).cos()],-1)
        h=self.input(torch.cat([x,c,clock[:,None].expand(-1,x.shape[1],-1)],-1))
        for local,filters,norm in zip(self.local,self.filters,self.norm):
            frequency=torch.einsum('qm,bqw->bmw',U,h)
            if self.domain=='spatial':
                # Degree-normalized adjacency powers: local polynomial graph messages.
                bands=[torch.ones_like(eigen),1-eigen,(1-eigen).square()]
            else:
                # Complete GFT; smooth spectral responses, no truncation/eigenvector phase dependence.
                bands=[torch.ones_like(eigen),torch.exp(-eigen),torch.exp(-4*eigen)]
            message=sum(layer(torch.einsum('qm,bmw->bqw',U,frequency*band[None,:,None])) for layer,band in zip(filters,bands))
            h=norm(h+torch.nn.functional.silu(local(h)+message))
        result=self.output(h)
        return result@self.D if self.domain=='joint' else result

    def loss(self,x,c,U,eigen):
        k=torch.randint(1,101,(len(x),),device=x.device);eps=torch.randn_like(x)
        a=self.alpha[k,None,None]
        noisy=a.sqrt()*x+(1-a).sqrt()*eps
        velocity=a.sqrt()*eps-(1-a).sqrt()*x
        return (self.denoise(noisy,c,k/100,U,eigen)-velocity).square().mean()

    @torch.no_grad()
    def sample(self,c,noise,U,eigen,steps=20,solver='ddim'):
        if solver not in ('ddim','heun'):raise ValueError(solver)
        x=noise.clone();schedule=torch.linspace(100,0,steps+1,device=x.device).round().long()
        for k,j in zip(schedule[:-1],schedule[1:]):
            v=self.denoise(x,c,torch.full((len(x),),float(k)/100,device=x.device),U,eigen)
            if solver=='heun':
                # Stochastic-interpolant probability flow: dX/dtheta=E[v|X,c],
                # theta=acos(sqrt(alpha)). A zero velocity preserves N(0,I).
                delta=self.alpha[j].sqrt().acos()-self.alpha[k].sqrt().acos()
                proposal=x+delta*v
                second=self.denoise(proposal,c,torch.full((len(x),),float(j)/100,device=x.device),U,eigen)
                x=x+.5*delta*(v+second)
                continue
            a=self.alpha[k];clean=a.sqrt()*x-(1-a).sqrt()*v
            eps=(1-a).sqrt()*x+a.sqrt()*v
            x=self.alpha[j].sqrt()*clean+(1-self.alpha[j]).sqrt()*eps
        return x


class NativeGaussian(nn.Module):
    def __init__(self,condition_dim=64,width=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(condition_dim,width),nn.SiLU(),nn.Linear(width,14))

    def forward(self,c):
        mean,logstd=self.net(c).chunk(2,-1)
        return mean,logstd.clamp(-3,2.5)

    def loss(self,x,c,U=None,eigen=None):
        mean,logstd=self(c)
        return (.5*((x-mean)*(-logstd).exp()).square()+logstd).mean()

    def sample(self,c,noise,U=None,eigen=None,steps=20,solver='ddim'):
        mean,logstd=self(c);return mean+logstd.exp()*noise
