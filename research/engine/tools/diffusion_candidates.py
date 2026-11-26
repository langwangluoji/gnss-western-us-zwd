"""Paired conditional Gaussian, diffusion MLP, DiT and sparse-FFN DiT."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
import graph_fusion_spike as old
import masked_graph_spike as masked
from gnss_field.artifacts import write_json,sha256

ROOT=old.core.PROJECT/'results/20261123_diffusion_v07'
KINDS=('gaussian','diffusion','transformer','sparse')


def spatial_basis(a):
    xy=old.XY
    anchors=np.array([(x,y) for y in np.linspace(xy[:,1].min(),xy[:,1].max(),4) for x in np.linspace(xy[:,0].min(),xy[:,0].max(),5)])
    dist=cdist(xy,anchors)/1000
    radial=np.concatenate([np.exp(-.5*(dist/r)**2) for r in (200,500)],-1)
    raw=np.concatenate([radial,radial*np.clip(old.TERRAIN[:,0:1],0,2)],-1)
    queries=(raw[old.QI]*old.QW[...,None]).sum(1)*np.exp(-a['heights']/3000)[:,None]
    _,s,v=np.linalg.svd(queries[a['split_fit']],full_matrices=False)
    transform=v[:24].T/s[:24]
    return anchors,(raw@transform).astype('f4'),(queries@transform).astype('f4'),s.tolist()


def condition(a,name,anchors):
    raw=a[name+'_raw'];valid=np.isfinite(raw).all(-1);safe=np.where(valid[...,None],raw,0)
    w=np.exp(-.5*(cdist(anchors,a['xy'][a['aids']])/300000)**2)
    den=np.einsum('qa,bta->btq',w,valid)
    avg=np.einsum('qa,btav->btqv',w,safe)/np.maximum(den[...,None],1e-8)
    now=(avg[:,6:]-[100,280,900])/[50,10,80]
    delta1=(avg[:,6:,:,0]-avg[:,5:29,:,0])/50
    delta3=(avg[:,6:,:,0]-avg[:,3:27,:,0])/50
    tok=np.concatenate([now,delta1[...,None],delta3[...,None],np.log1p(den[:,6:])[...,None]],-1)
    hours=a[name+'_hours'][:,6:];day=hours/24
    cyc=np.stack([np.sin(hours*2*np.pi/24),np.cos(hours*2*np.pi/24),np.sin(day*2*np.pi/365.25),np.cos(day*2*np.pi/365.25)],-1)
    return np.concatenate([tok.reshape(*tok.shape[:2],120),cyc],-1).astype('f4')


def prepare(a,name,seed,anchors,query):
    d=old.prepare(a,name,seed)
    out={k:d[k] for k in ('base','y','mask')};out['c']=torch.tensor(condition(a,name,anchors),device='cuda')
    if name=='train':
        ids=a['split_fit'];q=query[ids].astype('f8');m=d['mask'][:,:,ids].cpu().numpy().reshape(-1,len(ids)).astype('f8')
        r=(d['y'][:,:,ids]-d['base'][:,:,ids]).cpu().numpy().reshape(-1,len(ids))
        lhs=np.einsum('ni,bn,nj->bij',q,m,q)+.1*np.eye(24)
        rhs=np.einsum('ni,bn->bi',q,m*r)
        z=np.linalg.solve(lhs,rhs[...,None])[...,0]
        mean=z.mean(0);std=np.maximum(z.std(0),1.)
        reconstruction=z@q.T
        out['latent_diagnostics']=dict(residual_rmse=float(np.sqrt((r*r*m).sum()/m.sum())),projection_rmse=float(np.sqrt(((r-reconstruction)**2*m).sum()/m.sum())))
        out['zm']=torch.tensor(mean,dtype=torch.float32,device='cuda');out['zs']=torch.tensor(std,dtype=torch.float32,device='cuda')
        out['z']=torch.tensor(((z-mean)/std).reshape(len(d['base']),24,24),dtype=torch.float32,device='cuda')
    return out


def schedule():
    k=torch.arange(101,device='cuda',dtype=torch.float32)
    a=torch.cos((k/100+.008)/1.008*np.pi/2)**2
    return (a/a[0]).clamp_min(1e-7)


def embedding(k):
    freq=torch.exp(torch.arange(8,device=k.device)*(-np.log(10000.)/7))
    v=k[:,None]*freq[None]
    return torch.cat([v.sin(),v.cos()],-1)


class SparseFFN(nn.Module):
    def __init__(self):
        super().__init__();self.router=nn.Linear(64,4)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64)) for _ in range(4)])

    def forward(self,x):
        prob=self.router(x).softmax(-1);values,idx=prob.topk(2,dim=-1)
        gate=torch.zeros_like(prob).scatter_(-1,idx,values/values.sum(-1,keepdim=True))
        out=sum(expert(x)*gate[...,i,None] for i,expert in enumerate(self.experts))
        used=(gate>0).float().mean((0,1))/2
        self.utilization=used.detach()
        balance=4*(used*prob.mean((0,1))).sum()
        return out,balance


class Block(nn.Module):
    def __init__(self,sparse):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(3)])
        self.selfatt=nn.MultiheadAttention(64,4,batch_first=True)
        self.cross=nn.MultiheadAttention(64,4,batch_first=True)
        self.ff=SparseFFN() if sparse else nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))

    def forward(self,x,context):
        q=self.norms[0](x);x=x+self.selfatt(q,q,q,need_weights=False)[0]
        q=self.norms[1](x);x=x+self.cross(q,context,context,need_weights=False)[0]
        if isinstance(self.ff,SparseFFN):out,aux=self.ff(self.norms[2](x))
        else:out=self.ff(self.norms[2](x));aux=x.new_zeros(())
        return x+out,aux


class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind in ('gaussian','diffusion'):
            self.cond=nn.Sequential(nn.Linear(124,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.net=nn.Sequential(nn.Linear(129,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,48 if kind=='gaussian' else 24))
        else:
            self.value=nn.Linear(1,64);self.previous=nn.Linear(1,64);self.time=nn.Linear(17,64)
            self.position=nn.Parameter(torch.randn(24,64)*.02)
            self.context=nn.Linear(12,64)
            self.blocks=nn.ModuleList([Block(kind=='sparse') for _ in range(2)])
            self.out=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,1))
            pos=torch.tensor([(x,y) for y in np.linspace(-1,1,4) for x in np.linspace(-1,1,5)],dtype=torch.float32)
            self.register_buffer('anchor_pos',pos)

    def forward(self,x,prev,c,k,first):
        time=torch.cat([embedding(k),first[:,None]],-1)
        if self.kind in ('gaussian','diffusion'):
            out=self.net(torch.cat([x,prev,self.cond(c),time],-1));return out,out.new_zeros(())
        q=self.value(x[...,None])+self.position[None]+self.time(time)[:,None]
        tok=c[:,:120].reshape(-1,20,6)
        context=self.context(torch.cat([tok,self.anchor_pos[None].expand(len(c),-1,-1),c[:,None,120:].expand(-1,20,-1)],-1))
        context=torch.cat([context,self.previous(prev[...,None])+self.position[None]],1)
        aux=x.new_zeros(())
        for block in self.blocks:q,term=block(q,context);aux=aux+term
        return self.out(q).squeeze(-1),aux/2


@torch.no_grad()
def sample(model,c,k=64,noise_seed=9301):
    torch.manual_seed(noise_seed);b,t,_=c.shape
    # Pre-draw all noise so batching and causal-prefix checks can share draws.
    noise=torch.randn(b,k,t,24,device='cuda');prev=torch.zeros(b*k,24,device='cuda');seq=[];clips=0;total=0
    ab=schedule();steps=np.rint(np.linspace(100,1,20)).astype(int)
    for hour in range(t):
        cond=c[:,hour,None].expand(b,k,-1).reshape(b*k,-1);first=torch.full((b*k,),float(hour==0),device='cuda')
        x=noise[:,:,hour].reshape(b*k,24)
        if model.kind=='gaussian':
            out,_=model(torch.zeros_like(x),prev,cond,torch.zeros(b*k,device='cuda'),first)
            mu,logstd=out.chunk(2,-1);z=mu+logstd.clamp(-4,2).exp()*x
        else:
            for j,step in enumerate(steps):
                next_step=steps[j+1] if j+1<len(steps) else 0
                v,_=model(x,prev,cond,torch.full((b*k,),float(step),device='cuda'),first)
                a=ab[step];n=ab[next_step]
                z=a.sqrt()*x-(1-a).sqrt()*v
                clips+=(z.abs()>6).sum().item();total+=z.numel();z=z.clamp(-6,6)
                eps=(x-a.sqrt()*z)/(1-a).sqrt()
                x=n.sqrt()*z+(1-n).sqrt()*eps
            z=x
        prev=z;seq.append(z.reshape(b,k,24))
    return torch.stack(seq,2),clips/max(total,1)


@torch.no_grad()
def evaluate(model,d,train,query,ids,k=64,noise_seed=9301):
    z,clip=sample(model,d['c'],k,noise_seed)
    z=z*train['zs']+train['zm']
    s=d['base'][:,:,ids][:,None]+torch.einsum('bktl,nl->bktn',z,query[ids])
    y=d['y'][:,:,ids].cpu().numpy();m=d['mask'][:,:,ids].cpu().numpy();samples=s.cpu().numpy()
    score=old.inv.probability_score(samples,y,m)
    score.update(point=old.inv.point_score(samples.mean(1),y,m),clip_fraction=clip)
    return score,dict(samples=samples,prediction=samples.mean(1),target=y,mask=m,latent=z.cpu().numpy())


def fit(ds,a,query,seed,kind):
    torch.manual_seed(seed);model=Model(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=torch.Generator(device='cuda').manual_seed(seed+1000)
    z=ds['train']['z'];prev=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1]],1).reshape(-1,24)
    first=torch.zeros(z.shape[:2],device='cuda');first[:,0]=1;first=first.flatten()
    z=z.reshape(-1,24);c=ds['train']['c'].reshape(-1,124);ab=schedule();best=float('inf');trace=[];started=time.monotonic()
    for step in range(1,1501):
        idx=torch.randint(len(z),(128,),device='cuda',generator=rng);target=z[idx]
        k=torch.randint(1,101,(128,),device='cuda',generator=rng);eps=torch.randn(target.shape,device='cuda',generator=rng)
        if kind=='gaussian':
            out,aux=model(torch.zeros_like(target),prev[idx],c[idx],torch.zeros(128,device='cuda'),first[idx])
            mu,logstd=out.chunk(2,-1);logstd=logstd.clamp(-4,2)
            loss=(.5*((target-mu)/logstd.exp())**2+logstd).mean()
        else:
            ak=ab[k,None]
            x=ak.sqrt()*target+(1-ak).sqrt()*eps;v=ak.sqrt()*eps-(1-ak).sqrt()*target
            pred,aux=model(x,prev[idx],c[idx],k.float(),first[idx]);loss=((pred-v)**2).mean()+.01*aux
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%300==0:
            with torch.random.fork_rng(devices=[0]):score,_=evaluate(model,ds['inner'],ds['train'],query,a['split_inner'],16)
            crit=score['crps']+.5*score['increment_crps'];trace.append(dict(step=step,loss=float(loss.detach()),inner=score))
            if crit<best:best=crit;state=copy.deepcopy(model.state_dict());selected=step
            print(json.dumps(dict(seed=seed,kind=kind,step=step,best=best)),flush=True)
    model.load_state_dict(state);torch.save(dict(model=state,selected_step=selected),ROOT/f'{kind}_{seed}.pt')
    return dict(trace=trace,selected_step=selected,seconds=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()))


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);started=time.monotonic()
    a=dict(np.load(old.exp.ROOT/'data.npz'));source=json.loads((old.exp.ROOT/'results.json').read_text())
    assert sha256(old.exp.ROOT/'data.npz')==source['runtime']['data_sha256']
    anchors,grid,query,singular=spatial_basis(a);qt=torch.tensor(query,device='cuda')
    np.savez(ROOT/'geometry.npz',anchors=anchors,grid_basis=grid,station_basis=query,singular=singular)
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),architecture_sha256=sha256(ROOT/'ARCHITECTURE.md'),
        data_sha256=source['runtime']['data_sha256'],geometry_sha256=sha256(ROOT/'geometry.npz'),v04_selection_sha256=sha256(old.inv.ROOT/'selection.lock.json'),
        source_checkpoints=source['runtime']['checkpoints'],dependencies={n:sha256(old.core.PROJECT/'tools'/n) for n in ['graph_fusion_spike.py','masked_graph_spike.py','sequence_innovation_spike.py','explore_sequence_mechanisms.py','sequence_grid_spike.py','sequence_mechanism_spike.py']},C_D_labels_read=False))
    selection={};normalization={};diagnostics={}
    for seed in (42,17):
        ds={name:prepare(a,name,seed,anchors,query) for name in ('train','inner')}
        normalization[seed]={k:ds['train'][k] for k in ('zm','zs')};diagnostics[seed]=ds['train']['latent_diagnostics']
        torch.save(normalization[seed],ROOT/f'normalization_{seed}.pt')
        for kind in KINDS:selection[f'{kind}_{seed}']=fit(ds,a,qt,seed,kind)
        del ds;torch.cuda.empty_cache()
    write_json(ROOT/'selection.lock.json',selection);masked.add_reserved(a);results={}
    for seed in (42,17):
        for name in ('confirmation','reserved'):
            d=prepare(a,name,seed,anchors,query);ids=a['split_outer'];ref=np.load(old.inv.ROOT/f'history_{seed}_{name}.npz')
            assert np.array_equal(ref['mask'],d['mask'][:,:,ids].cpu().numpy())
            assert np.max(np.abs(ref['prediction']-d['base'][:,:,ids].cpu().numpy()))<1e-4
            for kind in KINDS:
                model=Model(kind).cuda();model.load_state_dict(torch.load(ROOT/f'{kind}_{seed}.pt',weights_only=False)['model'])
                score,p=evaluate(model,d,normalization[seed],qt,ids);check,_=evaluate(model,d,normalization[seed],qt,ids,128,11703)
                key=f'{kind}_{seed}_{name}';results[key]=dict(k64=score,k128=check)
                if kind=='sparse':results[key]['expert_utilization_last_call']=[b.ff.utilization.cpu().tolist() for b in model.blocks]
                np.savez_compressed(ROOT/f'{key}.npz',**p,station_ids=ids,hours=ref['hours'])
                print(json.dumps(dict(stage='eval',key=key,k128=check)),flush=True)
    write_json(ROOT/'results.json',dict(results=results,selection=selection,latent_diagnostics=diagnostics,runtime=dict(seconds=time.monotonic()-started,device=torch.cuda.get_device_name(),
        code_sha256=sha256(__file__),selection_sha256=sha256(ROOT/'selection.lock.json'),checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')})))


if __name__=='__main__':main()
