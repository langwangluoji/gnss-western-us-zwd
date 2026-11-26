"""Nested multiscale DiT trained through DDIM in observation space."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
import diffusion_candidates as d
from gnss_field.artifacts import write_json,sha256

ROOT=d.old.core.PROJECT/'results/20261123_diffusion_observation_v09'


class Model(d.Model):
    def __init__(self,rank):
        super().__init__('transformer')
        if rank>24:self.position=nn.Parameter(torch.cat([self.position.detach(),torch.randn(rank-24,64)*.02]))
        self.rank=rank


def geometry(a):
    source=dict(np.load(d.ROOT/'geometry.npz'));g24=source['grid_basis'].astype('f8');q24=source['station_basis'].astype('f8')
    xy=d.old.XY;anchors=np.array([(x,y) for y in np.linspace(xy[:,1].min(),xy[:,1].max(),8) for x in np.linspace(xy[:,0].min(),xy[:,0].max(),12)])
    dist=cdist(xy,anchors)/1000;radial=np.concatenate([np.exp(-.5*(dist/r)**2) for r in (75,150,300)],-1)
    raw=np.concatenate([radial,radial*np.clip(d.old.TERRAIN[:,0:1],0,2)],-1)
    qraw=(raw[d.old.QI]*d.old.QW[...,None]).sum(1)*np.exp(-a['heights']/3000)[:,None]
    ids=a['split_fit'];projection=np.linalg.lstsq(q24[ids],qraw[ids],rcond=None)[0]
    rg=raw-g24@projection;rq=qraw-q24@projection
    _,s,v=np.linalg.svd(rq[ids],full_matrices=False);transform=v[:40].T/s[:40]
    g=np.concatenate([g24,rg@transform],-1).astype('f4');q=np.concatenate([q24,rq@transform],-1).astype('f4')
    return source['anchors'],g,q


def targets(base,a,q):
    rank=q.shape[1];ids=a['split_fit'];phi=q[ids].astype('f8')
    mask=base['mask'][:,:,ids].cpu().numpy().reshape(-1,len(ids)).astype('f8')
    residual=(base['y'][:,:,ids]-base['base'][:,:,ids]).cpu().numpy().reshape(-1,len(ids))
    lhs=np.einsum('ni,bn,nj->bij',phi,mask,phi)+.1*np.eye(rank)
    rhs=np.einsum('ni,bn->bi',phi,mask*residual);z=np.linalg.solve(lhs,rhs[...,None])[...,0]
    mean=z.mean(0);std=np.maximum(z.std(0),1.);reconstruction=z@phi.T
    return dict(z=torch.tensor(((z-mean)/std).reshape(-1,24,rank),dtype=torch.float32,device='cuda'),
      zm=torch.tensor(mean,dtype=torch.float32,device='cuda'),zs=torch.tensor(std,dtype=torch.float32,device='cuda'),
      diagnostic=dict(residual_rmse=float(np.sqrt((residual**2*mask).sum()/mask.sum())),projection_rmse=float(np.sqrt(((residual-reconstruction)**2*mask).sum()/mask.sum()))))


def sample(model,c,noise,prev=None,start=0):
    b,k,t,rank=noise.shape
    if prev is None:prev=torch.zeros(b*k,rank,device=c.device)
    else:prev=prev[:,None].expand(b,k,rank).reshape(b*k,rank)
    ab=d.schedule();steps=np.rint(np.linspace(100,1,6)).astype(int);seq=[];clips=c.new_zeros(())
    for hour in range(t):
        cond=c[:,hour,None].expand(b,k,-1).reshape(b*k,-1);first=c.new_full((b*k,),float(hour+start==0));x=noise[:,:,hour].reshape(b*k,rank)
        for j,step in enumerate(steps):
            nxt=steps[j+1] if j+1<len(steps) else 0
            v,_=model(x,prev,cond,c.new_full((b*k,),float(step)),first)
            alpha=ab[step];next_alpha=ab[nxt];z=alpha.sqrt()*x-(1-alpha).sqrt()*v
            clips=clips+(z.detach().abs()>6).sum();z=z.clamp(-6,6)
            eps=(x-alpha.sqrt()*z)/(1-alpha).sqrt();x=next_alpha.sqrt()*z+(1-next_alpha).sqrt()*eps
        prev=x;seq.append(x.reshape(b,k,rank))
    return torch.stack(seq,2),clips/(noise.numel()*len(steps))


def decode(z,data,norm,q,ids):
    return data['base'][:,:,ids][:,None]+torch.einsum('bktl,nl->bktn',z*norm['zs']+norm['zm'],q[ids])


@torch.no_grad()
def evaluate(model,data,norm,q,ids,k=64,seed=9301):
    rng=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(len(data['c']),k,24,model.rank,device='cuda',generator=rng)
    z,clip=sample(model,data['c'],noise);s=decode(z,data,norm,q,ids).cpu().numpy()
    y=data['y'][:,:,ids].cpu().numpy();mask=data['mask'][:,:,ids].cpu().numpy()
    score=d.old.inv.probability_score(s,y,mask);score.update(point=d.old.inv.point_score(s.mean(1),y,mask),clip_fraction=float(clip))
    return score,dict(samples=s,prediction=s.mean(1),target=y,mask=mask,latent=(z*norm['zs']+norm['zm']).cpu().numpy())


def vloss(model,train,rng):
    rank=model.rank;z=train['z'];p=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1]],1).reshape(-1,rank)
    first=torch.zeros(z.shape[:2],device='cuda');first[:,0]=1
    idx=torch.randint(z.shape[0]*24,(128,),device='cuda',generator=rng);target=z.reshape(-1,rank)[idx]
    k=torch.randint(1,101,(128,),device='cuda',generator=rng);eps=torch.randn(target.shape,device='cuda',generator=rng);alpha=d.schedule()[k,None]
    x=alpha.sqrt()*target+(1-alpha).sqrt()*eps;v=alpha.sqrt()*eps-(1-alpha).sqrt()*target
    pred,_=model(x,p[idx],train['c'].reshape(-1,124)[idx],k.float(),first.flatten()[idx])
    return ((pred-v)**2).mean()


def pretrain(ds,a,q,seed,rank):
    torch.manual_seed(seed);model=Model(rank).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=torch.Generator(device='cuda').manual_seed(seed+1000);best=float('inf');trace=[];start=time.monotonic()
    for step in range(1,1201):
        loss=vloss(model,ds['train'],rng);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%300==0:
            score,_=evaluate(model,ds['inner'],ds['train'],q,a['split_inner'],16)
            criterion=score['crps']+.5*score['increment_crps'];trace.append(dict(step=step,loss=float(loss.detach()),inner=score))
            if criterion<best:best=criterion;state=copy.deepcopy(model.state_dict());selected=step
            print(json.dumps(dict(stage='pretrain',seed=seed,rank=rank,step=step,best=best)),flush=True)
    torch.save(dict(model=state,selected_step=selected),ROOT/f'r{rank}_pre_{seed}.pt')
    return state,dict(trace=trace,selected_step=selected,seconds=time.monotonic()-start)


def finetune(state,ds,a,q,seed,rank,temporal):
    model=Model(rank).cuda();model.load_state_dict(state);opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    rng=torch.Generator(device='cuda').manual_seed(seed+2000);rows=np.random.default_rng(seed+2000);train=ds['train'];ids=a['split_fit']
    score,_=evaluate(model,ds['inner'],train,q,a['split_inner'],16);best=score['crps']+.5*score['increment_crps']
    chosen=copy.deepcopy(state);selected=0;trace=[dict(step=0,inner=score)];start_clock=time.monotonic()
    for step in range(1,201):
        row=int(rows.integers(len(train['c'])));start=int(rows.integers(21));end=start+4
        sub={k:train[k][row:row+1,start:end] for k in ('c','base','y','mask')}
        history=train['z'][row:row+1,start-1] if start else None
        noise=torch.randn(1,4,4,rank,device='cuda',generator=rng)
        z,_=sample(model,sub['c'],noise,history,start);s=decode(z,sub,train,q,ids)
        y=sub['y'][:,:,ids];mask=sub['mask'][:,:,ids]
        marginal=d.old.core.fair_crps(s,y,mask)
        increment=d.old.core.fair_crps(s[:,:,1:]-s[:,:,:-1],y[:,1:]-y[:,:-1],mask[:,1:]&mask[:,:-1])
        regularizer=vloss(model,train,rng)
        loss=(marginal+(.5*increment if temporal else 0))/10+.05*regularizer
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%100==0:
            score,_=evaluate(model,ds['inner'],train,q,a['split_inner'],16);criterion=score['crps']+.5*score['increment_crps']
            trace.append(dict(step=step,inner=score,marginal=float(marginal.detach()),increment=float(increment.detach()),v_loss=float(regularizer.detach())))
            if criterion<best:best=criterion;chosen=copy.deepcopy(model.state_dict());selected=step
            print(json.dumps(dict(stage='observation',rank=rank,seed=seed,temporal=temporal,step=step,best=best)),flush=True)
    arm='temporal' if temporal else 'marginal';torch.save(dict(model=chosen,selected_step=selected),ROOT/f'r{rank}_{arm}_{seed}.pt')
    return dict(trace=trace,selected_step=selected,seconds=time.monotonic()-start_clock)


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.monotonic()
    a=dict(np.load(d.old.exp.ROOT/'data.npz'));anchors,grid,query=geometry(a)
    source=json.loads((d.ROOT/'experiment.lock.json').read_text())
    assert sha256(d.old.exp.ROOT/'data.npz')==source['data_sha256'];assert sha256(d.__file__)==source['code_sha256']
    np.savez(ROOT/'geometry.npz',anchors=anchors,grid_basis=grid,station_basis=query)
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
      parent_lock_sha256=sha256(d.ROOT/'experiment.lock.json'),diffusion_source_sha256=sha256(d.__file__),data_sha256=source['data_sha256'],geometry_sha256=sha256(ROOT/'geometry.npz'),C_D_labels_read=False))
    selection={};diagnostics={}
    for seed in (42,17):
        base={name:d.prepare(a,name,seed,anchors,query[:,:24]) for name in ('train','inner')}
        for rank in (24,64):
            q=torch.tensor(query[:,:rank].copy(),device='cuda');ds=dict(base);ds['train']={**base['train'],**targets(base['train'],a,query[:,:rank])}
            torch.save({k:ds['train'][k] for k in ('zm','zs')},ROOT/f'norm_r{rank}_{seed}.pt')
            diagnostics[f'r{rank}_{seed}']=ds['train']['diagnostic']
            state,selection[f'r{rank}_pre_{seed}']=pretrain(ds,a,q,seed,rank)
            for temporal in (False,True):selection[f'r{rank}_'+('temporal' if temporal else 'marginal')+f'_{seed}']=finetune(state,ds,a,q,seed,rank,temporal)
            del ds
        del base;torch.cuda.empty_cache()
    write_json(ROOT/'selection.lock.json',selection);d.masked.add_reserved(a);results={}
    for seed in (42,17):
        for name in ('confirmation','reserved'):
            data=d.prepare(a,name,seed,anchors,query[:,:24]);ids=a['split_outer'];ref=np.load(d.old.inv.ROOT/f'history_{seed}_{name}.npz')
            assert np.array_equal(ref['mask'],data['mask'][:,:,ids].cpu().numpy())
            for rank in (24,64):
                q=torch.tensor(query[:,:rank].copy(),device='cuda');norm=torch.load(ROOT/f'norm_r{rank}_{seed}.pt',weights_only=False)
                for arm in ('pre','marginal','temporal'):
                    key=f'r{rank}_{arm}_{seed}';model=Model(rank).cuda();model.load_state_dict(torch.load(ROOT/f'{key}.pt',weights_only=False)['model'])
                    score,p=evaluate(model,data,norm,q,ids);check,_=evaluate(model,data,norm,q,ids,128,11703)
                    results[key+'_'+name]=dict(k64=score,k128=check)
                    np.savez_compressed(ROOT/f'{key}_{name}.npz',**p,station_ids=ids,hours=ref['hours'])
                    print(json.dumps(dict(stage='evaluate',key=key+'_'+name,k128=check)),flush=True)
    write_json(ROOT/'results.json',dict(results=results,selection=selection,diagnostics=diagnostics,runtime=dict(seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),code_sha256=sha256(__file__),
       selection_sha256=sha256(ROOT/'selection.lock.json'),checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')})))


if __name__=='__main__':main()
