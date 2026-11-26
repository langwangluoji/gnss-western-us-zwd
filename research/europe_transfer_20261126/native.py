"""I1 and actual native directed-lag Transformer transfer at a six-hour cadence."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn

from .data import HERE,CACHE,sha,write_json,neural_cutoff
from .field import FieldData,EuropeBase
from tools.terrain_diffusion.innovation_refinement import innovation_fields
from tools.terrain_diffusion.native_weather import weather_fields
from tools.terrain_diffusion.native_directed_lag_v2 import NativeDirectedLagV2
from tools.terrain_diffusion.native_physics_lm import causal_station_loss


def make_model(nstations=411):
    model=NativeDirectedLagV2('joint','transformer',64,16)
    model.time.identity=nn.Embedding(nstations,8)
    return model


def i1_features(innovation):
    t=np.arange(len(innovation))
    return np.concatenate([innovation[np.maximum(t-lag,0)] for lag in (0,1,3)],-1)/10


def clocks(hours):
    phase=hours*2*np.pi/(365.25*24);day=hours%24*2*np.pi/24
    return np.stack([np.sin(day),np.cos(day),np.sin(phase),np.cos(phase)],-1).astype('f4')


def inputs(cache,a,positions,query,coef):
    # Positions index the versioned per-layout cache, not absolute hourly steps.
    c=cache['context'];p=np.asarray(positions);q=np.asarray(query);s=int(cache['start'])
    ix=p[:,None]-np.arange(15,-1,-1)[None];global_ix=ix+s
    raw=a['values'][global_ix[:,:,None],c[None,None]];valid=np.isfinite(raw)
    norm=np.where(valid,(raw-np.array([0,280,900]))/np.array([100,20,100]),0)
    if 'ages' not in a:
        # Same causal full-timeline construction used during training. A gap
        # at the history boundary retains its actual earlier observation age.
        absolute=np.arange(len(a['values']))[:,None,None]
        last=np.maximum.accumulate(np.where(np.isfinite(a['values']),absolute,-100000),axis=0)
        a['ages']=np.minimum((absolute-last)*6,10000).astype('f4')
    age=a['ages'][global_ix[:,:,None],c[None,None]]
    fx=i1_features(cache['innovation']);correction=fx@coef
    bkg=cache['base']+correction
    old_error=np.where(valid[...,0],cache['self_error'][ix],0)
    new_error=np.where(valid[...,0],old_error-correction[ix[:,:,None],c[None,None]],0)
    clock=clocks(a['hours'][global_ix]);B=len(p);C=len(c)
    history=np.concatenate([norm,valid.astype('f4'),np.log1p(age)/5,old_error[...,None]/10,new_error[...,None]/10,np.broadcast_to(clock[:,:,None],(B,16,C,4))],-1).astype('f4')
    innov=np.concatenate([cache['innovation'][p-lag][:,q] for lag in (0,1,2,4,8,12)],-1)/10
    current=bkg[p][:,q]
    levels=np.stack([current/100,*[(current-bkg[p-lag][:,q])/10 for lag in (1,2,4,8,12)]],-1)
    met=np.concatenate([cache['weather'][p-lag][:,q] for lag in (0,2,8,12)],-1)
    xy=a['xy'];dist=np.sqrt(((xy[q,None]-xy[c][None])**2).sum(-1))
    active=np.isfinite(a['values'][p[:,None]+s,c[None],0])
    nearest=np.where(active[:,None],dist[None],np.inf).min(-1)
    nearest=np.minimum(nearest,3000)
    condition=np.concatenate([innov,levels,met,np.broadcast_to(a['static'][q],(B,len(q),9)),np.broadcast_to(clocks(a['hours'][p+s])[:,None],(B,len(q),4)),np.stack([np.log1p(nearest)/6,(nearest<50).astype('f4')],-1)],-1).astype('f4')
    return dict(history=history,condition=condition,background=current.astype('f4'),nearest=nearest.astype('f4'))


@torch.no_grad()
def cache_segment(d,model,a,start,stop,context,path):
    begin=max(15,start-32);count=stop-begin;Q=len(a['static'])
    allq=np.arange(Q);base=[];innov=[];met=[];error=[]
    c=torch.tensor(context,device='cuda');qxy=d.xy;cheight=d.static[c,2]*3000;qheight=d.static[:,2]*3000
    for j in range(begin,stop,8):
        at=np.arange(j,min(j+8,stop));b=d.batch(at,context,allq);p=model(b)
        raw=d.values[torch.tensor(at,device='cuda')[:,None],c[None]]
        residual=raw[:,:,0]-p[:,c];valid=torch.isfinite(raw[:,:,0])
        iv=innovation_fields(residual,valid,qxy,d.xy[c],qheight,cheight)
        wx=weather_fields(raw,qxy,d.xy[c],qheight,cheight)
        base.append(p.cpu().numpy());innov.append(iv.cpu().numpy());met.append(wx.cpu().numpy());error.append(torch.nan_to_num(residual).cpu().numpy())
    np.savez_compressed(path,base=np.concatenate(base),innovation=np.concatenate(innov),weather=np.concatenate(met),self_error=np.concatenate(error),context=context,start=begin,fit_start=start-begin,stop=stop)


def prepare(seed,version=1):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    a=dict(np.load(CACHE/'data.npz'));d=FieldData(a)
    suffix='_v2' if version==2 else '';basepath=HERE/f'runs/base{suffix}_s{seed}/best.pt'
    model=EuropeBase().cuda();model.load_state_dict(torch.load(basepath,weights_only=True));model.eval();model.carrier.time.temporal.flatten_parameters()
    out=CACHE/f'native{suffix}_s{seed}';out.mkdir(exist_ok=False);train=np.flatnonzero(a['role']=='train');rng=np.random.default_rng(seed)
    devstart=int(np.searchsorted(a['hours'],np.datetime64('2023-01-01','h').astype('i8')));stop=int(np.searchsorted(a['hours'],np.datetime64('2024-01-01','h').astype('i8')));cut=neural_cutoff(a['hours'],version)
    matrix=np.zeros((9,9));rhs=np.zeros(9);total=0;records=[]
    for i,start in enumerate(range(48,cut-6,128)):
        end=min(start+128,cut-6);c=np.sort(rng.choice(train,int(rng.choice([96,160,224])),replace=False));path=out/f'train_{i:02d}.npz'
        cache_segment(d,model,a,start,end,c,path)
        cache=dict(np.load(path));q=train[~np.isin(a['groups'][train],a['groups'][c])];pos=np.arange(int(cache['fit_start']),len(cache['base']))
        x=i1_features(cache['innovation'])[pos][:,q].reshape(-1,9).astype('f8')
        y=(a['values'][pos[:,None]+int(cache['start']),q[None],0]-cache['base'][pos][:,q]).reshape(-1).astype('f8');mask=np.isfinite(y)
        matrix+=x[mask].T@x[mask];rhs+=x[mask].T@y[mask];total+=int(mask.sum())
        records.append(dict(file=path.name,start=start,stop=end,context=len(c),targets=len(q)))
        print(json.dumps(dict(stage='native_cache',seed=seed,segment=i,start=start,stop=end)),flush=True)
    coef=np.linalg.solve(matrix+.01*np.diag(np.maximum(np.diag(matrix),1e-8)),rhs)
    np.save(out/'i1.npy',coef)
    # The entire2023 monitoring year and2024 evaluation year use the same273 references.
    for name,start,end in [('dev',devstart,stop),('test',stop,len(a['hours']))]:
        cache_segment(d,model,a,start,end,train,out/f'{name}.npz')
    write_json(HERE/f'runs/base{suffix}_s{seed}/native_cache.json',dict(version=version,base_sha256=sha(basepath),i1_coefficient=coef.tolist(),i1_fit_records=total,segments=records,history_records=16,lag_hours=[0,6,18,36,72],query_lag_hours=[0,6,12,24,48,72],output_hours=[0,6,12,18,24,30,36],context_fixed_within_continuous_segment=True,training_targets_hide_entire_1km_site_group=True))


def full_test_cache(seed):
    """Add the final six current epochs to v1 coverage without rewriting it."""
    torch.set_num_threads(4);a=dict(np.load(CACHE/'data.npz'));d=FieldData(a)
    model=EuropeBase().cuda();model.load_state_dict(torch.load(HERE/f'runs/base_s{seed}/best.pt',weights_only=True));model.eval();model.carrier.time.temporal.flatten_parameters()
    start=int(np.searchsorted(a['hours'],np.datetime64('2024-01-01','h').astype('i8')))
    path=CACHE/f'native_s{seed}/test_full.npz'
    if path.exists():raise FileExistsError(path)
    cache_segment(d,model,a,start,len(a['hours']),np.flatnonzero(a['role']=='train'),path)
    write_json(HERE/f'runs/base_s{seed}/test_coverage_v2.json',dict(source='test.npz preserved',current_2024_epochs=1464,cache_sha256=sha(path),reason='current output can cover last36h even when future labels lie outside2024'))


def tensor_inputs(cache,a,pos,q,coef):
    b=inputs(cache,a,pos,q,coef)
    return {k:torch.tensor(v,device='cuda') for k,v in b.items()}


def training_blocks(caches):
    # Cache already contains32 preceding states; keep every four-record origin.
    return [(i,j) for i,cache in enumerate(caches) for j in range(int(cache['fit_start']),len(cache['base'])-3,4)]


@torch.no_grad()
def monitor(model,cache,a,coef):
    q=np.flatnonzero(a['role']=='dev');c=cache['context'];cs=torch.tensor(a['static'][c],device='cuda');ids=torch.tensor(c,device='cuda');qs=torch.tensor(a['static'][q],device='cuda')
    origins=np.linspace(int(cache['fit_start'])+16,len(cache['base'])-8,24).astype(int)
    square=np.zeros((len(q),2));count=np.zeros((len(q),2));model.eval()
    for p in origins:
        pos=np.arange(p,p+4);b=tensor_inputs(cache,a,pos,q,coef)
        correction,_=model(b['condition'],b['history'],cs,ids,qs)
        pred=b['background'][:,:,None]+correction[:,:,:2]
        target=a['values'][pos[:,None,None]+int(cache['start'])+np.arange(2)[None,None],q[None,:,None],0]
        error=pred.cpu().numpy()-target;valid=np.isfinite(target)
        square+=np.where(valid,error*error,0).sum(0);count+=valid.sum(0)
    return [float(np.sqrt(square[:,i][count[:,i]>0]/count[:,i][count[:,i]>0]).mean()) for i in range(2)]


def train(seed,steps=3000,smoke=False,version=1):
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    a=dict(np.load(CACHE/'data.npz'));positions=np.arange(len(a['values']))[:,None,None]
    a['ages']=np.minimum((positions-np.maximum.accumulate(np.where(np.isfinite(a['values']),positions,-100000),0))*6,10000).astype('f4')
    suffix='_v2' if version==2 else '';root=CACHE/f'native{suffix}_s{seed}';coef=np.load(root/'i1.npy');dev=dict(np.load(root/'dev.npz'))
    files=sorted(root.glob('train_*.npz'));caches=[dict(np.load(f)) for f in files]
    blocks=training_blocks(caches)
    model=make_model(len(a['static'])).cuda();ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    out=HERE/('smoke' if smoke else 'runs')/f'joint{suffix}_s{seed}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();best=float('inf')
    write_json(out/'lock.json',dict(version=version,seed=seed,steps=steps,base_sha256=sha(HERE/f'runs/base{suffix}_s{seed}/best.pt'),native_code_sha256=sha(__file__),class_name='NativeDirectedLagV2',length_records=16,identity_count=len(a['static']),train_blocks=len(blocks),order='chronological per complete pass; no time sampling',supervised_leads_hours=[0,6,12,18,24,30,36],selection='2023 dev current station macro RMSE; spatial holdout for v2',test_2024_used_for_selection=False))
    for step in range(1,steps+1):
        i,j=blocks[(step-1)%len(blocks)];cache=caches[i];c=cache['context'];pool=np.flatnonzero(a['role']=='train');q=pool[~np.isin(a['groups'][pool],a['groups'][c])]
        pos=np.arange(j,j+4);b=tensor_inputs(cache,a,pos,q,coef)
        cs=torch.tensor(a['static'][c],device='cuda');qs=torch.tensor(a['static'][q],device='cuda');ids=torch.tensor(c,device='cuda')
        target=a['values'][pos[:,None,None]+int(cache['start'])+np.arange(7)[None,None],q[None,:,None],0];target=torch.tensor(target,device='cuda');valid=torch.isfinite(target)
        model.train();opt.zero_grad(set_to_none=True)
        correction,_,aux=model(b['condition'],b['history'],cs,ids,qs,True);pred=b['background'][:,:,None]+correction
        error=(pred-torch.nan_to_num(target))/10
        weights=error.new_tensor([1.,.5,.2,.2,.2,.2,.2])[None,None].expand_as(error)
        loss=(error[valid].square()*weights[valid]).sum()/weights[valid].sum()
        auxloss,_=causal_station_loss(aux,b['history']);loss=loss+.01*auxloss
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('native nonfinite')
        opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.01)
        if step==1 or step%100==0 or step==steps:
            row=dict(step=step,seconds=time.monotonic()-start,loss=float(loss.detach()),current_batch_rmse_mm=float((pred[:,:,0]-torch.nan_to_num(target[:,:,0]))[valid[:,:,0]].square().mean().sqrt().detach()))
            if step%500==0 or step==steps:
                scores=monitor(ema,dev,a,coef);row.update(dev_current_rmse_mm=scores[0],dev_f6_rmse_mm=scores[1])
                if scores[0]<best:best=scores[0];torch.save(ema.state_dict(),out/'best.pt')
            write_json(out/'progress.json',row)
            with (out/'curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(stage=out.name,**row)),flush=True)
    torch.save(model.state_dict(),out/'last.pt');torch.save(ema.state_dict(),out/'ema.pt')
    write_json(out/'complete.json',dict(steps=steps,best_dev_current_rmse_mm=best,seconds=time.monotonic()-start,best_sha256=sha(out/'best.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--prepare',action='store_true');p.add_argument('--full-test-cache',action='store_true');p.add_argument('--steps',type=int,default=3000);p.add_argument('--smoke',action='store_true');p.add_argument('--version',type=int,choices=[1,2],default=1);args=p.parse_args()
    if args.prepare:prepare(args.seed,args.version)
    elif args.full_test_cache:full_test_cache(args.seed)
    else:train(args.seed,args.steps,args.smoke,args.version)
