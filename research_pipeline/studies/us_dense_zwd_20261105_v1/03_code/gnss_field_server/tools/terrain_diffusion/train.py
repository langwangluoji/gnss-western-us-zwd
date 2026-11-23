"""Bounded screening and configurable foundation training. No sealed-set access."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import Dataset,DEFAULT_DATA,write_json,sha
from .model import DiffusionModel,v_target
from .metrics import evaluate


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def history_push(d,value):
    item=torch.stack([value,torch.ones_like(value),torch.ones_like(value)],-1)[:,None]
    d['history']=torch.cat([d['history'][:,1:],item],1)
    return d['history']


def transfer_encoder(model,state):
    selected={k:v for k,v in state.items() if k.startswith(('context.','spatial.','temporal.'))}
    model.load_state_dict(selected,strict=False)


def checkpoint_receipt(path):
    if not path:return None
    source=torch.load(path,map_location='cpu',weights_only=False)
    return dict(path=str(path),sha256=sha(path),selected_step=source.get('step'),config=source.get('config'))


def fair_crps(samples,truth,mask):
    k=len(samples)
    score=(samples-truth).abs().mean(0)-(samples[:,None]-samples[None,:]).abs().sum((0,1))/(2*k*(k-1))
    return score[mask[0]].mean()


def trajectory_loss(model,batches,members=4,steps=20,increment_scale=1.):
    history=None;sequence=[];losses=[];previous=None;previous_mask=None
    for original in batches:
        d={k:v.expand(members,*v.shape[1:]) for k,v in original.items()}
        if history is not None:d['history']=history
        pred=model.sample(d,torch.randn_like(d['target']),steps,checkpoint=True)
        truth=original['target'];mask=original['mask']
        marginal=fair_crps(pred,truth,mask) if mask.any() else pred.sum()*0
        if previous is not None:
            paired=mask&previous_mask
            if paired.any():marginal=marginal+.5*fair_crps(pred-sequence[-1],truth-previous,paired)/increment_scale
        # Auxiliary denoising uses only valid current nodes, with detached generated history.
        if mask.any():
            q=mask[0];aux=dict(original);aux['query']=original['query'][:,q];aux['qgeo']=original['qgeo'][:,q]
            aux['history']=d['history'][:1,:,q].detach();s=torch.randint(1,101,(1,),device=pred.device)
            a=model.ab[s][:,None];noise=torch.randn_like(truth[:,q]);x=a.sqrt()*truth[:,q]+(1-a).sqrt()*noise
            v=model(x,s,aux);marginal=marginal+.05*(v-v_target(truth[:,q],noise,a)).square().mean()
        losses.append(marginal);sequence.append(pred);history=history_push(d,pred)
        previous=truth;previous_mask=mask
    return torch.stack(losses).mean(),sequence


@torch.no_grad()
def validate(model,data,arm,rows=(0,6),hours=4,members=4,steps=20,seed=9001):
    model.eval();gen=torch.Generator(device=data.device).manual_seed(seed)
    records=[];all_samples=[];all_truth=[]
    q=np.arange(len(data.a['val_ids']))
    for li,c in enumerate(data.layouts()):
        seqs=[];truths=[]
        for row in rows:
            history=None;sequence=[];truth=[]
            for t in range(12,12+hours):
                d=data.batch(row,t,c,q,'validation')
                truth.append(np.where(d['mask'].cpu()[0],d['target'].cpu()[0]*data.std[0]+data.mean[0],np.nan))
                d={k:v.expand(members,*v.shape[1:]) for k,v in d.items()}
                if history is not None:d['history']=history
                noise=torch.randn(members,len(q),generator=gen,device=data.device)
                if arm=='diffusion':pred=model.sample(d,noise,steps)
                else:
                    mean,scale=model.baseline(d);pred=mean+noise*scale if arm=='gaussian' else mean
                history=history_push(d,pred)
                sequence.append((pred*data.std[0]+data.mean[0]).cpu().numpy())
            seqs.append(np.stack(sequence,1));truths.append(np.stack(truth))
        sm=np.concatenate(seqs,1);yt=np.concatenate(truths,0)
        result=evaluate(sm,yt)
        # Differences are computed within each continuous sequence, never across months.
        inc=evaluate(np.concatenate([np.diff(s,axis=1) for s in seqs],1),np.concatenate([np.diff(y,axis=0) for y in truths])) if hours>1 else {}
        records.append(dict(layout=li,**result,increment=inc));all_samples.append(sm);all_truth.append(yt)
    return dict(layouts=records,mean_macro_rmse_mm=float(np.mean([r['station_macro_rmse_mm'] for r in records])),
                worst_layout_rmse_mm=max(r['station_macro_rmse_mm'] for r in records)),np.stack(all_samples),np.stack(all_truth)


def run(args):
    torch.set_num_threads(4);seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=Dataset(args.data,args.device);rng=np.random.default_rng(args.seed)
    model=DiffusionModel(args.width,args.blocks).to(args.device)
    if args.encoder:transfer_encoder(model,torch.load(args.encoder,map_location=args.device,weights_only=False)['model'])
    if args.initial:model.load_state_dict(torch.load(args.initial,map_location=args.device,weights_only=False)['model'])
    ema=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    manifest=vars(args).copy();manifest.update(data_sha256=data.contract['data_sha256'],parameters=sum(p.numel() for p in model.parameters()),
        encoder_receipt=checkpoint_receipt(args.encoder),initial_receipt=checkpoint_receipt(args.initial),
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        evaluation='validation2018 January/July day1, 3 layouts, cold-start self-history; bounded screening',
        checkpoint_selection='deterministic: macro RMSE; probabilistic: macro fair CRPS; point tradeoff reported separately',
        device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu')
    write_json(out/'config.json',manifest)
    fields=['step','loss','train_x0_rmse_mm','validation_macro_rmse_mm','lr','elapsed_seconds','peak_memory_mb']
    start=time.perf_counter();best=float('inf');logs=[]
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for step in range(1,args.steps+1):
            model.train();row=int(rng.integers(2 if args.overfit else len(data.a['train'])));t=int(rng.integers(12,33 if args.temporal else 36))
            if args.overfit:t=12
            c=data.layouts()[0] if args.overfit else rng.choice(410,82,replace=False)
            eligible=np.flatnonzero(~np.isin(data.a['groups'],data.a['groups'][c])&np.isfinite(data.a['train'][row,t,:,0]))
            q=eligible[:args.queries] if args.overfit else rng.choice(eligible,min(args.queries,len(eligible)),replace=False)
            d=data.batch(row,t,c,q,teacher=bool(rng.random()<.5));mask=d['mask']
            if not mask.any():continue
            opt.zero_grad(set_to_none=True)
            if args.temporal:
                batches=[data.batch(row,t+j,c,q) for j in range(4)]
                delta=np.diff(data.a['train'][...,0],axis=1)
                ratio=float(np.nanstd(delta)/data.std[0])
                loss,seq=trajectory_loss(model,batches,args.members,args.ddim_steps,ratio)
                clean=seq[0].mean(0,keepdim=True)
            elif args.arm=='diffusion':
                s=torch.randint(1,101,(1,),device=args.device);a=model.ab[s][:,None];noise=torch.randn_like(d['target'])
                x=a.sqrt()*d['target']+(1-a).sqrt()*noise
                v=model(x,s,d);target=v_target(d['target'],noise,a)
                loss=(v[mask]-target[mask]).square().mean()
                clean=a.sqrt()*x-(1-a).sqrt()*v
            else:
                clean,scale=model.baseline(d);err=clean-d['target']
                loss=(.5*(err/scale).square()+scale.log())[mask].mean() if args.arm=='gaussian' else err[mask].square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.05)
            record=dict(step=step,loss=float(loss),train_x0_rmse_mm=float((clean-d['target'])[mask].square().mean().sqrt()*data.std[0]),
                        validation_macro_rmse_mm='',lr=opt.param_groups[0]['lr'],elapsed_seconds=time.perf_counter()-start,
                        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0)
            if step%args.validate_every==0 or step==args.steps:
                metrics,samples,truth=validate(ema,data,args.arm,hours=args.eval_hours,members=args.members,steps=args.ddim_steps)
                record['validation_macro_rmse_mm']=metrics['mean_macro_rmse_mm']
                score=metrics['mean_macro_rmse_mm'] if args.arm=='deterministic' else float(np.mean([r['fair_crps_mm'] for r in metrics['layouts']]))
                if score<best:
                    best=score;torch.save(dict(model=ema.state_dict(),step=step,config=manifest),out/'best.pt')
                    write_json(out/'metrics.json',dict(selected_step=step,**metrics));np.savez_compressed(out/'evaluation.npz',samples=samples,truth=truth)
                print(json.dumps(dict(arm=args.arm,seed=args.seed,**record)),flush=True)
            writer.writerow(record);f.flush();logs.append(record)
    torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=args.steps),out/'last.pt')
    write_json(out/'completion.json',dict(status='completed_bounded_screen',seconds=time.perf_counter()-start,best_selection_score=best))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(DEFAULT_DATA));p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=['deterministic','gaussian','diffusion'],default='diffusion');p.add_argument('--seed',type=int,default=41)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--width',type=int,default=64);p.add_argument('--blocks',type=int,default=4);p.add_argument('--queries',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device',default='cuda');p.add_argument('--eval-hours',type=int,default=4)
    p.add_argument('--members',type=int,default=4);p.add_argument('--ddim-steps',type=int,default=20)
    p.add_argument('--encoder');p.add_argument('--initial');p.add_argument('--temporal',action='store_true');p.add_argument('--overfit',action='store_true')
    run(p.parse_args())
