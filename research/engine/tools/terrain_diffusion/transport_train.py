"""Paired v11 training: matched protocol, no target-history shortcut."""
import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .data import Dataset,PROJECT,sha,write_json
from .model import v_target
from .metrics import evaluate
from .train import seed_all,checkpoint_receipt
from .transport import TransportModel,transfer_conditioner

ROOT=PROJECT/'results/20261123_observation_transport_v11'


@torch.no_grad()
def validation(model,data,arm,rows=(0,3,6,9),hours=4,members=16,seed=90211,progress=False):
    model.eval();members=1 if arm=='deterministic' else members
    gen=torch.Generator(device=data.device).manual_seed(seed);records=[];predictions=[];truths=[]
    for li,c in enumerate(data.layouts()):
        ps=[];ys=[]
        for row in rows:
            for t in range(12,12+hours):
                d=data.batch(row,t,c,np.arange(137),'validation')
                ys.append(np.where(d['mask'].cpu()[0],d['target'].cpu()[0]*data.std[0]+data.mean[0],np.nan))
                d={k:v.expand(members,*v.shape[1:]) for k,v in d.items()}
                if arm=='diffusion':z=model.sample(d,torch.randn(d['target'].shape,device=data.device,generator=gen))
                else:
                    mean,scale=model.baseline(d);z=mean if arm=='deterministic' else mean+scale*torch.randn(mean.shape,device=data.device,generator=gen)
                ps.append((z*data.std[0]+data.mean[0]).cpu().numpy())
            if progress:print(json.dumps(dict(stage='evaluation_progress',layout=li,month_index=row)),flush=True)
        pred=np.stack(ps,1);truth=np.stack(ys);result=evaluate(pred,truth)
        # Each row is a separate sequence, no cross-month increments.
        ds=np.concatenate([np.diff(pred[:,i:i+hours],axis=1) for i in range(0,len(ys),hours)],1)
        dy=np.concatenate([np.diff(truth[i:i+hours],axis=0) for i in range(0,len(ys),hours)])
        result['increment']=evaluate(ds,dy) if hours>1 else {}
        records.append(dict(layout=li,**result));predictions.append(pred);truths.append(truth)
    return dict(layouts=records,macro_rmse_mm=float(np.mean([r['station_macro_rmse_mm'] for r in records])),
                crps_mm=float(np.mean([r['fair_crps_mm'] for r in records]))),np.stack(predictions),np.stack(truths)


def fixed_batches(data):
    c=data.layouts()[0];eligible=np.flatnonzero(~np.isin(data.a['groups'],data.a['groups'][c]))
    q=np.sort(np.random.default_rng(771).choice(eligible,137,False))
    return [data.batch(row,18,c,q) for row in (0,6,24,48)]


@torch.no_grad()
def fixed_score(model,batches,data,arm):
    model.eval();pred=[];truth=[];gen=torch.Generator(device=data.device).manual_seed(8710)
    for base in batches:
        truth.append(np.where(base['mask'].cpu()[0],base['target'].cpu()[0],np.nan))
        d={k:v.expand(16 if arm!='deterministic' else 1,*v.shape[1:]) for k,v in base.items()}
        if arm=='diffusion':p=model.sample(d,torch.randn(d['target'].shape,device=data.device,generator=gen))
        else:p=model.baseline(d)[0]
        pred.append(p.cpu().numpy())
    return evaluate(np.stack(pred,1)*data.std[0]+data.mean[0],np.stack(truth)*data.std[0]+data.mean[0])['station_macro_rmse_mm']


def run(a):
    torch.set_num_threads(4);seed_all(a.seed);data=Dataset();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=TransportModel(a.variant,data.mean[0],data.std[0]).cuda()
    if a.encoder:transfer_conditioner(model,torch.load(a.encoder,map_location='cuda',weights_only=False)['model'])
    ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    rng=np.random.default_rng(a.seed);fixed=fixed_batches(data);count_c=np.zeros(410,dtype=int);count_q=np.zeros(410,dtype=int)
    manifest=dict(**vars(a),data_sha256=data.contract['data_sha256'],encoder_receipt=checkpoint_receipt(a.encoder),
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},history='missing during training and inference',
        parameters=sum(p.numel() for p in model.parameters()),query_counts=[64,96,137,160],context_count=82,ema_decay=.99,
        optimizer='AdamW2e-4 warmup100 cosine floor2e-5 weight_decay1e-4 clip1',
        selection='2018 Jan/Apr/Jul/Oct day1 first4h 3layouts; RMSE deterministic / CRPS probabilistic',
        val_members=1 if a.arm=='deterministic' else 16,normalisation_mean=data.mean.tolist(),normalisation_std=data.std.tolist())
    write_json(out/'config.json',manifest);best=float('inf');start=time.perf_counter()
    fields=['step','loss','train_rmse_mm','gradient_norm','clipped','lr','fixed_train_rmse_mm','validation_rmse_mm','validation_crps_mm','height_scale_km','elapsed_seconds']
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for step in range(1,a.steps+1):
            model.train();row=int(rng.integers(72));t=int(rng.integers(12,36));c=rng.choice(410,82,False)
            available=np.flatnonzero(~np.isin(data.a['groups'],data.a['groups'][c])&np.isfinite(data.a['train'][row,t,:,0]))
            nq=(64,96,137,160)[(step-1)%4];q=rng.choice(available,min(nq,len(available)),False)
            count_c[c]+=1;count_q[q]+=1;d=data.batch(row,t,c,q)
            lr=2e-4*min(step/100,1)*(.1+.9*.5*(1+math.cos(math.pi*max(step-100,0)/max(a.steps-100,1))))
            opt.param_groups[0]['lr']=lr;opt.zero_grad(set_to_none=True)
            if a.arm=='diffusion':
                s=torch.randint(1,101,(1,),device='cuda');ab=model.ab[s][:,None];noise=torch.randn_like(d['target'])
                x=ab.sqrt()*d['target']+(1-ab).sqrt()*noise;v=model(x,s,d)
                loss=(v-v_target(d['target'],noise,ab)).square().mean();clean=ab.sqrt()*x-(1-ab).sqrt()*v
            else:
                clean,scale=model.baseline(d);err=clean-d['target']
                loss=(.5*(err/scale).square()+scale.log()).mean() if a.arm=='gaussian' else err.square().mean()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite optimisation state')
            opt.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.01)
            rec=dict(step=step,loss=float(loss.detach()),train_rmse_mm=float((clean-d['target']).detach().square().mean().sqrt()*data.std[0]),
                     gradient_norm=float(norm),clipped=int(norm>1),lr=lr,fixed_train_rmse_mm='',validation_rmse_mm='',validation_crps_mm='',
                     height_scale_km=float(1+5*model.summary.scale_logit.detach().sigmoid()) if a.variant=='transport' else '',elapsed_seconds=time.perf_counter()-start)
            if step%500==0 or step==a.steps:
                rec['fixed_train_rmse_mm']=fixed_score(ema,fixed,data,a.arm)
                scores,pred,truth=validation(ema,data,a.arm);rec['validation_rmse_mm']=scores['macro_rmse_mm'];rec['validation_crps_mm']=scores['crps_mm']
                score=scores['macro_rmse_mm'] if a.arm=='deterministic' else scores['crps_mm']
                payload=dict(model=ema.state_dict(),raw_model=model.state_dict(),optimizer=opt.state_dict(),step=step,config=manifest)
                torch.save(payload,out/f'step_{step}.pt')
                if score<best:
                    best=score;torch.save(payload,out/'best.pt');write_json(out/'metrics.json',dict(selected_step=step,**scores))
                    np.savez_compressed(out/'evaluation.npz',samples=pred,truth=truth)
                print(json.dumps(dict(run=out.name,**rec)),flush=True)
            writer.writerow(rec);f.flush()
    np.savetxt(out/'exposure.csv',np.c_[np.arange(410),count_c,count_q],delimiter=',',fmt='%d',header='station_index,context_count,target_count',comments='')
    write_json(out/'completion.json',dict(status='completed',seconds=time.perf_counter()-start,best_selection_score=best))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','raw','transport'],required=True)
    p.add_argument('--arm',choices=['deterministic','gaussian','diffusion'],default='deterministic');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--out',required=True);p.add_argument('--encoder')
    run(p.parse_args())
