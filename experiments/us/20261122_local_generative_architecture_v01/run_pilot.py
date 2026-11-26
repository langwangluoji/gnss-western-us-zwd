"""Finite local single-GPU architecture screen with observation-only supervision."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,copy,math,time,json,random
import numpy as np
import torch
from pilot_models import make_generator,PairCritic,SetDenoiser
from pilot_data import ROOT,PROJECT,Data,monthly_subset,schedule
from gnss_field.models.grid_generation import ObservationCritic,critic_gradient_penalty
from gnss_field.models.diffusion import keyed_noise,heun_sample,edm_coefficients
from gnss_field.models.grid_processor import query_field
from gnss_field.objectives.scores import energy_score
from gnss_field.artifacts import write_json,read_json,immutable_json,sha256
from gnss_field.training.engine import atomic_torch,capture_rng,restore_rng
from grid_generation_metrics import score_ensemble

ARMS={'G0':('gan','plain'),'G-pair':('gan','pair'),'G-noise':('gan','noise'),
      'D0':('diffusion','plain'),'D-attn':('diffusion','attention'),'D-MP':('diffusion','mp')}


def status(**kw):
    write_json(ROOT/'status.json',dict(updated=time.time(),**kw));print('STATUS',kw,flush=True)


def log(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def check_pause():
    if (ROOT/'PAUSE').exists():raise RuntimeError('User pause marker present; checkpoints retained')


def valid_mask(data,ix,fold,role):return data.valid[ix]&data.role(fold,role)[None]


def reuse_result(directory,metadata):
    if not (directory/'result.json').exists():return None
    if read_json(directory/'lock.json')!=metadata:raise ValueError('Completed arm contract differs from current request')
    result=read_json(directory/'result.json')
    if sha256(directory/result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('Completed checkpoint checksum differs')
    return result


@torch.no_grad()
def evaluate(data,model,hours,fold,members,seed,role='inner',set_model=False,scaled=True):
    samples=[];means=[];deltas=[]
    for hour in hours:
        check_pause();h=np.array([hour]);ix,x,base,amp=data.batch(h)
        if set_model:
            condition=data.conditions(x).expand(members,-1,-1)
            mask=torch.ones(members,len(data.a['station_ids']),device=data.device,dtype=torch.bool)
            initial=keyed_noise(h,members,seed,shape=(len(data.a['station_ids']),),device=data.device).flatten(0,1)
            z=heun_sample(lambda state,sigma:model.denoise(state,sigma,condition,data.coords,mask),initial,steps=18)
            s=base[:,None]+10*amp[:,None,None]*z[None]
            mean=s.mean(1);delta=s-base[:,None]
        else:
            row=data.sample(model,h,fold,members,seed,chunk=4,scaled=scaled)
            s=row['samples'];delta=row['delta'];mean=row['base']+delta.mean(1)
        samples.append(s.cpu().numpy());means.append(mean.cpu().numpy());deltas.append(delta.cpu().numpy())
    z=np.concatenate(samples);mean=np.concatenate(means);delta=np.concatenate(deltas);ix=data.indices(hours)
    population=data.a['valid'][ix]&np.array(data.lock['splits'][fold][role])[None]
    metrics=score_ensemble(data.a['truth'][ix],z,mean,population,data.a['xy'])
    metrics.update(hours=len(hours),mean_mc_se_mm=float(np.mean((delta.std(1,ddof=1)/np.sqrt(members))[population])),
        correction_mean_rms_mm=float(np.sqrt(np.mean(delta.mean(1)[population]**2))),
        point_mean_estimator='full Monte Carlo observation mean' if set_model else 'analytic Gaussian mean + MC correction control variate',
        cohort='B2018 incremental spatial x weather development; twelve calendar-selected hours',role=role)
    return metrics,dict(hours=np.array(hours),station_ids=data.a['station_ids'],truth=data.a['truth'][ix],mean=mean,
        samples=z,population=population,xy=data.a['xy'],correction_mean=delta.mean(1))


def save_state(path,model,opt,critic,copt,step,history,candidates,metadata):
    atomic_torch(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),critic=None if critic is None else critic.state_dict(),
        critic_optimizer=None if copt is None else copt.state_dict(),step=step,history=history,candidates=candidates,
        rng=capture_rng(),metadata=metadata))


def gradient_relation(es,adv,model):
    parameters=list(model.parameters());a=torch.autograd.grad(es,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(adv,parameters,retain_graph=True,allow_unused=True)
    aa=sum(float(g.detach().square().sum()) for g in a if g is not None)
    bb=sum(float(g.detach().square().sum()) for g in b if g is not None)
    ab=sum(float((x.detach()*y.detach()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return dict(ES_gradient_norm=math.sqrt(aa),adversarial_gradient_norm=math.sqrt(bb),gradient_cosine=ab/max(math.sqrt(aa*bb),1e-20))


def train_grid(data,arm,fold,steps):
    directory=ROOT/f'fold_{fold}'/arm;directory.mkdir(parents=True,exist_ok=True)
    kind,variant=ARMS[arm];torch.manual_seed(42)
    model=make_generator(kind,'plain' if variant=='pair' else variant,recompute=True).to(data.device)
    torch.manual_seed(142);critic=(PairCritic() if variant=='pair' else ObservationCritic()).to(data.device) if kind=='gan' else None
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    copt=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.9)) if critic is not None else None
    split=data.lock['splits'][fold];batches=schedule(split['train_hours'],steps)
    inner=monthly_subset(split['inner_hours']);outer=monthly_subset(data.lock['outer_hours'])
    metadata=dict(arm=arm,fold=fold,steps=steps,code=sha256(__file__),models=sha256(Path(__file__).with_name('pilot_models.py')),
        data=sha256(ROOT/'data.lock.json'),schedule=batches.tolist(),batch=2,K=4,seed=42)
    existing=reuse_result(directory,metadata)
    if existing is not None:return existing
    immutable_json(directory/'lock.json',metadata);history=[];candidates=[];start=0;elapsed=0.
    if (directory/'latest.pt').exists():
        state=torch.load(directory/'latest.pt',map_location=data.device,weights_only=False)
        if state['metadata']!=metadata:raise ValueError('Resume contract changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];history=state['history'];candidates=state['candidates']
        if critic is not None:critic.load_state_dict(state['critic']);copt.load_state_dict(state['critic_optimizer'])
        restore_rng(state['rng']);elapsed=history[-1]['elapsed_seconds'] if history else 0.
    else:
        metrics,_=evaluate(data,model,inner,fold,16,20261126)
        atomic_torch(directory/'candidate_000.pt',model.state_dict());candidates=[dict(step=0,metrics=metrics,checkpoint='candidate_000.pt')]
        write_json(directory/'inner_000.json',metrics)
    started=time.monotonic()
    for step in range(start+1,steps+1):
        check_pause();status(stage='GRID_TRAIN',arm=arm,fold=fold,step=step-1,max_steps=steps)
        hours=batches[step-1];lr=1e-4*(.1+.9*(1+math.cos(math.pi*(step-1)/steps))/2)
        for group in opt.param_groups:group['lr']=lr
        critic_value=0.
        if critic is not None:
            copt.zero_grad(set_to_none=True)
            for hour in hours:
                h=np.array([hour]);ix,x,base,_=data.batch(h);mask=valid_mask(data,ix,fold,'fit');condition=data.conditions(x)
                with torch.no_grad():fake=(data.sample(model,h,fold,1,500000+step)['samples'][:,0]-base)/10
                real=torch.where(mask,(data.truth[ix]-base)/10,0.)
                torch.manual_seed(700000+step)
                closs=critic(fake,data.coords,condition,mask).mean()-critic(real,data.coords,condition,mask).mean()+10*critic_gradient_penalty(critic,real,fake,data.coords,condition,mask)
                (closs/len(hours)).backward();critic_value+=float(closs.detach())/len(hours)
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);copt.step()
            for p in critic.parameters():p.requires_grad_(False)
        opt.zero_grad(set_to_none=True);totals=dict(loss=0.,ES_mm=0.,adversarial=0.);diagnostics={}
        for j,hour in enumerate(hours):
            h=np.array([hour]);row=data.sample(model,h,fold,4,800000+step)
            ix=row['indices'];mask=valid_mask(data,ix,fold,'fit');es=energy_score(row['samples'],data.truth[ix],mask)
            adv=es.new_zeros(())
            if critic is not None:
                _,x,_,_=data.batch(h);condition=data.conditions(x);z=(row['samples']-row['base'][:,None])/10
                adv=-critic(z.flatten(0,1),data.coords,condition.expand(4,-1,-1),mask.expand(4,-1)).mean()
                if step in (20,steps) and j==0:diagnostics=gradient_relation(es,.1*adv,model)
            loss=es+.1*adv
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            (loss/len(hours)).backward()
            for k,v in [('loss',loss),('ES_mm',es),('adversarial',adv)]:totals[k]+=float(v.detach())/len(hours)
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        if critic is not None:
            for p in critic.parameters():p.requires_grad_(True)
        record=dict(step=step,**totals,critic_loss=critic_value,gradient_norm=gn,lr=lr,**diagnostics,
            elapsed_seconds=elapsed+time.monotonic()-started,supervised_station_hours=int(sum(valid_mask(data,data.indices([h]),fold,'fit').sum() for h in hours)))
        history.append(record);log(directory/'training.jsonl',record)
        if step in (20,24,steps):
            torch.cuda.synchronize()
            if step in (24,steps):
                metrics,_=evaluate(data,model,inner,fold,16,20261126);path=f'candidate_{step:03d}.pt'
                atomic_torch(directory/path,model.state_dict());candidates.append(dict(step=step,metrics=metrics,checkpoint=path))
                write_json(directory/f'inner_{step:03d}.json',metrics)
            save_state(directory/'latest.pt',model,opt,critic,copt,step,history,candidates,metadata)
        if step%8==0:print('UPDATE',fold,arm,step,totals,round(record['elapsed_seconds'],1),flush=True)
    selected=min(candidates,key=lambda r:(r['metrics']['group_ES_mm'],r['step']))
    model.load_state_dict(torch.load(directory/selected['checkpoint'],map_location=data.device,weights_only=True))
    status(stage='GRID_OUTER',arm=arm,fold=fold,selected=selected['step'])
    metrics,arrays=evaluate(data,model,outer,fold,32,20261127,role='outer')
    np.savez_compressed(directory/'outer_predictions.npz',**arrays)
    with torch.no_grad():
        h=outer[:1];row=data.sample(model,h,fold,8,20261127,fields=True)
        amp=torch.tensor(data.a['amplitude'][data.indices(h)],device=data.device)
        fields=data.mean_grid(h)[:,None]+amp[:,None,None,None,None]*row['fields']
        q=query_field(fields.flatten(0,1),data.qi,data.qw).reshape(1,8,-1)
        error=float((q+row['observation_noise']-row['samples']).abs().max())
        if error>.001:raise ValueError(f'Published requery differs: {error}')
        np.savez_compressed(directory/'field_case.npz',hour=h,mean_Z0=fields.mean(1).cpu().numpy(),std_Z0=fields.std(1).cpu().numpy(),members_Z0=fields[:,:4].cpu().numpy())
    result=dict(arm=arm,fold=fold,steps=steps,selected_step=selected['step'],inner=selected['metrics'],outer=metrics,
        checkpoint=selected['checkpoint'],checkpoint_sha256=sha256(directory/selected['checkpoint']),
        publication_requery_max_mm=error,seconds=history[-1]['elapsed_seconds'],genuinely_updated=history[-1]['gradient_norm']>0,
        scope='short mechanism screen; not convergence or final architecture ranking')
    write_json(directory/'result.json',result);del model,opt,critic,copt;torch.cuda.empty_cache();return result


def train_set(data,attention,fold,steps=480):
    arm='N-set' if attention else 'N-point';directory=ROOT/f'fold_{fold}'/arm;directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);model=SetDenoiser(attention=attention).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    split=data.lock['splits'][fold];batches=schedule(split['train_hours'],steps,batch=8)
    inner=monthly_subset(split['inner_hours']);outer=monthly_subset(data.lock['outer_hours']);history=[];candidates=[];start=0
    metadata=dict(arm=arm,fold=fold,steps=steps,data=sha256(ROOT/'data.lock.json'),model=sha256(Path(__file__).with_name('pilot_models.py')),
        runner=sha256(__file__),objective='observed residual EDM-DSM; no additional tau at inference')
    existing=reuse_result(directory,metadata)
    if existing is not None:return existing
    immutable_json(directory/'lock.json',metadata)
    if (directory/'latest.pt').exists():
        state=torch.load(directory/'latest.pt',map_location=data.device,weights_only=False)
        if state['metadata']!=metadata:raise ValueError('Set resume contract changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];history=state['history'];candidates=state['candidates'];restore_rng(state['rng'])
    started=time.monotonic();offset=history[-1]['elapsed_seconds'] if history else 0.
    for step in range(start+1,steps+1):
        check_pause();hours=batches[step-1];ix,x,base,amp=data.batch(hours);condition=data.conditions(x)
        mask=valid_mask(data,ix,fold,'fit');clean=torch.where(mask,(data.truth[ix]-base)/(10*amp[:,None]),0.)
        torch.manual_seed(900000+step);sigma=(torch.randn(len(hours),device=data.device)*1.2-1.2).exp()
        noisy=torch.where(mask,clean+torch.randn_like(clean)*sigma[:,None],0.)
        prediction=model.denoise(noisy,sigma,condition,data.coords,mask)
        weight=edm_coefficients(sigma,1.)[-1]
        loss=(((prediction-clean).square()*mask).sum(1)/mask.sum(1)*weight).mean()
        opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        lr=1e-4*(.1+.9*(1+math.cos(math.pi*(step-1)/steps))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.step();record=dict(step=step,DSM=float(loss.detach()),gradient_norm=gn,lr=lr,elapsed_seconds=offset+time.monotonic()-started,
            supervised_station_hours=int(mask.sum()));history.append(record);log(directory/'training.jsonl',record)
        if step%80==0 or step==20:
            status(stage='SET_TRAIN',arm=arm,fold=fold,step=step,max_steps=steps);print('DSM',fold,arm,record,flush=True)
        if step%160==0 or step==steps:
            metrics,_=evaluate(data,model,inner,fold,16,20261126,set_model=True)
            path=f'candidate_{step:03d}.pt';atomic_torch(directory/path,model.state_dict());candidates.append(dict(step=step,metrics=metrics,checkpoint=path))
            write_json(directory/f'inner_{step:03d}.json',metrics)
            save_state(directory/'latest.pt',model,opt,None,None,step,history,candidates,metadata)
    selected=min(candidates,key=lambda r:(r['metrics']['group_ES_mm'],r['step']))
    model.load_state_dict(torch.load(directory/selected['checkpoint'],map_location=data.device,weights_only=True))
    metrics,arrays=evaluate(data,model,outer,fold,32,20261127,role='outer',set_model=True)
    np.savez_compressed(directory/'outer_predictions.npz',**arrays)
    result=dict(arm=arm,fold=fold,steps=steps,selected_step=selected['step'],inner=selected['metrics'],outer=metrics,
        seconds=history[-1]['elapsed_seconds'],checkpoint=selected['checkpoint'],checkpoint_sha256=sha256(directory/selected['checkpoint']),
        scope='NDP-inspired finite observation set; not original paper reproduction; no continuous-field product')
    write_json(directory/'result.json',result);del model,opt;torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','grid','set','controls'],default='all');p.add_argument('--steps',type=int,default=48);p.add_argument('--set-steps',type=int,default=480);args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.manual_seed(42);np.random.seed(42);random.seed(42)
    immutable_json(ROOT/'training.lock.json',dict(grid_steps=args.steps,set_steps=args.set_steps,batch_hours=2,micro_hours=1,members_train=4,
        widths=[32,64,128,128],Heun=18,seed=42,raw_primary=True,inner_K=16,outer_K=32,inner_hours=12,outer_hours=12,
        learning_rate=1e-4,min_lr_fraction=.1,lr_schedule='cosine, no warmup, local short-screen only',GAN_adversarial=.1,GP=10.,
        benchmark=sha256(ROOT/'benchmark.json'),plan=sha256(ROOT/'PLAN.md'),no_server=True))
    data=Data();summary={}
    for fold in (0,1):
        outer=monthly_subset(data.lock['outer_hours'])
        for scaled,name in ((False,'Gaussian-original'),(True,'Gaussian-calibrated')):
            directory=ROOT/f'fold_{fold}'/name;directory.mkdir(parents=True,exist_ok=True)
            if not (directory/'result.json').exists():
                metrics,arrays=evaluate(data,None,outer,fold,32,20261127,role='outer',scaled=scaled)
                np.savez_compressed(directory/'outer_predictions.npz',**arrays);write_json(directory/'result.json',dict(arm=name,fold=fold,outer=metrics,scale=data.scales[fold] if scaled else 1.))
        if args.mode in ('all','grid'):
            for arm in ARMS:summary[f'{fold}/{arm}']=train_grid(data,arm,fold,args.steps)
        if args.mode in ('all','set'):
            for attention in (False,True):summary[f'{fold}/N-{attention}']=train_set(data,attention,fold,args.set_steps)
    write_json(ROOT/'summary.json',summary);status(stage='COMPLETE',models=len(summary),server_jobs=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='FAILED',error=str(exc));raise
