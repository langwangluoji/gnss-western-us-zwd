"""Fresh matched residual heads on the frozen geometry-corrected mean."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import math,time,json,copy,signal,argparse
from pathlib import Path
import numpy as np
import torch
from retrain_common import ROOT,PROJECT,OLD
from retrain_data import Data
from pilot_data import monthly_subset,schedule
from pilot_models import make_generator,SetDenoiser
from gnss_field.models.grid_generation import ObservationCritic,critic_gradient_penalty
from gnss_field.models.diffusion import heun_sample,keyed_noise,edm_coefficients
from gnss_field.models.grid_processor import query_field
from gnss_field.objectives.scores import energy_score
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.training.engine import atomic_torch,restore_rng
from run_pilot import save_state,reuse_result,gradient_relation
from grid_generation_metrics import score_ensemble

ARMS=('gan','diffusion','ndp_point','ndp_GP_point')
DEADLINE=None


def status(**kw):write_json(ROOT/'status.json',dict(updated=time.time(),**kw));print('STATUS',kw,flush=True)


def check():
    if (ROOT/'PAUSE').exists() or (DEADLINE is not None and time.monotonic()>=DEADLINE):raise RuntimeError('New-base training paused or budget exhausted; saved checkpoint retained')


def mask_for(data,ix,fold,role):return data.valid[ix]&data.role(fold,role)[None]


@torch.no_grad()
def predict(data,model,hours,fold,members,seed,arm):
    data.set_fold(fold)
    if arm in ('gan','diffusion','gaussian'):
        row=data.sample(model,hours,fold,members,seed,chunk=4)
        return row['samples'],row['base']+row['delta'].mean(1),row['delta']
    ix,x,base,amp=data.batch(hours);k=members;b=len(hours)
    condition=data.conditions(x)[:,None].expand(-1,k,-1,-1).flatten(0,1)
    noise=keyed_noise(hours,k,seed,shape=(511,),device=data.device).flatten(0,1)
    if arm=='ndp_GP_point':noise=noise@data.factors[fold].T
    static_mask=torch.ones(b*k,511,dtype=torch.bool,device=data.device)
    z=heun_sample(lambda state,s:model.denoise(state,s,condition,data.coords,static_mask),noise,steps=18).reshape(b,k,511)
    samples=base[:,None]+10*amp[:,None,None]*z
    gain=float(heun_sample(lambda x,s:x/(1+s*s),torch.ones(1,1,dtype=torch.float64),steps=18)[0,0])
    correction=10*amp[:,None,None]*(z-gain*noise.reshape(b,k,511))
    return samples,base+correction.mean(1),correction


@torch.no_grad()
def evaluate(data,model,fold,hours,members,seed,arm,role):
    zs=[];means=[];changes=[]
    for h in hours:
        check();z,mean,change=predict(data,model,np.array([h]),fold,members,seed,arm)
        zs.append(z.cpu().numpy());means.append(mean.cpu().numpy());changes.append(change.cpu().numpy())
    samples=np.concatenate(zs);mean=np.concatenate(means);change=np.concatenate(changes);ix=data.indices(hours)
    population=data.a['valid'][ix]&np.array(data.lock['splits'][fold][role])[None]
    metrics=score_ensemble(data.a['truth'][ix],samples,mean,population,data.a['xy'])
    metrics.update(mean_mc_se_mm=float(np.mean((change.std(1,ddof=1)/np.sqrt(members))[population])),hours=len(hours),
        correction_mean_rms_mm=float(np.sqrt(np.mean(change.mean(1)[population]**2))),mean_estimator='fixed zero-mean noise control variate',
        population='B2018 exposed incremental development; frozen geometry base; no C evaluation of residual heads')
    return metrics,dict(hours=hours,station_ids=data.a['station_ids'],truth=data.a['truth'][ix],population=population,samples=samples,mean=mean,xy=data.a['xy'])


def train(data,arm,fold,engineering=False,stop_after=None,label='continuous'):
    data.set_fold(fold);directory=ROOT/'engineering'/arm/label if engineering else ROOT/f'fold_{fold}'/arm;directory.mkdir(parents=True,exist_ok=True)
    is_ndp=arm.startswith('ndp');steps=4 if engineering else 1440 if is_ndp else 72
    cfg=dict(arm=arm,fold=fold,steps=steps,batch=8,micro=8 if is_ndp else 1,seed=42,
        mean=sha256(ROOT/'mean.lock.json'),probability=sha256(ROOT/'probability.lock.json'),runner=sha256(__file__),
        model=sha256(OLD/'pilot_models.py'),new_data_code=sha256(ROOT/'retrain_data.py'),condition='unchanged compact67+XY',engineering=engineering)
    old=reuse_result(directory,cfg)
    if old is not None:return old
    immutable_json(directory/'lock.json',cfg);torch.manual_seed(42)
    model=(SetDenoiser(attention=False) if is_ndp else make_generator(arm,recompute=True)).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    torch.manual_seed(142);critic=ObservationCritic().to(data.device) if arm=='gan' else None
    copt=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(0.,.9)) if critic is not None else None
    split=data.lock['splits'][fold];batches=schedule(split['train_hours'],steps,batch=8);inner=monthly_subset(split['inner_hours']);outer=monthly_subset(data.lock['outer_hours'])
    history=[];candidates=[];start=0;offset=0.
    if (directory/'latest.pt').exists():
        state=torch.load(directory/'latest.pt',map_location=data.device,weights_only=False)
        if state['metadata']!=cfg:raise ValueError('Retraining checkpoint contract mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];history=state['history'];candidates=state['candidates'];restore_rng(state['rng'])
        if critic is not None:critic.load_state_dict(state['critic']);copt.load_state_dict(state['critic_optimizer'])
        if history:offset=history[-1]['elapsed_seconds']
    def due_at(step):return not engineering and (step>0 and step%480==0 if is_ndp else step in (0,36,72))
    def candidate(step):
        try:metrics,_=evaluate(data,model,fold,inner,32,20261130,arm,'inner')
        except BaseException:
            save_state(directory/'latest.pt',model,opt,critic,copt,step,history,candidates,cfg);raise
        filename=f'candidate_{step:04d}.pt';atomic_torch(directory/filename,model.state_dict())
        candidates.append(dict(step=step,metrics=metrics,checkpoint=filename));write_json(directory/f'inner_{step:04d}.json',metrics)
    if due_at(start) and not any(c['step']==start for c in candidates):candidate(start)
    started=time.monotonic()
    end=min(steps,stop_after) if stop_after is not None else steps
    for step in range(start+1,end+1):
        try:check()
        except BaseException:
            save_state(directory/'latest.pt',model,opt,critic,copt,step-1,history,candidates,cfg);raise
        hours=batches[step-1];lr=1e-4*(.1+.9*(1+math.cos(math.pi*(step-1)/steps))/2)
        for group in opt.param_groups:group['lr']=lr
        closs_value=0.;adv_value=0.
        if is_ndp:
            ix,x,base,amp=data.batch(hours);condition=data.conditions(x);mask=mask_for(data,ix,fold,'fit')
            clean=torch.where(mask,(data.truth[ix]-base)/(10*amp[:,None]),0.)
            torch.manual_seed(900000+step);sigma=(torch.randn(len(hours),device=data.device)*1.2-1.2).exp();noise=torch.randn_like(clean)
            if arm=='ndp_GP_point':noise=noise@data.factors[fold].T
            noisy=torch.where(mask,clean+sigma[:,None]*noise,0.)
            denoised=model.denoise(noisy,sigma,condition,data.coords,mask)
            loss=(((denoised-clean).square()*mask).sum(1)/mask.sum(1)*edm_coefficients(sigma,1.)[-1]).mean();es_value=None
            opt.zero_grad(set_to_none=True);loss.backward();loss_value=float(loss.detach())
        else:
            if critic is not None:
                copt.zero_grad(set_to_none=True)
                for hour in hours:
                    h=np.array([hour]);ix,x,base,_=data.batch(h);mask=mask_for(data,ix,fold,'fit');conditions=data.conditions(x)
                    with torch.no_grad():fake=(data.sample(model,h,fold,1,500000+step)['samples'][:,0]-base)/10
                    real=torch.where(mask,(data.truth[ix]-base)/10,0.);torch.manual_seed(700000+step)
                    closs=critic(fake,data.coords,conditions,mask).mean()-critic(real,data.coords,conditions,mask).mean()+10*critic_gradient_penalty(critic,real,fake,data.coords,conditions,mask)
                    (closs/8).backward();closs_value+=float(closs.detach())/8
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);copt.step()
                for p in critic.parameters():p.requires_grad_(False)
            opt.zero_grad(set_to_none=True);loss_value=es_value=0.
            for hour in hours:
                h=np.array([hour]);row=data.sample(model,h,fold,4,800000+step);ix=row['indices'];mask=mask_for(data,ix,fold,'fit')
                es=energy_score(row['samples'],data.truth[ix],mask);adv=es.new_zeros(())
                if critic is not None:
                    _,x,_,_=data.batch(h);z=(row['samples']-row['base'][:,None])/10
                    adv=-critic(z.flatten(0,1),data.coords,data.conditions(x).expand(4,-1,-1),mask.expand(4,-1)).mean()
                loss=es+.1*adv;(loss/8).backward();loss_value+=float(loss.detach())/8;es_value+=float(es.detach())/8;adv_value+=float(adv.detach())/8
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        if critic is not None:
            for p in critic.parameters():p.requires_grad_(True)
        row=dict(step=step,loss=loss_value,ES_mm=es_value,adversarial=adv_value,critic_loss=closs_value,gradient_norm=gn,lr=lr,
            elapsed_seconds=offset+time.monotonic()-started,supervised_station_hours=int(mask_for(data,data.indices(hours),fold,'fit').sum()))
        if not np.isfinite(loss_value):raise ValueError('Nonfinite new-base training loss')
        history.append(row)
        with (directory/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if step%8==0 or step==20:status(stage='RETRAIN',arm=arm,fold=fold,step=step,max_steps=steps,loss=loss_value)
        due=due_at(step)
        if due:candidate(step)
        if due or step==20 or step==end or (ROOT/'PAUSE').exists():save_state(directory/'latest.pt',model,opt,critic,copt,step,history,candidates,cfg)
    if engineering:return dict(directory=str(directory),step=end)
    selections={'point':min(candidates,key=lambda r:(r['metrics']['station_macro_rmse_mm'],r['step'])),
        'joint':min(candidates,key=lambda r:(r['metrics']['group_ES_mm'],r['step']))}
    products={};evaluated={}
    for name,selected in selections.items():
        model.load_state_dict(torch.load(directory/selected['checkpoint'],map_location=data.device,weights_only=True));status(stage='OUTER',fold=fold,arm=arm,product=name)
        if selected['step'] not in evaluated:evaluated[selected['step']]=evaluate(data,model,fold,outer,128,20261131,arm,'outer')
        metrics,arrays=evaluated[selected['step']];np.savez_compressed(directory/f'{name}_predictions.npz',**arrays)
        products[name]=dict(selected=selected,metrics=metrics,checkpoint_sha256=sha256(directory/selected['checkpoint']))
    result=dict(arm=arm,fold=fold,steps=steps,products=products,checkpoint=selections['joint']['checkpoint'],checkpoint_sha256=products['joint']['checkpoint_sha256'],
        seconds=history[-1]['elapsed_seconds'],new_mean=True)
    write_json(directory/'result.json',result);del model,opt,critic,copt;torch.cuda.empty_cache();return result


def engineering_check(data,tag):
    from gnss_field.training.observable import _same
    report=ROOT/f'{tag}_engineering.json'
    identity=dict(runner=sha256(__file__),probability=sha256(ROOT/'probability.lock.json'),model=sha256(OLD/'pilot_models.py'),device=torch.cuda.get_device_name())
    if report.exists():
        previous=read_json(report)
        if any(previous[k]!=v for k,v in identity.items()):raise ValueError('Engineering verification dependencies changed')
        for arm,row in previous['records'].items():
            if sha256(ROOT/'engineering'/arm/(tag+'_continuous')/'latest.pt')!=row['checkpoint'] or sha256(ROOT/'engineering'/arm/(tag+'_split')/'latest.pt')!=row['split_checkpoint']:raise ValueError('Engineering checkpoint changed')
        return previous['records']
    records={}
    for arm in ARMS:
        continuous=train(data,arm,0,engineering=True,label=tag+'_continuous');expected=torch.load(Path(continuous['directory'])/'latest.pt',map_location='cpu',weights_only=False)
        split=train(data,arm,0,engineering=True,label=tag+'_split',stop_after=2)
        at2=torch.load(Path(split['directory'])/'latest.pt',map_location='cpu',weights_only=False)
        if at2['step']!=2:raise ValueError('Engineering split did not stop at actual2')
        split=train(data,arm,0,engineering=True,label=tag+'_split');actual=torch.load(Path(split['directory'])/'latest.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','critic','critic_optimizer','step','rng'):_same(expected[key],actual[key])
        records[arm]=dict(passed=True,continuous=4,resumed=[2,2],seconds_per_update=expected['history'][-1]['elapsed_seconds']/4,
            checkpoint=sha256(Path(continuous['directory'])/'latest.pt'),split_checkpoint=sha256(Path(split['directory'])/'latest.pt'))
        del expected,actual,at2;torch.cuda.empty_cache()
    write_json(report,dict(passed=True,records=records,**identity,
        scope='actual batch8 paths,4 vs2+2; not a200-step production acceptance'))
    return records


def main():
    global DEADLINE
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('engineering','server'),default='server');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if args.mode=='server':
        if 'A800' not in torch.cuda.get_device_name() or torch.cuda.device_count()!=1:raise ValueError('Formal queue requires exactly one allocated A800')
        DEADLINE=time.monotonic()+int(os.environ.get('GNSS_ALLOWED_SECONDS','3420'))
    for sig in [signal.SIGTERM,signal.SIGINT]+([signal.SIGUSR1] if hasattr(signal,'SIGUSR1') else []):signal.signal(sig,lambda *_:(ROOT/'PAUSE').touch())
    annual=read_json(ROOT/'annual_results.json')
    if not annual['gate_passed']:raise ValueError('Annual geometry mean gate did not pass; no residual training')
    data=Data();results={}
    if args.mode=='engineering':engineering_check(data,'local');return
    benchmark_path=ROOT/'server_benchmark.json'
    if benchmark_path.exists():benchmark=read_json(benchmark_path)
    else:
        timings=engineering_check(data,'server');estimates={}
        for arm in ARMS:
            data.set_fold(0);torch.manual_seed(42);model=(SetDenoiser(attention=False) if arm.startswith('ndp') else make_generator(arm,recompute=True)).cuda()
            h=monthly_subset(data.lock['splits'][0]['inner_hours'])[:1];torch.cuda.synchronize();start=time.monotonic()
            evaluate(data,model,0,h,32,20261130,arm,'inner');torch.cuda.synchronize();evaluation=time.monotonic()-start
            steps=1440 if arm.startswith('ndp') else 72
            estimates[arm]=dict(update_seconds=timings[arm]['seconds_per_update'],evaluation32_hour_seconds=evaluation,
                estimated_seconds=2*steps*timings[arm]['seconds_per_update']+264*evaluation)
            del model;torch.cuda.empty_cache()
        benchmark=dict(runner=sha256(__file__),arms=estimates,with_margin_seconds=1.5*sum(r['estimated_seconds'] for r in estimates.values()),cap_seconds=3600)
        write_json(benchmark_path,benchmark)
    if benchmark['runner']!=sha256(__file__):raise ValueError('Benchmark source differs')
    if benchmark['with_margin_seconds']>DEADLINE-time.monotonic():
        status(stage='STOP_BUDGET',benchmark=benchmark,remaining_seconds=DEADLINE-time.monotonic());return
    immutable_json(ROOT/'training.lock.json',dict(plan=sha256(ROOT/'PLAN.md'),mean=sha256(ROOT/'mean.lock.json'),probability=sha256(ROOT/'probability.lock.json'),
        annual=sha256(ROOT/'annual_results.json'),runner=sha256(__file__),arms=ARMS,folds=2,grid_steps=72,ndp_steps=1440,batch=8,innerK=32,outerK=128,server_jobs=False))
    for fold in (0,1):
        data.set_fold(fold);directory=ROOT/f'fold_{fold}'/'gaussian';directory.mkdir(parents=True,exist_ok=True)
        metrics,arrays=evaluate(data,None,fold,monthly_subset(data.lock['outer_hours']),128,20261131,'gaussian','outer')
        np.savez_compressed(directory/'predictions.npz',**arrays);write_json(directory/'result.json',dict(fold=fold,arm='gaussian',metrics=metrics))
        for arm in ARMS:results[f'{fold}/{arm}']=train(data,arm,fold)
    write_json(ROOT/'results.json',results);status(stage='COMPLETE',models=8,server_jobs=False)


if __name__=='__main__':main()
