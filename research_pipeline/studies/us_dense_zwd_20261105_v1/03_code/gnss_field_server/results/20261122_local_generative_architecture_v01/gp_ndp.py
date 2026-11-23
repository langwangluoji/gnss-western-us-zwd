"""Matched observation diffusion using unit-marginal frozen GP noise correlation."""
import math,time
import numpy as np
import torch
from pilot_data import ROOT,Data,monthly_subset,schedule
from pilot_models import SetDenoiser
from run_pilot import log,check_pause,valid_mask,save_state,reuse_result,status
from gnss_field.artifacts import immutable_json,read_json,write_json,sha256
from gnss_field.training.engine import atomic_torch,restore_rng
from gnss_field.models.diffusion import heun_sample,keyed_noise,edm_coefficients
from grid_generation_metrics import score_ensemble


def correlation_factor(projection,remainder,tau):
    if not np.isfinite(tau) or tau<=0:raise ValueError('Positive observation tau required for fixed SPD correlation')
    a=np.asarray(projection,'f8');b=np.asarray(remainder,'f8')
    cov=a@a.T+b@b.T+tau**2*np.eye(len(a));std=np.sqrt(np.diag(cov))
    correlation=cov/std[:,None]/std[None]
    if not np.isfinite(correlation).all():raise ValueError('Nonfinite prior correlation')
    return np.linalg.cholesky(correlation).astype('f4')


@torch.no_grad()
def evaluate_gp(data,model,L,hours,fold,members,seed,role):
    rows=[]
    for h in hours:
        hour=np.array([h]);ix,x,base,amp=data.batch(hour);condition=data.conditions(x).expand(members,-1,-1)
        mask=torch.ones(members,511,dtype=torch.bool,device=data.device)
        initial=keyed_noise(hour,members,seed,shape=(511,),device=data.device).flatten(0,1)@L.T
        z=heun_sample(lambda state,sigma:model.denoise(state,sigma,condition,data.coords,mask),initial,steps=18)
        rows.append((base+10*amp[:,None]*z).cpu().numpy())
    samples=np.array(rows);ix=data.indices(hours);mean=samples.mean(1)
    population=data.a['valid'][ix]&np.array(data.lock['splits'][fold][role])[None]
    metrics=score_ensemble(data.a['truth'][ix],samples,mean,population,data.a['xy'])
    metrics.update(hours=len(hours),mean_mc_se_mm=float(np.mean((samples.std(1,ddof=1)/np.sqrt(members))[population])),
        correction_mean_rms_mm=float(np.sqrt(np.mean((mean-data.a['mean'][ix])[population]**2))),
        point_mean_estimator='full Monte Carlo observation mean',role=role,
        cohort='B2018 incremental adaptive GP-noise follow-up; same twelve calendar-selected hours')
    return metrics,dict(hours=hours,station_ids=data.a['station_ids'],truth=data.a['truth'][ix],mean=mean,samples=samples,population=population,xy=data.a['xy'])


def train(data,L,fold,attention):
    arm='N-GP-set' if attention else 'N-GP-point';directory=ROOT/f'fold_{fold}'/arm;directory.mkdir(parents=True,exist_ok=True)
    steps=480;metadata=dict(arm=arm,fold=fold,steps=steps,data=sha256(ROOT/'data.lock.json'),
        model=sha256(ROOT/'pilot_models.py'),runner=sha256(__file__),plan=sha256(ROOT/'GP_NDP_PLAN.md'),
        factor=sha256(ROOT/'GP_correlation_factor.npy'),change='unit marginal GP correlation noise only')
    existing=reuse_result(directory,metadata)
    if existing is not None:return existing
    immutable_json(directory/'lock.json',metadata);torch.manual_seed(42)
    model=SetDenoiser(attention=attention).to(data.device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    split=data.lock['splits'][fold];batches=schedule(split['train_hours'],steps,batch=8)
    inner=monthly_subset(split['inner_hours']);outer=monthly_subset(data.lock['outer_hours']);history=[];candidates=[];start=0
    if (directory/'latest.pt').exists():
        state=torch.load(directory/'latest.pt',map_location=data.device,weights_only=False)
        if state['metadata']!=metadata:raise ValueError('GP checkpoint contract changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['step'];history=state['history'];candidates=state['candidates'];restore_rng(state['rng'])
    started=time.monotonic();offset=history[-1]['elapsed_seconds'] if history else 0.
    for step in range(start+1,steps+1):
        check_pause();hours=batches[step-1];ix,x,base,amp=data.batch(hours);condition=data.conditions(x)
        mask=valid_mask(data,ix,fold,'fit');clean=torch.where(mask,(data.truth[ix]-base)/(10*amp[:,None]),0.)
        torch.manual_seed(900000+step);sigma=(torch.randn(len(hours),device=data.device)*1.2-1.2).exp()
        noise=torch.randn_like(clean)@L.T
        noisy=torch.where(mask,clean+noise*sigma[:,None],0.)
        prediction=model.denoise(noisy,sigma,condition,data.coords,mask);weight=edm_coefficients(sigma,1.)[-1]
        loss=(((prediction-clean).square()*mask).sum(1)/mask.sum(1)*weight).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite GP-DSM loss')
        opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        lr=1e-4*(.1+.9*(1+math.cos(math.pi*(step-1)/steps))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.step();record=dict(step=step,DSM=float(loss.detach()),gradient_norm=gn,lr=lr,elapsed_seconds=offset+time.monotonic()-started,supervised_station_hours=int(mask.sum()))
        history.append(record);log(directory/'training.jsonl',record)
        if step%80==0:status(stage='GP_SET_TRAIN',arm=arm,fold=fold,step=step,max_steps=steps)
        if step%160==0:
            metrics,_=evaluate_gp(data,model,L,inner,fold,16,20261126,'inner');path=f'candidate_{step:03d}.pt'
            atomic_torch(directory/path,model.state_dict());candidates.append(dict(step=step,metrics=metrics,checkpoint=path));write_json(directory/f'inner_{step:03d}.json',metrics)
            save_state(directory/'latest.pt',model,opt,None,None,step,history,candidates,metadata)
    selected=min(candidates,key=lambda r:(r['metrics']['group_ES_mm'],r['step']))
    model.load_state_dict(torch.load(directory/selected['checkpoint'],map_location=data.device,weights_only=True))
    metrics,arrays=evaluate_gp(data,model,L,outer,fold,32,20261127,'outer');np.savez_compressed(directory/'outer_predictions.npz',**arrays)
    result=dict(arm=arm,fold=fold,steps=steps,selected_step=selected['step'],inner=selected['metrics'],outer=metrics,seconds=history[-1]['elapsed_seconds'],
        checkpoint=selected['checkpoint'],checkpoint_sha256=sha256(directory/selected['checkpoint']),scope='adaptive NDP-inspired GP-noise observation experiment; no continuous field')
    write_json(directory/'result.json',result);del model,opt;torch.cuda.empty_cache();return result


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    data=Data();factor=correlation_factor(data.a['projection'],data.a['remainder'],data.tau)
    path=ROOT/'GP_correlation_factor.npy'
    if path.exists():
        if not np.array_equal(np.load(path),factor):raise ValueError('Frozen GP correlation changed')
    else:np.save(path,factor)
    L=torch.tensor(factor,device=data.device);results={}
    for fold in (0,1):
        for attention in (False,True):results[f'{fold}/{attention}']=train(data,L,fold,attention)
    write_json(ROOT/'GP_NDP_results.json',results);status(stage='EXTENDED_TRAINING_COMPLETE',training_models=20,total_results=24,server_jobs=False)


if __name__=='__main__':main()
