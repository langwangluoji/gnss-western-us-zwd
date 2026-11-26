"""Frozen weather-GP coupled local/affine seven-lead trajectory experiments."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path
from .native_process_experiment import Sources,validate_training
from .native_weather_gp_experiment import codes as gp_codes
from .native_weather_gp import WeatherProjectiveGP
from .native_model_seal import validate_model_seal,seal_completed_model
from .native_weather_gp_diffusion import condition265,make_model,trajectory_factors,masked_loss,transform_gp_path,affine_marginals
from .native_weather_experiment import checkpoint,stop_requested
from .native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def codes():
    return {**gp_codes(),**{name:sha(Path(__file__).with_name(name)) for name in
        ('native_weather_gp_diffusion.py','native_weather_gp_diffusion_experiment.py','native_model_seal.py')}}


def validate_gp(protocol,status,source,expected_codes,smoke):
    formal=status['status']=='COMPLETE_WEATHER_GP' and status['steps']==3000
    engineering=smoke and status['status']=='ENGINEERING_SMOKE_COMPLETE' and status['steps']<=4
    if not(formal or engineering):raise ValueError('weather GP must complete registered training')
    if protocol['source']!=source or protocol['code_sha256']!=expected_codes:raise ValueError('frozen GP source changed')


def validate_transport_gp(protocol,seed):
    if protocol['config']['view']!=265 or protocol['config']['seed']!=seed:
        raise ValueError('registered transport requires G265 with matching seed')


class WeatherSources(Sources):
    def __init__(self,args):
        super().__init__(args)
        protocol=read(args.gp/'protocol.json');status=read(args.gp/'status.json')
        validate_gp(protocol,status,self.receipt,gp_codes(),args.smoke_cache)
        if protocol['config']['seed']!=args.seed:raise ValueError('GP seed differs from requested experiment')
        seal=None if args.smoke_cache else validate_model_seal(args.gp,'COMPLETE_WEATHER_GP',gp_codes())
        self.weather_gp=WeatherProjectiveGP(protocol['config']['view'],self.copula).to(args.device).eval()
        self.weather_gp.load_state_dict(torch.load(args.gp/'last.pt',map_location=args.device,weights_only=True))
        self.weather_gp.requires_grad_(False)
        self.receipt={**self.receipt,'weather_gp_protocol_sha256':sha(args.gp/'protocol.json'),
            'weather_gp_weights_sha256':sha(args.gp/'last.pt')}
        if seal is not None:self.receipt['weather_gp_seal_sha256']=sha(args.gp/'model_artifact_seal.json')

    @torch.no_grad()
    def weather_batch(self,i,full=False):
        c,gm,gs,base,query,static=self.batch(i);inputs,_,_,_=self.data.batch(i,72)
        history=inputs['history'];condition=condition265(c,history)
        blocks=self.weather_gp.query_time_blocks(c,history,static,query)
        root,inverse=trajectory_factors(blocks)
        R=self.weather_gp.correlation(c,history,static,query) if full else None
        return condition,gm,gs,base,query,root,inverse,R


def train(args):
    source=WeatherSources(args);data=source.data
    validate_transport_gp(read(args.gp/'protocol.json'),args.seed)
    same=sha(args.cache/'manifest.json')==sha(source.training_cache/'manifest.json')
    validate_training(data.development,len(data),args.steps,args.smoke_cache,same,data.protocol,source.training_protocol)
    if not args.smoke_cache and sha(args.embeddings/'manifest.json')!=source.gp['embedding_manifest_sha256']:
        raise ValueError('canonical Gaussian training embedding required')
    torch.manual_seed(args.seed);model=make_model(args.mode).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.1*args.lr)
    torch.manual_seed(args.seed+261126);args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),
        code_sha256=codes(),parameters=sum(p.numel() for p in model.parameters()),noise_seed=args.seed+261126,
        condition='S175 plus fixed-visible masked90 weather summary; no query pooling',
        whitening='batched7x7 principal roots of query-local frozen GP blocks; float64',
        objective='marginal seven-lead v score; GP-coupled query-local deterministic transport, not full joint score',
        initialization='zero velocity, exact GP in continuous flow; Heun0 roundoff only',
        experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_3000_BLOCKS',strict_oof=False,confirmation_C_D_read=False))
    trace=[];updates=0;total=0;start=time.monotonic()
    for step in range(args.steps):
        i=step%len(data);c,gm,gs,base,query,root,inverse,_=source.weather_batch(i)
        u=((data.tensor(data.a['truth'][i])-base-source.bias)/source.scale-gm)/gs
        loss,count,origins=masked_loss(model,u,c,inverse)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite weather transport loss')
        if count:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1;total+=count
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),supervised_updates=updates,complete_trajectories=count,
                used_origins=origins,total_trajectories=total,last_origin_hour=int(data.a['origins'][i,-1]),seconds=time.monotonic()-start)
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(mode=args.mode,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(args.out,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_WEATHER_TRANSPORT',
        steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start,last_weights_sha256=sha(args.out/'last.pt')))
    if not args.smoke_cache:seal_completed_model(args.out,'COMPLETE_WEATHER_TRANSPORT',codes())


@torch.no_grad()
def evaluate(args):
    if args.smoke_cache:raise ValueError('formal evaluation rejects engineering smoke')
    source=WeatherSources(args);data=source.data
    validate_transport_gp(read(args.gp/'protocol.json'),args.seed)
    if not data.development:raise ValueError('registered development calendar required')
    protocol=read(args.process/'protocol.json');status=read(args.process/'status.json')
    if status['status']!='COMPLETE_WEATHER_TRANSPORT' or status['steps']!=3000:raise ValueError('transport training incomplete')
    if protocol['source']!=source.receipt or protocol['code_sha256']!=codes():raise ValueError('transport source changed')
    validate_model_seal(args.process,'COMPLETE_WEATHER_TRANSPORT',codes())
    mode=protocol['config']['mode'];model=make_model(mode).to(args.device).eval()
    model.load_state_dict(torch.load(args.process/'last.pt',map_location=args.device,weights_only=True))
    if args.members<2:raise ValueError('IID fair scores require at least2 members')
    args.out.mkdir(parents=True,exist_ok=False);torch.manual_seed(args.sample_seed)
    allsamples=[];analytic_mean=[];analytic_std=[];fixed=None;start=time.monotonic()
    for i in range(len(data)):
        c,gm,gs,base,query,root,inverse,R=source.weather_batch(i,full=True);B,Q,_=c.shape
        if Q!=137:raise ValueError('all137 registered queries required')
        if fixed is None:fixed=query.clone()
        if not torch.equal(query,fixed):raise ValueError('evaluation query set changed')
        eta=torch.randn(args.members,B,Q*7,device=args.device).double()
        gp_path=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(R),eta).reshape(args.members*B,Q,7)
        u=transform_gp_path(model,c.repeat(args.members,1,1),gp_path,root.repeat(args.members,1,1,1),args.sample_steps,args.solver)
        mu=base+source.bias+source.scale*gm;std=source.scale*gs
        samples=(mu[None].double()+std[None].double()*u.reshape(args.members,B,Q,7)).float()
        if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite weather transport sample')
        allsamples.append(samples.cpu().numpy())
        if mode=='affine':
            am,ast=affine_marginals(model,c,root,mu,std,args.sample_steps,args.solver)
            analytic_mean.append(am.cpu().numpy());analytic_std.append(ast.cpu().numpy())
        if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='weather_transport_sampling',mode=mode,completed=i+1,total=len(data))),flush=True)
        if stop_requested():raise SystemExit(75)
    months=data.protocol['config']['months'];truth=calendar_shape(np.asarray(data.a['truth']),months)
    deterministic=calendar_shape(np.asarray(source.mean),months)
    samples=np.stack([calendar_shape(x,months) for x in np.stack(allsamples,axis=1)]);del allsamples
    valid=np.isfinite(truth);q=np.quantile(samples,[.1,.9],axis=0);axes=(0,1,2,3);crps=fair_crps(samples,truth)
    payload=dict(deterministic_mean=point_metrics(deterministic,truth),sample_mean=point_metrics(samples.mean(0),truth),
        fair_crps_mm_by_horizon=safe_mean(crps,axes).tolist(),coverage80_by_horizon=(((truth>=q[0])&(truth<=q[1])&valid).sum(axes)/valid.sum(axes)).tolist(),
        forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
        negative_sample_fraction=float((samples<0).mean()),seconds=time.monotonic()-start)
    artifacts=dict(prediction=deterministic,truth=truth,samples=samples,crps=crps.astype('float32'))
    if args.members==32:
        rank=np.sort(samples,axis=0)[[2,29]]
        payload['rank3_30_coverage']=(((truth>=rank[0])&(truth<=rank[1])&valid).sum(axes)/valid.sum(axes)).tolist()
    if analytic_mean:
        am=calendar_shape(np.stack(analytic_mean),months);ast=calendar_shape(np.stack(analytic_std),months)
        artifacts.update(gaussian_distribution_mean=am,gaussian_distribution_std=ast)
        payload['gaussian_distribution_mean']=point_metrics(am,truth)
        payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(gaussian_crps(am,ast,truth),axes).tolist()
        payload['analytic_gaussian_80_coverage']=(((abs((truth-am)/ast)<=1.2815515655446004)&valid).sum(axes)/valid.sum(axes)).tolist()
    np.savez_compressed(args.out/'evaluation.npz',**artifacts);write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),mode=mode,
        process_protocol_sha256=sha(args.process/'protocol.json'),probability_weights_sha256=sha(args.process/'last.pt'),
        members=args.members,sampling_steps=args.sample_steps,solver=args.solver,seed=args.sample_seed,
        independent_iid_member_noise=True,query_count=137,query_selection='fixed registered population; future masks used for scoring only',
        neural_evaluations_per_sample=2*args.sample_steps-1 if args.solver=='heun0' else args.sample_steps,
        prediction_sha256=sha(args.out/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['train','evaluate'],required=True)
    for name in ('cache','embeddings','gaussian','copula','gp','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--process',type=Path);p.add_argument('--mode',choices=['local','affine'],default='local')
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--smoke-cache',action='store_true')
    p.add_argument('--members',type=int,default=32);p.add_argument('--sample-seed',type=int,default=261126)
    p.add_argument('--sample-steps',type=int,default=20);p.add_argument('--solver',choices=['heun0','ddim'],default='heun0')
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.phase=='evaluate' and not args.process:p.error('--process required')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.45)
    (train if args.phase=='train' else evaluate)(args)


if __name__=='__main__':main()
