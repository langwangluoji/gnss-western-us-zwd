"""Mean-preserving shape/scale experiments on the frozen G265 process."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path
from .native_process_experiment import validate_training
from .native_weather_gp_diffusion_experiment import WeatherSources,validate_transport_gp,codes as transport_codes
from .native_weather_gp_diffusion import condition265
from .native_model_seal import seal_completed_model,validate_model_seal
from .native_centered_shape import CenteredShape,masked_loss
from .native_weather_experiment import checkpoint,stop_requested
from .native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def codes():
    return {**transport_codes(),**{name:sha(Path(__file__).with_name(name)) for name in
        ('native_centered_shape.py','native_centered_shape_experiment.py','native_weather_evaluation.py')}}


def training_guard(development,count,steps,smoke,same,actual,source,gp,seed):
    validate_training(development,count,steps,smoke,same,actual,source)
    validate_transport_gp(gp,seed)


def load_centered_model(folder,receipt,device):
    protocol=read(folder/'protocol.json');status=read(folder/'status.json')
    if status['status']!='COMPLETE_CENTERED_SHAPE' or status['steps']!=3000:raise ValueError('centered shape training incomplete')
    if protocol['source']!=receipt or protocol['code_sha256']!=codes():raise ValueError('centered shape source changed')
    validate_model_seal(folder,'COMPLETE_CENTERED_SHAPE',codes())
    model=CenteredShape(protocol['config']['mode']).to(device).eval()
    model.load_state_dict(torch.load(folder/'last.pt',map_location=device,weights_only=True))
    return model,protocol


def train(args):
    source=WeatherSources(args);data=source.data
    same=sha(args.cache/'manifest.json')==sha(source.training_cache/'manifest.json')
    training_guard(data.development,len(data),args.steps,args.smoke_cache,same,data.protocol,source.training_protocol,read(args.gp/'protocol.json'),args.seed)
    if not args.smoke_cache and sha(args.embeddings/'manifest.json')!=source.gp['embedding_manifest_sha256']:
        raise ValueError('canonical Gaussian training embeddings required')
    torch.manual_seed(args.seed);model=CenteredShape(args.mode).to(args.device)
    c,gm,gs,base,query,static=source.batch(0);inputs,_,_,_=data.batch(0,72);c=condition265(c,inputs['history'])
    with torch.no_grad():
        eta=torch.randn_like(gm);warm=float((model.transform(c,eta)-eta).abs().max())
        if warm!=0:raise ValueError('centered control failed exact Gaussian initialization')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.1*args.lr)
    torch.manual_seed(args.seed+261127);args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),
        code_sha256=codes(),parameters=sum(p.numel() for p in model.parameters()),warm_gaussian_max_difference=warm,
        fixed_mean='source Transformer3000 mu_base, ignores S175 Gaussian location head and training normalization bias',
        frozen_sigma='source.training_scale * S175.exp(logstd)',shape='normalized Hermite1/2/3, exact mean0 variance1',
        loss='K16 fair CRPS in physical mm / training global scale[h], all finite labels; no full-trajectory requirement',
        training_noise='independent standard Gaussian per member/origin/query/lead; marginal loss expectation matches GP inference',
        noise_seed=args.seed+261127,experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_3000_BLOCKS',
        strict_oof=False,confirmation_C_D_read=False))
    start=time.monotonic();trace=[];updates=0;total=0
    for step in range(args.steps):
        i=step%len(data);c,gm,gs,base,query,static=source.batch(i);inputs,_,_,_=data.batch(i,72)
        condition=condition265(c,inputs['history']);target=(data.tensor(data.a['truth'][i])-base)/source.scale
        loss,count=masked_loss(model,target,condition,gs,args.training_members)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite centered fair CRPS')
        if count:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1;total+=count
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),supervised_updates=updates,finite_label_cells=count,total_label_cells=total,
                last_origin_hour=int(data.a['origins'][i,-1]),seconds=time.monotonic()-start)
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(mode=args.mode,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(args.out,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_CENTERED_SHAPE',
        steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start,last_weights_sha256=sha(args.out/'last.pt')))
    if not args.smoke_cache:seal_completed_model(args.out,'COMPLETE_CENTERED_SHAPE',codes())


@torch.no_grad()
def evaluate(args):
    if args.smoke_cache:raise ValueError('engineering sources cannot enter formal evaluation')
    source=WeatherSources(args);data=source.data
    validate_transport_gp(read(args.gp/'protocol.json'),args.seed)
    if not data.development:raise ValueError('registered development calendar required')
    model=None;mode='reference'
    if args.model:model,protocol=load_centered_model(args.model,source.receipt,args.device);mode=protocol['config']['mode']
    if args.members<2:raise ValueError('fair scoring requires at least2 IID members')
    args.out.mkdir(parents=True,exist_ok=False);torch.manual_seed(args.sample_seed)
    allsamples=[];allstd=[];fixed=None;start=time.monotonic()
    for i in range(len(data)):
        c,gm,gs,base,query,root,inverse,R=source.weather_batch(i,full=True);B,Q,_=c.shape
        if Q!=137:raise ValueError('all137 registered queries required')
        if fixed is None:fixed=query.clone()
        if not torch.equal(query,fixed):raise ValueError('query population changed')
        eta=torch.randn(args.members,B,Q*7,device=args.device).double()
        eta=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(R),eta).reshape(args.members,B,Q,7).float()
        if model is None:z=eta;mult=torch.ones_like(gs)
        else:z=model.transform(c,eta);mult=model.moments(c)[1]
        std=source.scale*gs*mult;samples=base[None]+(source.scale*gs)[None]*z
        if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite centered samples')
        allsamples.append(samples.cpu().numpy());allstd.append(std.cpu().numpy())
        if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='centered_shape_sampling',mode=mode,completed=i+1,total=len(data))),flush=True)
        if stop_requested():raise SystemExit(75)
    months=data.protocol['config']['months'];truth=calendar_shape(np.asarray(data.a['truth']),months)
    mu=calendar_shape(np.asarray(source.mean),months);std=calendar_shape(np.stack(allstd),months)
    samples=np.stack([calendar_shape(x,months) for x in np.stack(allsamples,axis=1)]);del allsamples
    crps=fair_crps(samples,truth);valid=np.isfinite(truth);axes=(0,1,2,3);q=np.quantile(samples,[.1,.9],axis=0)
    payload=dict(deterministic_mean=point_metrics(mu,truth),analytic_distribution_mean=point_metrics(mu,truth),sample_mean=point_metrics(samples.mean(0),truth),
        fair_crps_mm_by_horizon=safe_mean(crps,axes).tolist(),coverage80_by_horizon=(((truth>=q[0])&(truth<=q[1])&valid).sum(axes)/valid.sum(axes)).tolist(),
        forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
        negative_sample_fraction=float((samples<0).mean()),seconds=time.monotonic()-start,
        distribution_family='Gaussian' if mode in ('reference','scale') else 'normalized Hermite1/2/3 transform of weather GP')
    artifacts=dict(prediction=mu,truth=truth,samples=samples,crps=crps.astype('float32'),distribution_mean=mu,distribution_std=std)
    if args.members==32:
        rank=np.sort(samples,axis=0)[[2,29]];payload['rank3_30_coverage']=(((truth>=rank[0])&(truth<=rank[1])&valid).sum(axes)/valid.sum(axes)).tolist()
    if mode in ('reference','scale'):
        artifacts.update(gaussian_distribution_mean=mu,gaussian_distribution_std=std)
        payload['gaussian_distribution_mean']=point_metrics(mu,truth)
        payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(gaussian_crps(mu,std,truth),axes).tolist()
        payload['analytic_gaussian_80_coverage']=(((abs((truth-mu)/std)<=1.2815515655446004)&valid).sum(axes)/valid.sum(axes)).tolist()
    np.savez_compressed(args.out/'evaluation.npz',**artifacts);write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),
        code_sha256=codes(),mode=mode,model_protocol_sha256=sha(args.model/'protocol.json') if args.model else None,
        probability_weights_sha256=sha(args.model/'last.pt') if args.model else None,members=args.members,seed=args.sample_seed,
        independent_iid_member_noise=True,query_count=137,neural_evaluations_per_sample=0,
        coefficient_head_evaluations_per_origin=2 if model is not None else 0,fixed_analytic_mean='source mu_base; no Monte Carlo estimate',
        prediction_sha256=sha(args.out/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['train','reference','evaluate'],required=True)
    for name in ('cache','embeddings','gaussian','copula','gp','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--mode',choices=['scale','shape','both'],default='both')
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--smoke-cache',action='store_true');p.add_argument('--training-members',type=int,choices=[16],default=16)
    p.add_argument('--members',type=int,default=32);p.add_argument('--sample-seed',type=int,default=261126);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.phase=='evaluate' and not args.model:p.error('--model required for evaluate')
    if args.phase=='reference' and args.model:p.error('reference does not load shape weights')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.35)
    (train if args.phase=='train' else evaluate)(args)


if __name__=='__main__':main()
