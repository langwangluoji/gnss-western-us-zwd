"""Train query-consistent covariance; freeze S175 Gaussian point marginals."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path
from .native_process_experiment import Sources,validate_training,codes as source_codes
from .native_weather_gp import WeatherProjectiveGP,gaussian_nll
from .native_copula import spatial_correlation
from .native_weather_experiment import checkpoint,stop_requested
from .native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def codes():
    return {**source_codes(),**{name:sha(Path(__file__).with_name(name)) for name in
        ('native_weather_gp.py','native_weather_gp_experiment.py')}}


def masked_gp_nll(model,u,c175,history,static,query_ids):
    eligible=torch.isfinite(u).all(-1);patterns,inverse=torch.unique(eligible,dim=0,return_inverse=True)
    loss=next(model.parameters()).sum()*0;count=0;used_origins=0
    for group,pattern in enumerate(patterns):
        if not pattern.any():continue
        origins=torch.nonzero(inverse==group).flatten();queries=torch.nonzero(pattern).flatten()
        R=model.correlation(c175[origins][:,queries],history[origins],static[queries],query_ids[queries])
        x=u[origins][:,queries].reshape(len(origins),-1)
        weight=len(origins)*len(queries)
        loss=loss+gaussian_nll(x,R)*weight;count+=weight;used_origins+=len(origins)
    return loss/max(count,1),count,used_origins


def contrast_marginals(mean,std,correlation,left,right):
    mu=mean[:,left].double()-mean[:,right].double()
    a=std[:,left].double();b=std[:,right].double()
    variance=a.square()+b.square()-2*a*b*correlation[:,left,right]
    if variance.min()<-1e-8:raise ValueError('negative contrast variance')
    return mu,variance.clamp_min(0).sqrt()


def reference_correlation(source,static):
    T=torch.tensor(source.copula['time_correlation'],dtype=torch.float64,device=static.device)
    return torch.kron(spatial_correlation(static,source.copula['spatial_rho']).double(),T)


def train(args):
    source=Sources(args);data=source.data
    same=sha(args.cache/'manifest.json')==sha(source.training_cache/'manifest.json')
    validate_training(data.development,len(data),args.steps,args.smoke_cache,same,data.protocol,source.training_protocol)
    if not args.smoke_cache and sha(args.embeddings/'manifest.json')!=source.gp['embedding_manifest_sha256']:
        raise ValueError('formal covariance training requires canonical Gaussian embedding')
    torch.manual_seed(args.seed);model=WeatherProjectiveGP(args.view,source.copula).to(args.device)
    c,gm,gs,base,query,static=source.batch(0);inputs,_,_,_=data.batch(0,72)
    with torch.no_grad():
        warm=model.correlation(c,inputs['history'],static,query);ref=reference_correlation(source,static)
        difference=float((warm-ref[None]).abs().max())
        if difference>1e-6:raise ValueError('initial covariance differs from reference')
        torch.linalg.cholesky(warm)
    del warm,ref,inputs,c,gm,gs,base
    optimizer=torch.optim.AdamW([dict(params=[p for name,p in model.named_parameters() if name!='metric'],lr=args.lr),
        dict(params=[model.metric],lr=args.metric_lr)],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:.1+.9*.5*(1+math.cos(math.pi*min(step,args.steps)/args.steps)))
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),
        parameters=sum(p.numel() for p in model.parameters()),warm_covariance_max_difference=difference,
        covariance_precision='float64 including Cholesky and NLL; neural parameters float32',
        marginals='frozen S175 location and scale; diagonal correlation=1',
        rho_bounds=[model.rho_min,model.rho_max],time_shrinkage=.05,
        spatial_range_multipliers=[.25,4],spatial_range_base_km_km_m=[300,300,1500],
        spatial_kernel='normalized Gaussian convolution with lead-specific ranges; not independently rescaled coordinates',
        optimizer=dict(neural_lr=args.lr,metric_lr=args.metric_lr,eta_min_fraction=.1,scheduler='matched per-group cosine factor'),
        query_consistency='kernel and global weather depend only on fixed visible history and each query; no query pooling',
        experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_3000_BLOCKS',
        strict_oof=False,confirmation_C_D_read=False))
    start=time.monotonic();trace=[];updates=0;total=0
    for step in range(args.steps):
        i=step%len(data);c,gm,gs,base,query,static=source.batch(i);inputs,_,_,_=data.batch(i,72)
        u=((data.tensor(data.a['truth'][i])-base-source.bias)/source.scale-gm)/gs
        loss,count,origins=masked_gp_nll(model,u,c,inputs['history'],static,query)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite covariance NLL')
        if count:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1;total+=count
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),supervised_updates=updates,
                complete_trajectories=count,used_origins=origins,total_trajectories=total,
                group_lrs=scheduler.get_last_lr(),
                last_origin_hour=int(data.a['origins'][i,-1]),seconds=time.monotonic()-start)
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(view=args.view,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(args.out,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_WEATHER_GP',
        steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start))


@torch.no_grad()
def evaluate(args):
    source=Sources(args);data=source.data;model=None
    if not data.development:raise ValueError('evaluation requires a registered development calendar')
    if args.model:
        protocol=read(args.model/'protocol.json')
        if read(args.model/'status.json')['status']!='COMPLETE_WEATHER_GP':raise ValueError('covariance trajectory incomplete')
        if protocol['source']!=source.receipt or protocol['code_sha256']!=codes():raise ValueError('covariance source changed')
        model=WeatherProjectiveGP(protocol['config']['view'],source.copula).to(args.device).eval()
        model.load_state_dict(torch.load(args.model/'last.pt',map_location=args.device,weights_only=True))
    if args.members<2:raise ValueError('fair scores require at least2 IID members')
    args.out.mkdir(parents=True,exist_ok=False)
    months=data.protocol['config']['months'];torch.manual_seed(args.sample_seed)
    allsamples=[];means=[];stds=[];query_ids=None;pairs=None;pair_acc={};increment_sum=0.;increment_count=0
    start=time.monotonic();reference_matrix=None;reference_root=None
    for i in range(len(data)):
        c,gm,gs,base,query,static=source.batch(i);inputs,_,_,_=data.batch(i,72);B,Q,_=c.shape
        if Q!=137:raise ValueError('fixed all137 development queries required')
        if query_ids is None:
            query_ids=query.clone();a,b=np.triu_indices(Q,1)
            xy=static[:,:2].double().cpu().numpy()
            distance=np.linalg.norm((xy[a]-xy[b])*1000,axis=-1)
            pairs=[]
            for lo,hi in ((0,50),(50,150),(150,300),(300,np.inf)):
                keep=(distance>=lo)&(distance<hi);label=f'{lo}-{hi:g}'
                pairs.append((label,data.tensor(a[keep],torch.long),data.tensor(b[keep],torch.long)))
                pair_acc[label]={f'h{h}':dict(sum=np.zeros(months),count=np.zeros(months,dtype='int64')) for h in (0,6)}
        if not torch.equal(query,query_ids):raise ValueError('development query population changed')
        if model is not None:
            R=model.correlation(c,inputs['history'],static,query);root=torch.linalg.cholesky(R)
        else:
            if reference_matrix is None:
                reference_matrix=reference_correlation(source,static);reference_root=torch.linalg.cholesky(reference_matrix)
            R=reference_matrix[None].expand(B,-1,-1);root=reference_root[None].expand(B,-1,-1)
        eta=torch.randn(args.members,B,Q*7,device=args.device).double()
        u=torch.einsum('bij,kbj->kbi',root,eta).reshape(args.members,B,Q,7)
        mu=base+source.bias+source.scale*gm;std=source.scale*gs
        samples=(mu[None].double()+std[None].double()*u).float()
        if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite Gaussian field')
        allsamples.append(samples.cpu().numpy());means.append(mu.cpu().numpy());stds.append(std.cpu().numpy())
        truth=data.tensor(data.a['truth'][i]);meanflat=mu.reshape(B,-1);stdflat=std.reshape(B,-1)
        month=(i//6)%months
        for label,a,b in pairs:
            if not len(a):continue
            for h in (0,6):
                am,ast=contrast_marginals(meanflat,stdflat,R,a*7+h,b*7+h)
                target=(truth[:,a,h]-truth[:,b,h]).cpu().numpy()
                score=gaussian_crps(am.cpu().numpy(),ast.cpu().numpy(),target)
                pair_acc[label][f'h{h}']['sum'][month]+=np.nansum(score)
                pair_acc[label][f'h{h}']['count'][month]+=np.isfinite(score).sum()
        left=(torch.arange(Q,device=args.device)[:,None]*7+torch.arange(1,7,device=args.device)).flatten();right=left-1
        am,ast=contrast_marginals(meanflat,stdflat,R,left,right)
        target=truth.diff(dim=-1).reshape(B,-1).cpu().numpy()
        score=gaussian_crps(am.cpu().numpy(),ast.cpu().numpy(),target)
        increment_sum+=np.nansum(score);increment_count+=np.isfinite(score).sum()
        if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='weather_gp_sampling',completed=i+1,total=len(data))),flush=True)
        if stop_requested():raise SystemExit(75)
    samples=np.stack([calendar_shape(x,months) for x in np.stack(allsamples,axis=1)]);del allsamples
    truth=calendar_shape(np.asarray(data.a['truth']),months);mu=calendar_shape(np.stack(means),months);std=calendar_shape(np.stack(stds),months)
    deterministic=calendar_shape(np.asarray(source.mean),months);crps=fair_crps(samples,truth)
    quantile=np.quantile(samples,[.1,.9],axis=0);valid=np.isfinite(truth);axes=(0,1,2,3)
    payload=dict(deterministic_mean=point_metrics(deterministic,truth),gaussian_distribution_mean=point_metrics(mu,truth),
        sample_mean=point_metrics(samples.mean(0),truth),fair_crps_mm_by_horizon=safe_mean(crps,axes).tolist(),
        analytic_gaussian_crps_mm_by_horizon=safe_mean(gaussian_crps(mu,std,truth),axes).tolist(),
        coverage80_by_horizon=(((truth>=quantile[0])&(truth<=quantile[1])&valid).sum(axes)/valid.sum(axes)).tolist(),
        analytic_gaussian_80_coverage=(((abs((truth-mu)/std)<=1.2815515655446004)&valid).sum(axes)/valid.sum(axes)).tolist(),
        forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
        analytic_forecast_increment_crps_mm=float(increment_sum/increment_count),negative_sample_fraction=float((samples<0).mean()),
        analytic_station_pair_crps={},seconds=time.monotonic()-start)
    for label,leads in pair_acc.items():
        payload['analytic_station_pair_crps'][label]=dict(pair_count=next(len(a) for name,a,b in pairs if name==label))
        for lead,record in leads.items():
            n=record['count'].sum()
            payload['analytic_station_pair_crps'][label][lead]=dict(analytic_crps_mm=float(record['sum'].sum()/n) if n else None,
                monthly_score_sum=record['sum'].tolist(),monthly_valid_count=record['count'].tolist())
    if args.members==32:
        rank=np.sort(samples,axis=0)[[2,29]]
        payload['rank3_30_coverage']=(((truth>=rank[0])&(truth<=rank[1])&valid).sum(axes)/valid.sum(axes)).tolist()
    np.savez_compressed(args.out/'evaluation.npz',prediction=deterministic,truth=truth,samples=samples,
        crps=crps.astype('float32'),gaussian_distribution_mean=mu,gaussian_distribution_std=std)
    write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),
        model_protocol_sha256=sha(args.model/'protocol.json') if model is not None else None,
        probability_weights_sha256=sha(args.model/'last.pt') if model is not None else None,
        members=args.members,seed=args.sample_seed,independent_iid_member_noise=True,
        sampling='float32 IID white noise, float64 Gaussian coloring, physicalfloat32 arrays',
        neural_evaluations_per_sample=0,kernel_head_evaluations_per_origin=1 if model is not None else 0,
        frozen_marginal_head_evaluations_per_block=1,device=args.device,
        marginal_means_and_std='frozen S175 Gaussian; analytic marginal scores identical across covariance candidates',
        prediction_sha256=sha(args.out/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(dict(fair_crps=payload['fair_crps_mm_by_horizon'],analytic_station_pairs=payload['analytic_station_pair_crps'],
        analytic_increment_crps=payload['analytic_forecast_increment_crps_mm'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','reference','evaluate'],required=True)
    for name in ('cache','embeddings','gaussian','copula','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--view',type=int,choices=[175,265],default=265)
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--smoke-cache',action='store_true')
    p.add_argument('--metric-lr',type=float,default=1e-3)
    p.add_argument('--members',type=int,default=32);p.add_argument('--sample-seed',type=int,default=261126)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.phase=='evaluate' and not args.model:p.error('--model required for evaluate')
    if args.phase=='reference' and args.model:p.error('reference requires no covariance weights')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.60)
    (train if args.phase=='train' else evaluate)(args)


if __name__=='__main__':main()
