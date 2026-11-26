"""Joint-scored inducing-innovation probability flow; fixed point mean and GP source."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path
from .native_weather_gp_diffusion_experiment import WeatherSources,validate_transport_gp,codes as source_codes
from .native_process_experiment import validate_training
from .native_inducing_flow import InducingLatentFlow,anchor_locations,inducing_decomposition
from .native_latent_flow import joint_score
from .native_weather_gp_diffusion import condition265
from .native_weather_gp import weather_summary
from .native_weather_experiment import checkpoint,stop_requested
from .native_model_seal import validate_model_seal,seal_completed_model
from .native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def codes():
    return {**source_codes(),**{n:sha(Path(__file__).with_name(n)) for n in
        ['native_latent_flow.py','native_inducing_flow.py','native_inducing_flow_experiment.py','native_weather_evaluation.py']}}


def require_budget(development,count,steps,smoke,same):
    if development or not same:raise ValueError('canonical training population required')
    if smoke:
        if not 0<steps<=4:raise ValueError('smoke permits one to four updates')
    elif count!=3000 or steps!=3000:raise ValueError('formal flow requires all3000 chronological blocks')


def require_calendar(protocol,count,query_ids,seed,actual_hash,expected_hash):
    cfg=protocol['config']
    if cfg['seed']!=seed or cfg['months']!=12 or cfg['calendar_day'] not in (8,15) or count!=216:
        raise ValueError('full registered calendar and matching layout seed required')
    if not expected_hash or actual_hash!=expected_hash:raise ValueError('locked evaluation cache hash differs')
    expected=np.broadcast_to(np.arange(137),(216,137))
    if not np.array_equal(query_ids,expected):raise ValueError('canonical ordered137 query IDs required')


def geometry_pairs(static,rng):
    a,b=np.triu_indices(len(static),1);xy=static[:,:2].detach().cpu().numpy()*1000
    distance=np.linalg.norm(xy[a]-xy[b],axis=-1);selected=[]
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        pool=np.flatnonzero((distance>=lo)&(distance<hi));selected.extend(rng.permutation(pool)[:4].tolist())
    return torch.tensor(np.stack([a[selected],b[selected]],-1),device=static.device,dtype=torch.long)


def coupled_draw(residual,basis,members):
    B,N,D=basis.shape
    eta=torch.randn(members,B,D//7,7,device=basis.device)
    noise=torch.randn(members,B,N,device=basis.device,dtype=torch.float64)
    path=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(residual),noise)
    path+=torch.einsum('bnd,kbd->kbn',basis,eta.double().flatten(2))
    return path.reshape(members,B,N//7,7).float(),eta


@torch.no_grad()
def batch(source,i,anchors,members,seed,subset):
    c,gm,gs,mean,ids,static=source.batch(i);inputs,truth,_,_=source.data.batch(i,72)
    history=inputs['history'];rng=np.random.default_rng(seed+261126+i)
    index=(torch.tensor(np.sort(rng.choice(c.shape[1],32,replace=False)),device=c.device,dtype=torch.long)
        if subset else torch.arange(c.shape[1],device=c.device))
    c=c[:,index];ids=ids[index];static=static[index]
    R,W,residual=inducing_decomposition(source.weather_gp,c,history,static,ids,anchors)
    path,eta=coupled_draw(residual,W,members)
    return dict(condition=condition265(c,history),global_condition=weather_summary(history),mean=mean[:,index],
        std=source.scale*gs[:,index],gp_path=path,latent_noise=eta,basis=W.float()),truth[:,index],R,residual,static,ids,rng


@torch.no_grad()
def affine_std(model,b,residual):
    scale,phi=model.query(b['condition']);B,Q,H,D=phi.shape
    eye=torch.eye(D*H,device=scale.device).reshape(D*H,1,D,H).expand(-1,B,-1,-1)
    F=model.latent(eye,b['global_condition']).flatten(2).permute(1,2,0).double()
    P=(phi[...,None]*torch.eye(H,device=phi.device)[None,None,:,None,:]).reshape(B,Q*H,D*H).double()
    A=scale.flatten(1).double();M=(A[...,None]*b['basis'].double()+P)@F
    variance=A.square()*residual.diagonal(dim1=-2,dim2=-1)+M.square().sum(-1)
    return b['std'].double()*variance.reshape(B,Q,H).sqrt()


def train(args):
    if args.family=='gaussian':raise ValueError('Gaussian reference is frozen, not trained')
    # Even engineering uses a formally sealed G265 source.
    source_args=argparse.Namespace(**vars(args));source_args.smoke_cache=False
    source=WeatherSources(source_args);data=source.data
    validate_transport_gp(read(args.gp/'protocol.json'),args.seed)
    same=sha(args.cache/'manifest.json')==sha(source.training_cache/'manifest.json')
    require_budget(data.development,len(data),args.steps,args.smoke_cache,same)
    validate_training(data.development,len(data),args.steps,args.smoke_cache,same,data.protocol,source.training_protocol)
    if sha(args.embeddings/'manifest.json')!=source.gp['embedding_manifest_sha256']:
        raise ValueError('canonical training embedding required')
    anchors=anchor_locations(data.train_static,8)
    torch.manual_seed(args.seed);model=InducingLatentFlow(args.family,args.domain).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.1*args.lr)
    torch.manual_seed(args.seed+261126);args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),
        code_sha256=codes(),parameters=sum(p.numel() for p in model.parameters()),anchors=anchors.cpu().tolist(),
        mean='frozen TF3000; no S175 mean shift',std='frozen S175 scale times conditional gs',members=8,queries=32,
        noise_seed=args.seed+261126,objective='joint proper endpoint scores: marginal+.25increment+.25pair+.1energy',
        noise='coupled eta and GP; complete independent residual retained',strict_oof=False,confirmation_C_D_read=False,
        experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_3000_BLOCKS'))
    trace=[];start=time.monotonic();updates=0;total=0
    for step in range(args.steps):
        b,truth,R,residual,static,ids,rng=batch(source,step,anchors,8,args.seed,True)
        pairs=geometry_pairs(static,rng);samples=model(**b)
        if step==0:torch.testing.assert_close(samples,b['mean'][None]+b['std'][None]*b['gp_path'],rtol=0,atol=0)
        loss,terms=joint_score(samples,truth,b['mean'],b['std'],R,pairs)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite inducing-flow loss')
        optimizer.zero_grad();loss.backward()
        flow_norm=sum(float(p.grad.square().sum()) for p in model.flow.parameters() if p.grad is not None)**.5
        if step==0 and flow_norm<=0:raise ValueError('initial flow gradient is blocked')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite inducing-flow gradient')
        count=int(torch.isfinite(truth).sum())
        if count:optimizer.step();scheduler.step();updates+=1;total+=count
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                supervised_updates=updates,finite_labels=count,total_labels=total,flow_gradient_norm=flow_norm,
                inducing_variance_fraction=float(b['basis'].square().sum(-1).mean()),
                last_origin_hour=int(data.a['origins'][step,-1]),seconds=time.monotonic()-start)
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(family=args.family,domain=args.domain,**row)),flush=True)
        if (step+1)%500==0 or step+1==args.steps or stop_requested():checkpoint(args.out,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_INDUCING_FLOW',
        steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start,last_weights_sha256=sha(args.out/'last.pt')))
    if not args.smoke_cache:seal_completed_model(args.out,'COMPLETE_INDUCING_FLOW',codes())


@torch.no_grad()
def evaluate(args):
    if args.smoke_cache:raise ValueError('formal evaluation rejects engineering smoke')
    source=WeatherSources(args);data=source.data
    validate_transport_gp(read(args.gp/'protocol.json'),args.seed)
    require_calendar(data.protocol,len(data),np.asarray(data.a['query_ids']),args.seed,
        sha(args.cache/'manifest.json'),args.evaluation_cache_sha256)
    if sha(args.embeddings/'manifest.json')!=args.evaluation_embedding_sha256:
        raise ValueError('locked evaluation embedding hash differs')
    if args.members!=32:raise ValueError('formal evaluation requires32 IID members')
    model=None;seal=None;protocol=None
    anchors=anchor_locations(data.train_static,8)
    if args.family!='gaussian':
        seal=validate_model_seal(args.process,'COMPLETE_INDUCING_FLOW',codes());protocol=read(args.process/'protocol.json')
        if protocol['source']!=source.receipt:raise ValueError('probability source differs')
        if protocol['config']['family']!=args.family or protocol['config']['domain']!=args.domain:
            raise ValueError('trained family/domain differs')
        torch.testing.assert_close(anchors.cpu(),torch.tensor(protocol['anchors'],dtype=torch.float64),rtol=0,atol=0)
        model=InducingLatentFlow(args.family,args.domain).to(args.device).eval()
        model.load_state_dict(torch.load(args.process/'last.pt',map_location=args.device,weights_only=True))
    args.out.mkdir(parents=True,exist_ok=False);torch.manual_seed(args.sample_seed)
    allsamples=[];allstd=[];fixed=None;start=time.monotonic()
    for i in range(len(data)):
        b,truth,R,residual,static,ids,rng=batch(source,i,anchors,args.members,args.seed,False)
        if fixed is None:fixed=ids.clone()
        if len(ids)!=137 or not torch.equal(ids,fixed):raise ValueError('all137 fixed queries required')
        samples=(b['mean'][None]+b['std'][None]*b['gp_path']) if model is None else model(**b)
        if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite inducing-flow samples')
        allsamples.append(samples.cpu().numpy())
        if model is None:allstd.append(b['std'].cpu().numpy())
        elif args.family=='affine':allstd.append(affine_std(model,b,residual).cpu().numpy())
        if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='inducing_sampling',family=args.family,domain=args.domain,completed=i+1,total=len(data))),flush=True)
        if stop_requested():raise SystemExit(75)
    months=data.protocol['config']['months'];truth=calendar_shape(np.asarray(data.a['truth']),months)
    deterministic=calendar_shape(np.asarray(source.mean),months)
    samples=np.stack([calendar_shape(x,months) for x in np.stack(allsamples,axis=1)]);del allsamples
    crps=fair_crps(samples,truth);valid=np.isfinite(truth);quantile=np.quantile(samples,[.1,.9],axis=0);axes=(0,1,2,3)
    payload=dict(deterministic_mean=point_metrics(deterministic,truth),sample_mean=point_metrics(samples.mean(0),truth),
        fair_crps_mm_by_horizon=safe_mean(crps,axes).tolist(),
        coverage80_by_horizon=(((truth>=quantile[0])&(truth<=quantile[1])&valid).sum(axes)/valid.sum(axes)).tolist(),
        forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
        seconds=time.monotonic()-start,analytic_point_mean='exact frozen TF3000; sample mean has finite ensemble error')
    artifacts=dict(prediction=deterministic,truth=truth,samples=samples,crps=crps.astype('float32'))
    if allstd:
        std=calendar_shape(np.stack(allstd),months);artifacts.update(gaussian_distribution_mean=deterministic,gaussian_distribution_std=std)
        payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(gaussian_crps(deterministic,std,truth),axes).tolist()
    np.savez_compressed(args.out/'evaluation.npz',**artifacts);write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),family=args.family,domain=args.domain,
        process_protocol_sha256=seal['protocol_sha256'] if seal else None,probability_weights_sha256=seal['weights_sha256'] if seal else None,
        members=args.members,seed=args.sample_seed,sample_seed=args.sample_seed,layout_training_seed=args.seed,device=args.device,
        anchors=anchors.cpu().tolist(),independent_iid_member_noise=True,
        coupled_gp_and_latent_noise=True,query_count=137,mean='exact frozen TF3000',
        neural_evaluations_per_sample=0 if model is None else 8,prediction_sha256=sha(args.out/'evaluation.npz'),
        strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['train','evaluate'],required=True)
    for n in ('cache','embeddings','gaussian','copula','gp','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--process',type=Path);p.add_argument('--family',choices=['affine','odd','gaussian'],default='odd')
    p.add_argument('--domain',choices=['time','frequency'],default='time');p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--smoke-cache',action='store_true')
    p.add_argument('--members',type=int,default=32);p.add_argument('--sample-seed',type=int,default=261126);p.add_argument('--device',default='cuda')
    p.add_argument('--evaluation-cache-sha256');p.add_argument('--evaluation-embedding-sha256')
    args=p.parse_args();torch.set_num_threads(2)
    for n in ('cache','embeddings','gaussian','copula','gp','out','process'):
        if getattr(args,n) is not None:setattr(args,n,file_path(getattr(args,n)))
    if stop_requested():raise SystemExit(75)
    if args.phase=='evaluate' and args.family!='gaussian' and args.process is None:p.error('--process required for trained evaluation')
    (train if args.phase=='train' else evaluate)(args)


if __name__=='__main__':main()
