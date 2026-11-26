"""Frozen S175 Gaussian-reference Full/Local/Affine trajectory experiments."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path
from .native_weather_experiment import CachedData,checkpoint,stop_requested
from .native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean
from .native_probability_context import condition_view,gaussian_from_protocol
from .native_process_models import make_process_model
from .native_process_ops import masked_process_loss,generate_process,time_factors


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def validate_training(development,count,steps,smoke,same,actual,source):
    if development:raise ValueError('development cache cannot train a process')
    if count<1 or steps<1:raise ValueError('positive training budgets required')
    for name in ('data_sha256','source_sha256','I1_sha256'):
        if actual[name]!=source[name]:raise ValueError('underlying cache source differs')
    if smoke:
        if steps>4:raise ValueError('engineering smoke permits at most4 updates')
    elif not same or count!=3000 or steps!=3000:
        raise ValueError('formal process requires exact3000-block Gaussian training cache')


def restore_mm(u,base,bias,scale,gm,gs):return base+bias+scale*(gm+gs*u)


def affine_marginals(m,M,At,base,bias,scale,gm,gs):
    mean=restore_mm(m@At,base,bias,scale,gm,gs)
    factor=(scale*gs)[...,None]*(At@M)
    return mean,factor.square().sum(-1).sqrt()


def codes():
    names=('native_process_experiment.py','native_process_models.py','native_process_ops.py',
        'native_weather_models.py','native_probability_context.py','native_copula.py',
        'native_weather.py','native_weather_cache.py','native_weather_experiment.py')
    return {name:sha(Path(__file__).with_name(name)) for name in names}


def checked_embedding(path,cache=None):
    manifest=read(path/'manifest.json')
    if not manifest['complete']:raise ValueError('embedding is incomplete')
    if cache is not None and manifest['cache_manifest_sha256']!=sha(cache/'manifest.json'):
        raise ValueError('embedding cache differs')
    for name,expected in manifest['files'].items():
        if sha(path/name)!=expected:raise ValueError('embedding array changed')
    return manifest


class Sources:
    def __init__(self,args):
        self.data=CachedData(args.cache,args.device)
        self.gp=read(args.gaussian/'protocol.json')
        if self.gp['config']['domain']!='gaussian' or self.gp['config'].get('condition_view')!='skip':
            raise ValueError('requires registered S175 Gaussian')
        status=read(args.gaussian/'status.json')
        if status['status']!='COMPLETE_PROBABILITY' or status['steps']!=3000:
            raise ValueError('Gaussian reference must complete3000 updates')
        for name,expected in {'native_weather_models.py':self.gp['code_sha256'],**self.gp['extra_model_sources']}.items():
            if sha(Path(__file__).with_name(name))!=expected:raise ValueError('Gaussian implementation changed')
        training=file_path(self.gp['config']['embeddings'])
        if sha(training/'manifest.json')!=self.gp['embedding_manifest_sha256']:
            raise ValueError('Gaussian training embedding changed')
        tm=checked_embedding(training)
        self.training_cache=file_path(self.gp['config']['cache'])
        if tm['cache_manifest_sha256']!=sha(self.training_cache/'manifest.json'):
            raise ValueError('Gaussian training cache changed')
        self.training_protocol=read(self.training_cache/'protocol.json')
        if self.training_protocol['config']['calendar_day']:
            raise ValueError('Gaussian source must use training data')
        for name in ('data_sha256','source_sha256','I1_sha256'):
            if self.data.protocol[name]!=self.training_protocol[name]:
                raise ValueError('cache underlying data or reference mean differs')
        stats=read(training/'statistics.json')
        if stats['role']!='training_only' or stats!=self.gp['normalization']:
            raise ValueError('normalization must use frozen complete training residual statistics')
        manifest=checked_embedding(args.embeddings,args.cache)
        for name in ('mean_protocol_sha256','mean_weights_sha256','model_kind','mean_variant','weights'):
            if manifest[name]!=tm[name]:raise ValueError('frozen mean source differs')
        self.copula=read(args.copula)
        if self.copula['gaussian_weights_sha256']!=sha(args.gaussian/'last.pt') or self.copula['embedding_manifest_sha256']!=sha(training/'manifest.json'):
            raise ValueError('copula does not match frozen Gaussian')
        if not self.copula['no_development_labels_used'] or self.copula['training_blocks']!=3000:
            raise ValueError('copula training role differs')
        if self.copula['code_sha256']!=sha(Path(__file__).with_name('native_copula.py')):
            raise ValueError('copula implementation changed')
        self.gaussian=gaussian_from_protocol(self.gp).to(args.device).eval()
        self.gaussian.load_state_dict(torch.load(args.gaussian/'last.pt',map_location=args.device,weights_only=True))
        self.gaussian.requires_grad_(False)
        self.features=np.load(args.embeddings/'embedding.npy',mmap_mode='r')
        self.mean=np.load(args.embeddings/'mean.npy',mmap_mode='r')
        self.bias=self.data.tensor(stats['bias_mm']);self.scale=self.data.tensor(stats['scale_mm'])
        self.At,self.Ati,_=time_factors(self.copula,args.device)
        self.receipt=dict(gaussian_protocol_sha256=sha(args.gaussian/'protocol.json'),
            gaussian_weights_sha256=sha(args.gaussian/'last.pt'),copula_sha256=sha(args.copula),
            training_embedding_manifest_sha256=sha(training/'manifest.json'),
            mean_weights_sha256=manifest['mean_weights_sha256'],normalization=stats)

    @torch.no_grad()
    def batch(self,i):
        d=self.data;c=condition_view(d.tensor(self.features[i]),d.tensor(d.a['condition'][i]),'skip')
        gm,logstd=self.gaussian(c);gs=logstd.exp();base=d.tensor(self.mean[i])
        query=d.tensor(d.a['query_ids'][i],torch.long);static=d.query_static[query]
        return c,gm,gs,base,query,static


def train(args):
    source=Sources(args);data=source.data
    same=sha(args.cache/'manifest.json')==sha(source.training_cache/'manifest.json')
    validate_training(data.development,len(data),args.steps,args.smoke_cache,same,data.protocol,source.training_protocol)
    if not args.smoke_cache and sha(args.embeddings/'manifest.json')!=source.gp['embedding_manifest_sha256']:
        raise ValueError('formal training embedding must exactly match Gaussian')
    torch.manual_seed(args.seed);model=make_process_model(args.mode).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=.1*args.lr)
    # Equal training k/epsilon streams independent of parameter-count-dependent initialization.
    torch.manual_seed(args.seed+261126)
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),
        parameters=sum(p.numel() for p in model.parameters()),frozen_gaussian_parameters=sum(p.numel() for p in source.gaussian.parameters()),
        initialization='zero velocity; exact same GP sample under Heun0',noise_seed=args.seed+261126,
        forward_noise='k per origin broadcast queries; iid epsilon; common group ordering',
        alpha_endpoints=[float(model.alpha[0]),float(model.alpha[-1])],strict_oof=False,confirmation_C_D_read=False,
        experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_3000_BLOCKS')
    write_json(args.out/'protocol.json',protocol)
    trace=[];updates=0;start=time.monotonic();total_trajectories=0
    for step in range(args.steps):
        i=step%len(data);c,gm,gs,base,query,static=source.batch(i)
        target=(data.tensor(data.a['truth'][i])-base-source.bias)/source.scale
        u=(target-gm)/gs
        loss,count,origins=masked_process_loss(model,u,c,static,source.copula,args.mode,query,source.Ati)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite process objective')
        if count:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1;total_trajectories+=count
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),supervised_updates=updates,
                complete_trajectories=count,used_origins=origins,total_trajectories=total_trajectories,
                last_origin_hour=int(data.a['origins'][i,-1]),seconds=time.monotonic()-start)
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(mode=args.mode,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(args.out,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_PROCESS',
        steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start))


@torch.no_grad()
def evaluate(args):
    source=Sources(args);data=source.data;model=None;mode='reference'
    if not data.development:raise ValueError('evaluation requires registered development calendar')
    if args.process:
        protocol=read(args.process/'protocol.json');status=read(args.process/'status.json')
        if status['status']!='COMPLETE_PROCESS' or status['steps']!=3000:raise ValueError('formal evaluation needs complete process')
        if protocol['source']!=source.receipt or protocol['code_sha256']!=codes():raise ValueError('frozen process source changed')
        mode=protocol['config']['mode'];model=make_process_model(mode).to(args.device).eval()
        model.load_state_dict(torch.load(args.process/'last.pt',map_location=args.device,weights_only=True))
    if args.members<2:raise ValueError('IID fair scores need at least2 members')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.sample_seed);all_samples=[];analytic_mean=[];analytic_std=[]
    fixed_queries=None
    for i in range(len(data)):
        c,gm,gs,base,query,static=source.batch(i);B,Q,_=c.shape
        if Q!=137:raise ValueError('evaluate all137 registered query stations')
        if fixed_queries is None:fixed_queries=query.clone()
        if not torch.equal(query,fixed_queries):raise ValueError('development query population changed')
        eta=torch.randn(args.members*B,Q,7,device=args.device)
        u=generate_process(model,mode,c.repeat(args.members,1,1),eta,static,source.copula,
            steps=args.sample_steps,solver=args.solver,query_ids=query)
        samples=restore_mm(u,base.repeat(args.members,1,1),source.bias,source.scale,
            gm.repeat(args.members,1,1),gs.repeat(args.members,1,1)).reshape(args.members,B,Q,7)
        if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite process sample')
        all_samples.append(samples.cpu().numpy())
        if mode=='reference':
            am=restore_mm(torch.zeros_like(gm),base,source.bias,source.scale,gm,gs);ast=source.scale*gs
        elif mode=='affine':
            m,M=model.affine_map(c,args.sample_steps,args.solver)
            am,ast=affine_marginals(m,M,source.At,base,source.bias,source.scale,gm,gs)
        else:am=None
        if am is not None:analytic_mean.append(am.cpu().numpy());analytic_std.append(ast.cpu().numpy())
        if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='process_sampling',mode=mode,completed=i+1,total=len(data))),flush=True)
        if stop_requested():raise SystemExit(75)
    months=data.protocol['config']['months'];truth=calendar_shape(np.asarray(data.a['truth']),months)
    deterministic=calendar_shape(np.asarray(source.mean),months)
    samples=np.stack([calendar_shape(x,months) for x in np.stack(all_samples,axis=1)])
    del all_samples
    crps=fair_crps(samples,truth);valid=np.isfinite(truth);q=np.quantile(samples,[.1,.9],axis=0)
    rank=np.sort(samples,axis=0)[[2,29]] if args.members==32 else None
    payload=dict(deterministic_mean=point_metrics(deterministic,truth),sample_mean=point_metrics(samples.mean(0),truth),
        fair_crps_mm_by_horizon=safe_mean(crps,(0,1,2,3)).tolist(),
        coverage80_by_horizon=(((truth>=q[0])&(truth<=q[1])&valid).sum((0,1,2,3))/valid.sum((0,1,2,3))).tolist(),
        width80_mm_by_horizon=safe_mean(np.where(valid,q[1]-q[0],np.nan),(0,1,2,3)).tolist(),
        forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
        negative_sample_fraction=float((samples<0).mean()))
    artifacts=dict(prediction=deterministic,truth=truth,samples=samples,crps=crps.astype('float32'))
    if rank is not None:payload['rank3_30_coverage']=(np.sum((truth>=rank[0])&(truth<=rank[1])&valid,axis=(0,1,2,3))/valid.sum((0,1,2,3))).tolist()
    if analytic_mean:
        am=calendar_shape(np.stack(analytic_mean),months);ast=calendar_shape(np.stack(analytic_std),months)
        artifacts.update(gaussian_distribution_mean=am,gaussian_distribution_std=ast)
        payload['gaussian_distribution_mean']=point_metrics(am,truth)
        payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(gaussian_crps(am,ast,truth),(0,1,2,3)).tolist()
        z=(truth-am)/ast
        payload['analytic_gaussian_80_coverage']=(np.sum((abs(z)<=1.2815515655446004)&valid,axis=(0,1,2,3))/valid.sum((0,1,2,3))).tolist()
    np.savez_compressed(args.out/'evaluation.npz',**artifacts)
    write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(source=source.receipt,cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=codes(),mode=mode,
        process_protocol_sha256=sha(args.process/'protocol.json') if args.process else None,
        probability_weights_sha256=sha(args.process/'last.pt') if args.process else None,
        members=args.members,sampling_steps=args.sample_steps,solver=args.solver,seed=args.sample_seed,
        independent_iid_member_noise=True,query_count=137,query_selection='fixed registered population; no future label filtering',
        neural_evaluations_per_sample=(0 if mode=='reference' else 2*args.sample_steps-1 if args.solver=='heun0' else args.sample_steps),
        prediction_sha256=sha(args.out/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','reference','evaluate'],required=True)
    for name in ('cache','embeddings','gaussian','copula','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--process',type=Path);p.add_argument('--mode',choices=['full','local','affine'],default='local')
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--smoke-cache',action='store_true')
    p.add_argument('--members',type=int,default=32);p.add_argument('--sample-steps',type=int,default=20)
    p.add_argument('--sample-seed',type=int,default=261126);p.add_argument('--solver',choices=['heun0','ddim'],default='heun0')
    p.add_argument('--device',default='cuda');args=p.parse_args()
    if args.phase=='evaluate' and not args.process:p.error('--process required for evaluate')
    if args.phase=='reference' and args.process:p.error('reference uses only frozen Gaussian')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.35)
    (train if args.phase=='train' else evaluate)(args)


if __name__=='__main__':main()
