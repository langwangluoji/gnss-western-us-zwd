"""Matched 3000-update temporal means; frozen embeddings for native probability."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .data import sha,write_json
from .native_weather_models import NativeMean,NativeDiffusion,NativeGaussian
from .native_weather import graph_basis
from .native_weather_cache import static
from .native_frequency_tokens import probability_model
from .native_physics_lm import mean_model,causal_station_loss
from .native_paths import file_path


def checkpoint(folder,step,model,optimizer,scheduler,ema=None):
    state=dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    if ema is not None:state['ema']=ema.state_dict()
    torch.save(state,folder/f'checkpoint_{step}.pt')


def stop_requested():
    path=os.environ.get('GNSS_RESEARCH_STOP_FILE')
    return bool(path and Path(path).exists())


class CachedData:
    def __init__(self,path,device='cuda',verify=True):
        self.path=Path(path);self.device=device
        self.protocol=json.loads((self.path/'protocol.json').read_text(encoding='utf-8'))
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        if sha(self.path/'protocol.json')!=self.manifest['protocol_sha256']:raise ValueError('cache protocol hash differs')
        status=json.loads((self.path/'status.json').read_text(encoding='utf-8'))
        if status['status']!='COMPLETE':raise ValueError('cache is incomplete')
        self.a={}
        for name,expected in self.manifest['files'].items():
            if verify and sha(self.path/name)!=expected:raise ValueError(f'cache hash differs: {name}')
            self.a[Path(name).stem]=np.load(self.path/name,mmap_mode='r')
        self.development=bool(self.protocol['config']['calendar_day'])
        geo=json.loads((self.path/'geometry.json').read_text(encoding='utf-8'))
        source=Path(self.protocol['config']['data'])
        if sha(source/'contract.json')!=self.protocol['data_sha256']:raise ValueError('source data protocol differs')
        original=json.loads((source/'contract.json').read_text(encoding='utf-8'))
        if sha(source/'geometry.npz')!=original['files']['geometry.npz']:raise ValueError('source geometry hash differs')
        registered=dict(np.load(source/'geometry.npz'))
        for name,prefix in [('train_static',''),('dev_static','dev_')]:
            if not np.array_equal(np.asarray(geo[name],dtype='float32'),static(registered,prefix)):raise ValueError('cached geometry differs from frozen source')
        self.train_static=torch.tensor(geo['train_static'],device=device)
        self.query_static=torch.tensor(geo['dev_static' if self.development else 'train_static'],device=device)

    def __len__(self):return len(self.a['mean'])

    def tensor(self,x,dtype=torch.float32):return torch.as_tensor(np.array(x),device=self.device,dtype=dtype)

    def batch(self,index,length=72):
        a=self.a;context=self.tensor(a['context_ids'][index],torch.long);query=self.tensor(a['query_ids'][index],torch.long)
        history=self.tensor(a['history'][index])[-(length+11):]
        history=history.unfold(0,length,1).permute(0,3,1,2)
        inputs=dict(condition=self.tensor(a['condition'][index]),history=history,
            context_static=self.train_static[context],context_ids=context,query_static=self.query_static[query])
        return inputs,self.tensor(a['truth'][index]),self.tensor(a['mean'][index]),self.tensor(a['weights'][index])

    def graph(self,index):
        query=self.tensor(self.a['query_ids'][index],torch.long);st=self.query_static[query]
        return graph_basis(st[:,:2]*1000,st[:,2]*3000)


def mean_loss(correction,truth,base,weights):
    error=(correction-(truth-base[:,:,None]))/10
    valid=torch.isfinite(truth);error=torch.where(valid,error,0)
    w=weights[:,:,None]*valid
    losses=(error.square()*w).sum((0,1))/w.sum((0,1)).clamp_min(1)
    level=losses[0]+.5*losses[1:].mean()
    adjacent=valid[1:,:,0]&valid[:-1,:,0]
    delta=error[1:,:,0]-error[:-1,:,0]
    increment=delta[adjacent].square().mean() if adjacent.any() else level*0
    return level+.1*increment


def complete_trajectory_loss(model,target,condition,query_static):
    eligible=torch.isfinite(target).all(-1)
    patterns,inverse=torch.unique(eligible,dim=0,return_inverse=True)
    total=0;used_origins=0;loss=next(model.parameters()).sum()*0
    for group,pattern in enumerate(patterns):
        if not pattern.any():continue
        origins=torch.nonzero(inverse==group).flatten();queries=torch.nonzero(pattern).flatten()
        x=target[origins][:,queries];c=condition[origins][:,queries];st=query_static[queries]
        eigen,U=graph_basis(st[:,:2]*1000,st[:,2]*3000)
        count=len(origins)*len(queries)
        loss=loss+model.loss(x,c,U,eigen)*count;total+=count;used_origins+=len(origins)
    return loss/max(total,1),total,used_origins


def train_mean(args):
    train=CachedData(args.cache)
    if train.development:raise ValueError('development cache cannot train a mean model')
    if args.no_weather and args.kind!='weather':raise ValueError('explicit weather-summary ablation is defined for weather MLP only')
    folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=mean_model(args.kind,args.length,args.variant,weather=not args.no_weather).cuda()
    ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.1)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
        if k in ('phase','cache','out','kind','length','no_weather','seed','steps','lr','variant','aux_weight')},
        cache_manifest_sha256=sha(args.cache/'manifest.json'),source_protocol=train.protocol,
        model_code_sha256=sha(Path(__file__).with_name('native_weather_models.py')),runner_sha256=sha(Path(__file__)),
        parameters=sum(p.numel() for p in model.parameters()),ema_decay=.999,
        extra_model_sources={'native_physics_lm.py':sha(Path(__file__).with_name('native_physics_lm.py'))} if args.variant!='baseline' else {},
        causal_auxiliary_active=args.variant in ('aux','physics_aux'),
        initialization='new mean head exactly zero, I1 current and origin-persistence forecast',strict_oof=False)
    write_json(folder/'protocol.json',protocol);trace=[];start=time.monotonic();updates=0
    for step in range(args.steps):
        inputs,truth,base,weights=train.batch(step%len(train),args.length)
        model.train();aux_loss=torch.tensor(0.,device='cuda');aux_cells=0
        if args.variant in ('aux','physics_aux'):
            correction,_,aux=model(**inputs,return_aux=True)
            aux_loss,aux_cells=causal_station_loss(aux,inputs['history'])
        else:correction,_=model(**inputs)
        query_loss=mean_loss(correction,truth,base,weights);loss=query_loss+args.aux_weight*aux_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite mean loss')
        supervised=bool(torch.isfinite(truth).any()) or aux_cells>0
        if supervised:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.001)
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start,
                current_correction_rms_mm=float(correction[:,:,0].detach().square().mean().sqrt()),
                last_origin_hour=int(train.a['origins'][step%len(train),-1]))
            row['supervised_updates']=updates;row['finite_label_cells']=int(torch.isfinite(truth).sum())
            row.update(query_loss=float(query_loss.detach()),causal_station_loss=float(aux_loss.detach()),auxiliary_label_cells=aux_cells,
                future6_correction_rms_mm=float(correction[:,:,6].detach().square().mean().sqrt()))
            trace.append(row);write_json(folder/'curve.json',trace);write_json(folder/'progress.json',row)
            print(json.dumps(dict(kind=args.kind,seed=args.seed,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(folder,step+1,model,optimizer,scheduler,ema)
        if stop_requested():
            write_json(folder/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),folder/'last.pt');torch.save(ema.state_dict(),folder/'ema.pt')
    write_json(folder/'status.json',dict(status='COMPLETE_MEAN',steps=args.steps,supervised_updates=updates,seconds=time.monotonic()-start))


@torch.no_grad()
def export_embeddings(args):
    data=CachedData(args.cache);protocol=json.loads((args.mean/'protocol.json').read_text(encoding='utf-8'))
    if protocol['model_code_sha256']!=sha(Path(__file__).with_name('native_weather_models.py')):raise ValueError('frozen mean implementation changed')
    for name,expected in protocol.get('extra_model_sources',{}).items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('frozen temporal variant changed')
    config=protocol['config'];variant=config.get('variant','baseline')
    model=mean_model(config['kind'],config['length'],variant,weather=not config['no_weather']).cuda().eval()
    model.load_state_dict(torch.load(args.mean/f'{args.weights}.pt',weights_only=True))
    folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    n=len(data);B,Q=data.a['mean'].shape[1:]
    embedding=np.lib.format.open_memmap(folder/'embedding.npy',mode='w+',dtype='float16',shape=(n,B,Q,64))
    prediction=np.lib.format.open_memmap(folder/'mean.npy',mode='w+',dtype='float32',shape=(n,B,Q,7))
    square=np.zeros(7);total=np.zeros(7);summed=np.zeros(7)
    for i in range(n):
        inputs,truth,base,w=data.batch(i,config['length']);correction,h=model(**inputs)
        pred=base[:,:,None]+correction
        embedding[i]=h.cpu().numpy();prediction[i]=pred.cpu().numpy()
        e=(truth-pred).cpu().numpy().astype('float64');square+=np.nansum(e*e,(0,1));summed+=np.nansum(e,(0,1));total+=np.isfinite(e).sum((0,1))
        if i==0 or (i+1)%500==0:print(json.dumps(dict(stage='embeddings',completed=i+1,total=n)),flush=True)
    embedding.flush();prediction.flush();bias=summed/total;scale=np.sqrt(np.maximum(square/total-bias*bias,1))
    # Development statistics are descriptive only; probability runner loads training stats.
    write_json(folder/'statistics.json',dict(bias_mm=bias.tolist(),scale_mm=scale.tolist(),count=total.tolist(),
        role='descriptive_development_not_fit' if data.development else 'training_only'))
    write_json(folder/'manifest.json',dict(cache=str(args.cache.resolve()),cache_manifest_sha256=sha(args.cache/'manifest.json'),
        mean_protocol_sha256=sha(args.mean/'protocol.json'),mean_weights_sha256=sha(args.mean/f'{args.weights}.pt'),
        model_kind=config['kind'],mean_variant=variant,weights=args.weights,files={n:sha(folder/n) for n in ('embedding.npy','mean.npy')},complete=True))


def train_probability(args):
    data=CachedData(args.cache)
    if data.development:raise ValueError('development cache cannot train a probability model')
    folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_manifest_sha256']!=sha(args.cache/'manifest.json'):raise ValueError('embedding cache source differs')
    for name,expected in manifest['files'].items():
        if sha(args.embeddings/name)!=expected:raise ValueError('embedding hash differs')
    stats=json.loads((args.embeddings/'statistics.json').read_text(encoding='utf-8'))
    if stats['role']!='training_only':raise ValueError('probability normalization requires training residuals')
    condition=np.load(args.embeddings/'embedding.npy',mmap_mode='r');mean=np.load(args.embeddings/'mean.npy',mmap_mode='r')
    bias=torch.tensor(stats['bias_mm'],device='cuda');scale=torch.tensor(stats['scale_mm'],device='cuda')
    torch.manual_seed(args.seed)
    model=probability_model(args.domain).cuda()
    preconditioner=None
    if args.preconditioner:
        if args.domain=='gaussian':raise ValueError('Gaussian is the preconditioner control')
        pre_protocol=json.loads((args.preconditioner/'protocol.json').read_text(encoding='utf-8'))
        if pre_protocol['embedding_manifest_sha256']!=sha(args.embeddings/'manifest.json'):raise ValueError('preconditioner information differs')
        if pre_protocol['normalization']!=stats:raise ValueError('preconditioner normalization differs')
        preconditioner=NativeGaussian().cuda().eval()
        preconditioner.load_state_dict(torch.load(args.preconditioner/'last.pt',weights_only=True))
        for param in preconditioner.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.1)
    write_json(folder/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
        if k in ('phase','cache','out','embeddings','domain','seed','steps','lr','preconditioner')},
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),normalization=stats,
        condition_encoder_kind=manifest['model_kind'],condition_encoder_weights_sha256=manifest['mean_weights_sha256'],
        condition_encoder_variant=manifest.get('mean_variant','baseline'),
        code_sha256=sha(Path(__file__).with_name('native_weather_models.py')),strict_oof=False,
        extra_model_sources={'native_frequency_tokens.py':sha(Path(__file__).with_name('native_frequency_tokens.py'))} if args.domain=='frequency_tokens' else {},
        model_class=type(model).__name__,parameters=sum(p.numel() for p in model.parameters()),
        preconditioner_sha256=sha(args.preconditioner/'last.pt') if args.preconditioner else None,
        source_exposure='frozen source fitted all2016-2019; development screen, not strict OOF'))
    trace=[];start=time.monotonic();updates=0;trajectory_count=0;origin_count=0
    for step in range(args.steps):
        i=step%len(data)
        truth=data.tensor(data.a['truth'][i]);mu=data.tensor(mean[i]);c=data.tensor(condition[i])
        target=(truth-mu-bias)/scale
        if preconditioner is not None:
            with torch.no_grad():location,logscale=preconditioner(c)
            target=(target-location)*(-logscale).exp()
        query=data.tensor(data.a['query_ids'][i],torch.long)
        loss,count,origins=complete_trajectory_loss(model,target,c,data.query_static[query])
        if count:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1
        trajectory_count+=count;origin_count+=origins
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start)
            row.update(supervised_updates=updates,fully_observed_trajectories=trajectory_count,supervised_origins=origin_count)
            if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite probability loss')
            trace.append(row);write_json(folder/'curve.json',trace);print(json.dumps(dict(domain=args.domain,**row)),flush=True)
        if (step+1)%500==0 or stop_requested():checkpoint(folder,step+1,model,optimizer,scheduler)
        if stop_requested():
            write_json(folder/'status.json',dict(status='STOPPED_AT_BUDGET',steps=step+1));raise SystemExit(75)
    torch.save(model.state_dict(),folder/'last.pt')
    write_json(folder/'status.json',dict(status='COMPLETE_PROBABILITY',steps=args.steps,supervised_updates=updates,
        fully_observed_trajectories=trajectory_count,supervised_origins=origin_count,seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['mean','embeddings','probability'],required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['weather','gru','lstm','transformer'],default='transformer')
    p.add_argument('--variant',choices=['baseline','local','aux','physics','physics_aux'],default='baseline')
    p.add_argument('--aux-weight',type=float,default=.25)
    p.add_argument('--length',type=int,choices=[24,72],default=72);p.add_argument('--no-weather',action='store_true')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--mean',type=Path)
    p.add_argument('--weights',choices=['last','ema'],default='ema');p.add_argument('--embeddings',type=Path)
    p.add_argument('--preconditioner',type=Path)
    p.add_argument('--domain',choices=['gaussian','spatial','graph','joint','frequency_tokens'],default='spatial')
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.35)
    if args.phase=='mean':train_mean(args)
    elif args.phase=='embeddings':export_embeddings(args)
    else:train_probability(args)


if __name__=='__main__':main()
