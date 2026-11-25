"""Matched 3000-update temporal means; frozen embeddings for native probability."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from .data import sha,write_json
from .native_weather_models import NativeMean,NativeDiffusion,NativeGaussian
from .native_weather import graph_basis


class CachedData:
    def __init__(self,path,device='cuda',verify=True):
        self.path=Path(path);self.device=device
        self.protocol=json.loads((self.path/'protocol.json').read_text(encoding='utf-8'))
        self.manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
        status=json.loads((self.path/'status.json').read_text(encoding='utf-8'))
        if status['status']!='COMPLETE':raise ValueError('cache is incomplete')
        self.a={}
        for name,expected in self.manifest['files'].items():
            if verify and sha(self.path/name)!=expected:raise ValueError(f'cache hash differs: {name}')
            self.a[Path(name).stem]=np.load(self.path/name,mmap_mode='r')
        self.development=bool(self.protocol['config']['calendar_day'])
        geo=json.loads((self.path/'geometry.json').read_text(encoding='utf-8'))
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


def train_mean(args):
    train=CachedData(args.cache);folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=NativeMean(args.kind,64,args.length,weather=not args.no_weather).cuda()
    ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.1)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        cache_manifest_sha256=sha(args.cache/'manifest.json'),source_protocol=train.protocol,
        model_code_sha256=sha(Path(__file__).with_name('native_weather_models.py')),runner_sha256=sha(Path(__file__)),
        parameters=sum(p.numel() for p in model.parameters()),ema_decay=.999,
        initialization='new mean head exactly zero, I1 current and origin-persistence forecast',strict_oof=False)
    write_json(folder/'protocol.json',protocol);trace=[];start=time.monotonic()
    for step in range(args.steps):
        inputs,truth,base,weights=train.batch(step%len(train),args.length)
        model.train();correction,_=model(**inputs);loss=mean_loss(correction,truth,base,weights)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite mean loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step();scheduler.step()
        with torch.no_grad():
            for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.001)
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start,
                current_correction_rms_mm=float(correction[:,:,0].detach().square().mean().sqrt()),
                last_origin_hour=int(train.a['origins'][step%len(train),-1]))
            trace.append(row);write_json(folder/'curve.json',trace);write_json(folder/'progress.json',row)
            print(json.dumps(dict(kind=args.kind,seed=args.seed,**row)),flush=True)
    torch.save(model.state_dict(),folder/'last.pt');torch.save(ema.state_dict(),folder/'ema.pt')
    write_json(folder/'status.json',dict(status='COMPLETE_MEAN',steps=args.steps,seconds=time.monotonic()-start))


@torch.no_grad()
def export_embeddings(args):
    data=CachedData(args.cache);protocol=json.loads((args.mean/'protocol.json').read_text(encoding='utf-8'))
    config=protocol['config'];model=NativeMean(config['kind'],64,config['length'],weather=not config['no_weather']).cuda().eval()
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
        model_kind=config['kind'],weights=args.weights,files={n:sha(folder/n) for n in ('embedding.npy','mean.npy')},complete=True))


def train_probability(args):
    data=CachedData(args.cache);folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_manifest_sha256']!=sha(args.cache/'manifest.json'):raise ValueError('embedding cache source differs')
    for name,expected in manifest['files'].items():
        if sha(args.embeddings/name)!=expected:raise ValueError('embedding hash differs')
    stats=json.loads((args.embeddings/'statistics.json').read_text(encoding='utf-8'))
    if stats['role']!='training_only':raise ValueError('probability normalization requires training residuals')
    condition=np.load(args.embeddings/'embedding.npy',mmap_mode='r');mean=np.load(args.embeddings/'mean.npy',mmap_mode='r')
    bias=torch.tensor(stats['bias_mm'],device='cuda');scale=torch.tensor(stats['scale_mm'],device='cuda')
    torch.manual_seed(args.seed)
    model=(NativeGaussian() if args.domain=='gaussian' else NativeDiffusion(domain=args.domain)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=args.lr*.1)
    write_json(folder/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),normalization=stats,
        code_sha256=sha(Path(__file__).with_name('native_weather_models.py')),strict_oof=False,
        source_exposure='frozen source fitted all2016-2019; development screen, not strict OOF'))
    trace=[];start=time.monotonic()
    for step in range(args.steps):
        i=step%len(data)
        truth=data.tensor(data.a['truth'][i]);mu=data.tensor(mean[i]);c=data.tensor(condition[i])
        target=(truth-mu-bias)/scale
        if not torch.isfinite(target).all():raise ValueError('training native diffusion requires complete sampled target stations')
        eigen,U=data.graph(i);loss=model.loss(target,c,U,eigen)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step();scheduler.step()
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start)
            if not np.isfinite(row['loss']):raise FloatingPointError('nonfinite probability loss')
            trace.append(row);write_json(folder/'curve.json',trace);print(json.dumps(dict(domain=args.domain,**row)),flush=True)
    torch.save(model.state_dict(),folder/'last.pt')
    write_json(folder/'status.json',dict(status='COMPLETE_PROBABILITY',steps=args.steps,seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['mean','embeddings','probability'],required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['weather','gru','lstm','transformer'],default='transformer')
    p.add_argument('--length',type=int,choices=[24,72],default=72);p.add_argument('--no-weather',action='store_true')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--mean',type=Path)
    p.add_argument('--weights',choices=['last','ema'],default='ema');p.add_argument('--embeddings',type=Path)
    p.add_argument('--domain',choices=['gaussian','spatial','graph','joint'],default='spatial')
    args=p.parse_args();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.35)
    if args.phase=='mean':train_mean(args)
    elif args.phase=='embeddings':export_embeddings(args)
    else:train_probability(args)


if __name__=='__main__':main()
