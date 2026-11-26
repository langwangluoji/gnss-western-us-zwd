"""Recoverable sequential stage-one training; no implicit architecture search."""
import copy
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from field_generation_v1.prepare import ROOT,DATA
from field_generation_v1.data import FieldData
from field_generation_v1.geometry import mosaic,query_grid,curvature
from field_generation_v1.model import MeanField,ResidualField,gaussian_losses
from residual_edm_v1.data import write_json,sha256,stratified_hours,canonical_hash

class PauseRequested(Exception):pass

def check_pause():
    if (ROOT/'PAUSE').exists():raise PauseRequested('Pause requested; resume by removing PAUSE and rerunning the same queue')

def next_steps(step,metadata):
    return [] if metadata['training_finished'] else range(step if metadata['pending_evaluation'] else step+1,metadata['budget']+1)

def update_ema(ema,model):
    with torch.no_grad():
        for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.001)
        for e,p in zip(ema.buffers(),model.buffers()):e.copy_(p)

def save_checkpoint(path,model,ema,optimizers,rng,step,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model=model.state_dict(),ema=ema.state_dict(),optimizers=[o.state_dict() for o in optimizers],
        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        step=step,metadata=metadata)
    temporary=path.with_suffix('.partial.pt');torch.save(state,temporary);temporary.replace(path)

def load_checkpoint(path,model,ema,optimizers,rng):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);ema.load_state_dict(state['ema'])
    for o,s in zip(optimizers,state['optimizers']):o.load_state_dict(s)
    rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state

def forward_patches(model,batch):
    # Avoid changing mathematical contributors when memory is constrained.
    if model.training:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda dummy:model(batch),batch['coarse'],use_reentrant=False)
    return model(batch)

class Predictor:
    def __init__(self,data,model,base=None,scale=1,base_hash=None):
        self.data=data;self.model=model;self.base=base;self.scale=scale
        self.base_hash=base_hash
        self.cache=ROOT/'mean_cache';self.cache.mkdir(exist_ok=True)

    @torch.no_grad()
    def frozen_mean(self,t):
        path=self.cache/f'{self.data.contract[:12]}_{self.base_hash}_{t}.npy'
        if path.exists():return torch.as_tensor(np.load(path),device=self.data.device)
        field=Predictor(self.data,self.base).predict_field(t)[0]
        temp=path.with_suffix('.partial.npy');np.save(temp,field.cpu().numpy());temp.replace(path)
        return field

    def predict_field(self,t,station=None):
        data=self.data
        ids=np.arange(len(data.origins)) if station is None else data.patch_ids(station)
        origins=data.origins[ids];offset=origins.min(0);local=origins-offset
        shape=tuple((local.max(0)+24).tolist())
        mu=self.frozen_mean(t) if self.base is not None else None
        results=[];features=[]
        for start in range(0,len(ids),4):
            b=data.batch(t,ids[start:start+4],mu)
            r,f=forward_patches(self.model,b)
            results.append(r)
            if f is not None:features.append(f)
        field=mosaic(torch.cat(results),local,shape)
        y,x=offset
        if mu is not None:field=mu[:,y:y+shape[0],x:x+shape[1]]+self.scale*field
        feat=mosaic(torch.cat(features),local,shape) if features else None
        return field,feat,offset

    def query(self,t,field,features,offset,station_ids):
        data=self.data
        xy=torch.as_tensor(data.qxy[station_ids]-offset[::-1].copy(),device=data.device,dtype=torch.float32)
        heights=torch.as_tensor(data.height[station_ids],device=data.device,dtype=torch.float32)
        mean=query_grid(field,xy)[:,0]*torch.exp(-heights/3000)
        std=None
        if features is not None:
            _,_,_,near,count=data.geometry_for(data.available[t].tobytes(),'stations')
            geometry=torch.stack([heights/3000,torch.as_tensor(near[station_ids]/400000,device=data.device,dtype=torch.float32),
                torch.as_tensor(count[station_ids]/16,device=data.device,dtype=torch.float32)],-1)
            std=self.model.standard_deviation(query_grid(features,xy),geometry,self.surface_scale,self.faithful)
        return mean,std

def evaluate(predictor,hours,station_ids,output=None):
    data=predictor.data;model=predictor.model;was_training=model.training;model.eval()
    totals={int(s):[0.,0.,0.,0] for s in station_ids};records=[];start=0
    partial=Path(output).with_suffix('.partial.npz') if output is not None else None
    from architecture_search import state_fingerprint
    contract=canonical_hash(dict(data=data.contract,model=state_fingerprint(model.state_dict()),hours=np.asarray(hours).tolist(),stations=np.asarray(station_ids).tolist()))
    if partial is not None and partial.exists():
        saved=np.load(partial)
        if str(saved['contract'])!=contract:raise ValueError('Partial evaluation contract mismatch')
        start=int(saved['cursor']);records=saved['records'].tolist()
        for t,s,p,z,sd,lp in records:
            error=p-z;row=totals[int(s)];row[0]+=error*error;row[1]+=abs(error);row[2]+=error;row[3]+=1
    with torch.no_grad():
        for cursor in range(start,len(hours)):
            t=hours[cursor]
            check_pause()
            support,_,_=data.support(int(t))
            selected=[]
            for s in station_ids:
                x,y=np.floor(data.qxy[s]).astype(int)
                if support[y:y+2,x:x+2].all() and np.isfinite(data.values[t,s,0]):selected.append(int(s))
            if not selected:continue
            field,feat,offset=predictor.predict_field(int(t))
            mean,std=predictor.query(int(t),field,feat,offset,selected)
            if not torch.isfinite(mean).all() or (std is not None and not torch.isfinite(std).all()):raise ValueError('Nonfinite evaluation prediction')
            target=data.values[t,selected,0]*np.exp(-data.height[selected]/3000)
            pred=mean.cpu().numpy();sig=std.cpu().numpy() if std is not None else np.full(len(selected),np.nan)
            low=torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(field[None],4,4),size=field.shape[-2:],mode='bilinear',align_corners=False)[0]
            low_mean,_=predictor.query(int(t),low,None,offset,selected)
            for s,p,z,sd,lp in zip(selected,pred,target,sig,low_mean.cpu().numpy()):
                error=float(p-z);row=totals[s];row[0]+=error*error;row[1]+=abs(error);row[2]+=error;row[3]+=1
                records.append((int(t),s,float(p),float(z),float(sd),float(lp)))
            if partial is not None and cursor%50==49:
                temp=partial.with_suffix('.writing.npz');np.savez_compressed(temp,records=np.asarray(records),cursor=cursor+1,contract=contract);temp.replace(partial)
    model.train(was_training)
    rows=np.array([v for v in totals.values() if v[3]>0])
    if not len(rows):raise ValueError('Empty fixed evaluation cohort')
    metric=dict(station_macro_rmse_mm=float(np.mean(np.sqrt(rows[:,0]/rows[:,3]))),rmse_mm=float(np.sqrt(rows[:,0].sum()/rows[:,3].sum())),
        mae_mm=float(rows[:,1].sum()/rows[:,3].sum()),bias_mm=float(rows[:,2].sum()/rows[:,3].sum()),stations=len(rows),station_hours=int(rows[:,3].sum()))
    a=np.array(records)
    if np.isfinite(a[:,4]).all():
        from scipy.special import ndtr
        z=(a[:,2]-a[:,3])/a[:,4];sd=a[:,4]
        metric['crps_mm']=float(np.mean(sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))))
        metric['nll']=float(np.mean(np.log(sd)+.5*z*z+.5*np.log(2*np.pi)))
        metric['intervals']={str(level):dict(coverage=float(np.mean(np.abs(z)<=q)),width_mm=float(np.mean(2*q*sd))) for level,q in [(80,1.2815515655),(90,1.644853627),(95,1.959963985)]}
    if output is not None:
        np.savez_compressed(output,records=a,columns=np.array(['time_index','station_index','prediction_mm','target_mm','std_mm','coarsened_prediction_mm']))
        partial.unlink(missing_ok=True)
    return metric

def status(**kwargs):write_json(ROOT/'status.json',dict(updated=time.time(),pid=os.getpid(),**kwargs))

def train_arm(name,data,base=None,scales=None,base_hash=None):
    check_pause()
    directory=ROOT/name;directory.mkdir(exist_ok=True)
    if (directory/'selected.json').exists():return json.loads((directory/'selected.json').read_text())
    torch.manual_seed(42);rng=np.random.default_rng(42)
    model=(MeanField() if name=='M0' else ResidualField()).to(data.device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    variance=list(model.variance_head.parameters()) if name!='M0' else []
    variance_ids={id(p) for p in variance};mean_params=[p for p in model.parameters() if id(p) not in variance_ids]
    optimizers=[torch.optim.AdamW(mean_params,lr=1e-4,weight_decay=1e-4)]
    if variance:optimizers.append(torch.optim.AdamW(variance,lr=1e-4,weight_decay=1e-4))
    predictor=Predictor(data,model,base,scales['z0'] if scales else 1,base_hash)
    predictor.faithful=name=='M2-F';predictor.surface_scale=scales['surface'] if scales else data.stats['surface_std']
    evaluator=Predictor(data,ema,base,predictor.scale,base_hash);evaluator.faithful=predictor.faithful;evaluator.surface_scale=predictor.surface_scale
    samples={int(s):np.concatenate([data.qualified_hours(s,y) for y in (2016,2017,2018)]) for s in data.train}
    samples={s:v for s,v in samples.items() if len(v)};stations=np.array(sorted(samples))
    if not len(stations):raise ValueError('No supported training stations')
    dev=np.flatnonzero(data.dates.year==2019);screen=stratified_hours(data.hours,dev)
    np.savez_compressed(ROOT/'evaluation_manifest.npz',screen=screen,development=dev,stations=data.dev)
    budget=30000 if name=='M0' else 20000
    metadata=dict(best=[],patience=0,arm=name,budget=budget,base_hash=base_hash,scales=scales,contract=data.contract,pending_evaluation=False,training_finished=False)
    step0=0
    if (directory/'last.pt').exists():
        state=load_checkpoint(directory/'last.pt',model,ema,optimizers,rng);step0=state['step']
        for key in ['contract','arm','budget','base_hash','scales']:
            if state['metadata'].get(key)!=metadata[key]:raise ValueError(f'Checkpoint contract mismatch: {key}')
        metadata=state['metadata']
    started=time.time();status(phase='training',arm=name,step=step0,budget=budget)
    steps=next_steps(step0,metadata)
    for step in steps:
        if (ROOT/'PAUSE').exists():
            save_checkpoint(directory/'last.pt',model,ema,optimizers,rng,step if metadata['pending_evaluation'] else step-1,metadata)
            check_pause()
        if not (step==step0 and metadata['pending_evaluation']):
            lr=1e-4*step/500 if step<=500 else 1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*(step-500)/(budget-500)))
            for o in optimizers:
                for group in o.param_groups:group['lr']=lr
                o.zero_grad(set_to_none=True)
            sums=np.zeros(4)
            for _ in range(8):
                s=int(rng.choice(stations));t=int(rng.choice(samples[s]))
                field,features,offset=predictor.predict_field(t,s)
                mean,std=predictor.query(t,field,features,offset,[s])
                target=mean.new_tensor([data.values[t,s,0]*np.exp(-data.height[s]/3000)])
                if std is None:mse=.5*((mean-target)/predictor.surface_scale).square().mean();nll=mse*0
                else:mse,nll=gaussian_losses(mean,target,std.square(),predictor.surface_scale,predictor.faithful)
                support,_,_=data.support(t);y,x=offset;mask=torch.as_tensor(support[y:y+field.shape[1],x:x+field.shape[2]],device=data.device)
                # Only query-neighbor center stencils have every global contributor present.
                local=torch.zeros_like(mask);qx,qy=np.floor(data.qxy[s]-offset[::-1]).astype(int)
                local[max(0,qy-1):qy+3,max(0,qx-1):qx+3]=mask[max(0,qy-1):qy+3,max(0,qx-1):qx+3]
                reg=curvature(field,local,data.stats['z0_std'])
                loss=(mse if name=='M0' else (mse+nll if predictor.faithful else nll))+.001*reg
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss/8).backward();sums+=np.array([loss.item(),mse.item(),nll.item(),reg.item()])/8
            torch.nn.utils.clip_grad_norm_(mean_params,1,error_if_nonfinite=True)
            if variance:torch.nn.utils.clip_grad_norm_(variance,1,error_if_nonfinite=True)
            for o in optimizers:o.step()
            update_ema(ema,model)
            metadata['pending_evaluation']=step==200 or step%1000==0
            if step%100==0:
                record=dict(step=step,loss=sums[0],mean_loss=sums[1],nll=sums[2],curvature=sums[3],lr=lr,seconds=time.time()-started,
                    steps_per_second=(step-step0)/(time.time()-started),gpu_memory_mb=torch.cuda.max_memory_allocated()/1048576,updated=time.time())
                with (directory/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                print(name,json.dumps(record),flush=True);save_checkpoint(directory/'last.pt',model,ema,optimizers,rng,step,metadata);status(phase='training',arm=name,step=step,budget=budget)
        if step==200 or step%1000==0:
            status(phase='development',arm=name,step=step,budget=budget)
            metrics=evaluate(evaluator,screen,data.dev,directory/f'screen_{step:06d}.npz');metrics.update(step=step,startup=step==200,scope='2019_screen',updated=time.time())
            write_json(directory/f'development_{step:06d}.json',metrics);print('DEVELOPMENT',name,json.dumps(metrics),flush=True)
            if step!=200:
                score=metrics['station_macro_rmse_mm'];previous=min([r['score'] for r in metadata['best']],default=float('inf'))
                metadata['patience']=0 if score<previous else metadata['patience']+1
                path=directory/f'candidate_{step:06d}.pt'
                if len(metadata['best'])<3 or score<max(r['score'] for r in metadata['best']):
                    metadata['best'].append(dict(step=step,score=score,path=str(path)));metadata['best']=sorted(metadata['best'],key=lambda r:(r['score'],r['step']))[:3]
                    save_checkpoint(path,model,ema,optimizers,rng,step,metadata)
                write_json(directory/'best.json',min(metadata['best'],key=lambda r:r['score']))
            metadata['pending_evaluation']=False
            metadata['training_finished']=metadata['patience']>=6 or step>=budget
            save_checkpoint(directory/'last.pt',model,ema,optimizers,rng,step,metadata)
            if metadata['training_finished']:break
        if (ROOT/'PAUSE').exists():
            save_checkpoint(directory/'last.pt',model,ema,optimizers,rng,step,metadata);status(phase='paused',arm=name,step=step);return None
    ranking=[]
    for candidate in metadata['best']:
        check_pause()
        state=torch.load(candidate['path'],map_location=data.device,weights_only=False);ema.load_state_dict(state['ema'])
        metrics=evaluate(evaluator,dev,data.dev,directory/f"full2019_{candidate['step']}.npz")
        ranking.append(dict(**candidate,metrics=metrics))
        write_json(directory/'full2019.json',ranking)
    selected=min(ranking,key=lambda r:(r['metrics']['station_macro_rmse_mm'],r['metrics'].get('crps_mm',0)))
    write_json(directory/'selected.json',selected);return selected

def main():
    check_pause()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    data=FieldData('cuda')
    gates=json.loads((ROOT/'smoke.json').read_text())
    if not gates['passed'] or gates.get('contract')!=data.contract:raise ValueError('Source/data-bound real-data gates must pass before training')
    restore=json.loads((ROOT/'restore_gate.json').read_text())
    if not restore['passed'] or not restore['exact']:raise ValueError('Actual-model exact resume gate must pass')
    selected=train_arm('M0',data)
    if selected is None:return
    base=MeanField().to(data.device);base.load_state_dict(torch.load(selected['path'],map_location=data.device,weights_only=False)['ema']);base.eval().requires_grad_(False)
    base_hash=sha256(selected['path'])
    # Exact streaming training-station residual scales; no development labels.
    scale_path=ROOT/'residual_scales.json'
    if scale_path.exists():
        scales=json.loads(scale_path.read_text());assert scales['base_hash']==base_hash
    else:
        predictor=Predictor(data,base);sse=surface_sse=0.;count=0;start=18
        partial_scale=ROOT/'residual_scale_progress.json'
        if partial_scale.exists():
            saved=json.loads(partial_scale.read_text())
            if saved['base_hash']!=base_hash or saved['contract']!=data.contract:raise ValueError('Scale progress contract mismatch')
            sse=saved['sse'];surface_sse=saved['surface_sse'];count=saved['count'];start=saved['cursor']
        status(phase='residual_scale',arm='M0')
        with torch.no_grad():
            for t in range(start,data.train_stop):
                check_pause()
                support,_,_=data.support(t);ids=[]
                for s in data.train:
                    x,y=np.floor(data.qxy[s]).astype(int)
                    if support[y:y+2,x:x+2].all() and np.isfinite(data.values[t,s,0]):ids.append(s)
                if not ids:continue
                field,feat,offset=predictor.predict_field(t);xy=torch.as_tensor(data.qxy[ids],device=data.device,dtype=torch.float32)
                err=query_grid(field,xy)[:,0].cpu().numpy().astype('f8')-data.values[t,ids,0].astype('f8')
                sse+=float((err*err).sum());surface_sse+=float(((err*np.exp(-data.height[ids]/3000))**2).sum());count+=len(ids)
                if t%100==99:write_json(partial_scale,dict(base_hash=base_hash,contract=data.contract,sse=sse,surface_sse=surface_sse,count=count,cursor=t+1))
                if t%1000==0:print('SCALE',t,data.train_stop,flush=True)
        scales=dict(z0=float(np.sqrt(sse/count)),surface=float(np.sqrt(surface_sse/count)),count=count,base_hash=base_hash)
        if min(scales['z0'],scales['surface'])<1e-6:raise ValueError('Degenerate residual scale')
        write_json(scale_path,scales)
    for arm in ['M2-F','M2-J']:
        if train_arm(arm,data,base,scales,base_hash) is None:return
    from field_generation_v1.report import run
    run(data,base,base_hash,scales)
    status(phase='complete',arm='M2-J',message='Training and locked evaluation complete; no next-stage training launched')

if __name__=='__main__':
    try:main()
    except PauseRequested as error:status(phase='paused',message=str(error))
    except Exception as error:
        status(phase='failed',error=repr(error));raise
