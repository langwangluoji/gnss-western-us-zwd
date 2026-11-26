"""Matched Gaussian, weather-GP and v-diffusion fits on European residuals."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json
from .diffusion_data import condition_view
from .prob_models import gaussian,EuropeWeatherGP,process
from tools.terrain_diffusion.native_weather_gp_experiment import masked_gp_nll
from tools.terrain_diffusion.native_weather_gp_diffusion import trajectory_factors,masked_loss
from tools.terrain_diffusion.native_copula import spatial_correlation


def codes():
    local={name:sha(HERE/name) for name in ('diffusion_data.py','prob_models.py','prob_train.py','model.py','features.py')}
    from maintenance.europe_transfer_20261126.data import ENGINE
    for name in ('native_weather_models.py','native_weather_gp.py','native_process_models.py','native_process_ops.py','native_weather_gp_diffusion.py'):
        local[name]=sha(ENGINE/'tools/terrain_diffusion'/name)
    return local


class Data:
    def __init__(self,seed):
        self.seed=seed;self.root=CACHE/f'diffusion_s{seed}';self.source=json.loads((HERE/f'diffusion_source_s{seed}.json').read_text(encoding='utf-8'))
        if self.source['code_sha256']!=sha(HERE/'diffusion_data.py'):raise ValueError('exporter source changed')
        if self.source['statistics_sha256']!=sha(self.root/'statistics.json'):raise ValueError('residual statistics changed')
        receipt={v['file']:v['sha256'] for v in self.source['files']};self.caches=[]
        for path in sorted(self.root.glob('train_*.npz')):
            if sha(path)!=receipt[path.name]:raise ValueError('training cache changed')
            self.caches.append(dict(np.load(path)))
        self.blocks=[(i,p) for i,c in enumerate(self.caches) for p in range(int(c['fit_start']),len(c['mean'])-3,4)]
        a=np.load(OLD_CACHE/'data.npz');self.static=a['static'];self.hours=a['hours']
        self.stats=json.loads((self.root/'statistics.json').read_text(encoding='utf-8'))
        self.bias=torch.tensor(self.stats['bias_mm'],device='cuda',dtype=torch.float32);self.scale=torch.tensor(self.stats['scale_mm'],device='cuda',dtype=torch.float32)

    def batch(self,index,view,subset=None):
        i,p=self.blocks[index%len(self.blocks)];return self.arrays(self.caches[i],np.arange(p,p+4),view,subset)

    def arrays(self,cache,pos,view,subset=None):
        q=cache['query'];ix=np.arange(len(q)) if subset is None else np.asarray(subset)
        condition=torch.tensor(cache['condition'][pos][:,ix],device='cuda');condition=condition_view(condition,view)
        summary=torch.tensor(cache['summary'][pos],device='cuda');mean=torch.tensor(cache['mean'][pos][:,ix],device='cuda');truth=torch.tensor(cache['truth'][pos][:,ix],device='cuda')
        ids=torch.tensor(q[ix],device='cuda');static=torch.tensor(self.static[q[ix]],device='cuda')
        return condition,summary,mean,truth,static,ids


def destination(seed,view,kind,smoke=False):
    return HERE/('smoke' if smoke else 'probability')/f's{seed}'/view/kind


def gaussian_loss(model,target,condition):
    gm,logs=model(condition);valid=torch.isfinite(target);count=int(valid.sum())
    if not count:return gm.sum()*0,0
    clean=torch.nan_to_num(target)
    return (.5*((clean-gm)*(-logs).exp()).square()+logs)[valid].mean(),count


def load_model(seed,view,kind,smoke=False):
    folder=destination(seed,view,kind,smoke);status=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    if not smoke and status['steps']!=3000:raise ValueError('upstream probability fit incomplete')
    lock=json.loads((folder/'lock.json').read_text(encoding='utf-8'))
    if lock['code_sha256']!=codes():raise ValueError('upstream probability code changed')
    if sha(folder/'last.pt')!=status['last_sha256']:raise ValueError('upstream weights changed')
    model=(gaussian() if kind=='gaussian' else EuropeWeatherGP(json.loads((destination(seed,view,'gaussian',smoke)/'copula.json').read_text(encoding='utf-8'))) if kind=='gp' else process(kind)).cuda().eval()
    model.load_state_dict(torch.load(folder/'last.pt',weights_only=True));model.requires_grad_(False);return model


@torch.no_grad()
def fit_copula(data,view,smoke=False):
    model=load_model(data.seed,view,'gaussian',smoke);folder=destination(data.seed,view,'gaussian',smoke);path=folder/'copula.json'
    if path.exists():raise FileExistsError(path)
    covariance=torch.zeros(7,7,device='cuda',dtype=torch.float64);count=0;num=0.;den=0.
    for cache in data.caches[:2] if smoke else data.caches:
        pos=np.arange(int(cache['fit_start']),len(cache['mean']));c,summary,mean,truth,static,ids=data.arrays(cache,pos,view)
        gm,logs=model(c);z=((truth-mean-data.bias)/data.scale-gm)*(-logs).exp()
        eligible=torch.isfinite(z).all(-1);flat=z[eligible].double();covariance+=flat.T@flat;count+=len(flat)
        kernel=spatial_correlation(static,1);kernel.fill_diagonal_(0);clean=torch.where(eligible[:,:,None],z,0)
        products=torch.einsum('bqh,brh->qr',clean,clean);pair=eligible.float().T@eligible.float()*7
        num+=float((kernel*products).sum());den+=float((kernel.square()*pair).sum())
    diagonal=covariance.diag().clamp_min(1e-8).sqrt();T=covariance/diagonal[:,None]/diagonal[None]
    T=.95*T+.05*torch.eye(7,device='cuda',dtype=torch.float64);rho=float(np.clip(num/max(den,1e-8),1e-4,.95))
    write_json(path,dict(spatial_rho=rho,time_correlation=T.cpu().tolist(),complete_trajectories=count,gaussian_weights_sha256=sha(folder/'last.pt'),source_sha256=sha(HERE/f'diffusion_source_s{data.seed}.json'),only_training_labels=True,strict_oof=False))


def train(data,view,kind,steps=3000,smoke=False):
    torch.manual_seed(data.seed);rng=np.random.default_rng(data.seed+7410)
    g=None;gp=None
    if kind=='gaussian':model=gaussian().cuda()
    elif kind=='gp':
        g=load_model(data.seed,view,'gaussian',smoke);model=EuropeWeatherGP(json.loads((destination(data.seed,view,'gaussian',smoke)/'copula.json').read_text(encoding='utf-8'))).cuda()
    else:
        g=load_model(data.seed,view,'gaussian',smoke);gp=load_model(data.seed,view,'gp',smoke);model=process(kind).cuda()
    parameters=([dict(params=[p for n,p in model.named_parameters() if n!='metric'],lr=3e-4),dict(params=[model.metric],lr=1e-3)] if kind=='gp' else model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda k:.1+.9*.5*(1+np.cos(np.pi*min(k,steps)/steps)))
    torch.manual_seed(data.seed+261126);out=destination(data.seed,view,kind,smoke);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'lock.json',dict(seed=data.seed,view=view,kind=kind,steps=steps,code_sha256=codes(),source_sha256=sha(HERE/f'diffusion_source_s{data.seed}.json'),statistics=data.stats,parameters=sum(p.numel() for p in model.parameters()),gp_query_count=32,query_selection='fixed seeded random subset independent of future labels',mean_fixed=True,strict_oof=False,condition='native175 plus final_mean7 plus explicit_VMF36; base zeros only explicit_VMF36',forecast_hours=list(range(0,37,6))))
    updates=0;total=0;trace=[];start=time.monotonic()
    for step in range(1,steps+1):
        subset=None
        if kind=='gp':
            i,_=data.blocks[(step-1)%len(data.blocks)];n=len(data.caches[i]['query']);subset=np.sort(rng.choice(n,min(n,32),replace=False))
        c,summary,mean,truth,static,ids=data.batch(step-1,view,subset)
        target=(truth-mean-data.bias)/data.scale
        if kind=='gaussian':
            loss,count=gaussian_loss(model,target,c)
        else:
            with torch.no_grad():gm,logs=g(c);u=(target-gm)*(-logs).exp()
            if kind=='gp':loss,count,_=masked_gp_nll(model,u,c,summary,static,ids)
            else:
                with torch.no_grad():_,inverse=trajectory_factors(gp.query_time_blocks(c,summary,static,ids))
                loss,count,_=masked_loss(model,u,torch.cat([c,summary[:,None].expand(-1,c.shape[1],-1)],-1),inverse)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite probability objective')
        if count:
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();updates+=1;total+=count
        if step==1 or step%100==0 or step==steps:
            row=dict(step=step,loss=float(loss.detach()),supervised_updates=updates,observed_units=count,total_observed_units=total,seconds=time.monotonic()-start)
            trace.append(row);write_json(out/'curve.json',trace);write_json(out/'progress.json',row);print(json.dumps(dict(seed=data.seed,view=view,kind=kind,**row)),flush=True)
    torch.save(model.state_dict(),out/'last.pt');torch.save(dict(step=steps,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),out/'checkpoint.pt')
    clocks=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
    write_json(out/'complete.json',dict(steps=steps,supervised_updates=updates,optimizer_state_steps=sorted(set(clocks)),optimizer_state_parameters=len(clocks),total_observed_units=total,seconds=time.monotonic()-start,last_sha256=sha(out/'last.pt'),engineering_smoke=smoke))


def run(seed,steps=3000,smoke=False):
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.45);data=Data(seed)
    for view in ('base','vmf'):
        train(data,view,'gaussian',steps,smoke);fit_copula(data,view,smoke);train(data,view,'gp',steps,smoke)
        for kind in (('local',) if view=='base' else ('affine','local')):train(data,view,kind,steps,smoke)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=3000);p.add_argument('--smoke',action='store_true');args=p.parse_args();run(args.seed,args.steps,args.smoke)
