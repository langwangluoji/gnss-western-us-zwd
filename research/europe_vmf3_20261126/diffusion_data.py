"""Native European seven-lead residual and VMF3 conditioning exports."""
import argparse
import json

import numpy as np
import torch

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json
from .model import BackgroundCorrection,batch
from maintenance.europe_transfer_20261126.native import make_model,inputs


def anchored_mean(joint,current):
    return joint+(current-joint[...,0])[...,None]


def condition_view(condition,view):
    if view not in ('base','vmf'):raise ValueError('unknown probability condition')
    result=condition.clone() if isinstance(condition,torch.Tensor) else condition.copy()
    if view=='base':result[...,182:]=0
    return result


def weather_summary(history):
    if history.shape[1]!=16 or history.shape[-1]!=15:raise ValueError('European16-record history required')
    clean=history.copy();clean[...,:3]=np.where(history[...,3:6]>0,history[...,:3],0)
    clean[...,9:11]=np.where(history[...,3:4]>0,history[...,9:11],0)
    return np.concatenate([v for n in (1,4,16) for v in (clean[:,-n:].mean((1,2)),clean[:,-n:].std((1,2)))],-1).astype('f4')


def seal_mean():
    destination=HERE/'selection_2023.json'
    if destination.exists():raise FileExistsError(destination)
    records=[]
    for seed in (42,17):
        score={arm:json.loads((HERE/f'runs/{arm}_s{seed}/complete.json').read_text(encoding='utf-8')) for arm in ('control','vmf3')}
        records.append(dict(seed=seed,scores={arm:v['best_dev_rmse_mm'] for arm,v in score.items()},weights={arm:sha(HERE/f'runs/{arm}_s{seed}/best.pt') for arm in score},joint_weights_sha256=sha(PARENT/f'runs/joint_v2_s{seed}/best.pt')))
    write_json(destination,dict(selected_mean='vmf3',rule='VMF3 branch for requested fusion; fixed2023 spatial-dev monitoring',records=records,test_2024_metrics_used=False,cnn_retraining_cancelled_by_user=True,future_mean_rule='retain frozen native joint increments and shift each lead by new-current minus old-current'))


@torch.no_grad()
def export(seed):
    torch.set_num_threads(4)
    selection=json.loads((HERE/'selection_2023.json').read_text(encoding='utf-8'))
    a=dict(np.load(OLD_CACHE/'data.npz'));old=OLD_CACHE/f'native_v2_s{seed}'
    coef=np.load(old/'i1.npy');point=BackgroundCorrection(True).cuda().eval();point.load_state_dict(torch.load(HERE/f'runs/vmf3_s{seed}/best.pt',weights_only=True));point.time.flatten_parameters()
    joint=make_model(len(a['static'])).cuda().eval();joint.load_state_dict(torch.load(PARENT/f'runs/joint_v2_s{seed}/best.pt',weights_only=True))
    out=CACHE/f'diffusion_s{seed}';out.mkdir(exist_ok=False);sumx=np.zeros(7);sumxx=np.zeros(7);count=np.zeros(7,dtype='i8');receipts=[]
    cutoff=int(np.searchsorted(a['hours'],np.datetime64('2024-01-01','h').astype('i8')))
    for path in [*sorted(old.glob('train_*.npz')),old/'dev.npz',old/'test.npz']:
        native=dict(np.load(path));feature=dict(np.load(CACHE/f's{seed}'/path.name));q=feature['query'];c=native['context'];T=len(native['base']);start=int(native['start'])
        condition=np.zeros((T,len(q),218),dtype='f4');mean=np.zeros((T,len(q),7),dtype='f4');summary=np.zeros((T,90),dtype='f4');truth=np.full_like(mean,np.nan)
        cs=torch.tensor(a['static'][c],device='cuda');qs=torch.tensor(a['static'][q],device='cuda');ids=torch.tensor(c,device='cuda')
        for p in range(15,T,8):
            pos=np.arange(p,min(p+8,T));b=inputs(native,a,pos,q,coef);cond=torch.tensor(b['condition'],device='cuda');hist=torch.tensor(b['history'],device='cuda')
            correction,h=joint(cond,hist,cs,ids,qs);old_mean=torch.tensor(b['background'],device='cuda')[:,:,None]+correction
            model_inputs=batch(feature,pos);current=point(*model_inputs);mu=anchored_mean(old_mean,current)
            vh=model_inputs[-1];B,L,Q,_=vh.shape;state=point.time(vh.permute(0,2,1,3).reshape(B*Q,L,4))[0][:,-1].reshape(B,Q,32)
            condition[pos]=torch.cat([h,cond,mu/100,vh[:,-1],state],-1).cpu().numpy();mean[pos]=mu.cpu().numpy();summary[pos]=weather_summary(b['history'])
            future=pos[:,None]+start+np.arange(7)[None];limit=len(a['hours']) if path.stem=='test' else cutoff
            safe=np.minimum(future,len(a['hours'])-1);y=a['values'][safe[:,:,None],q[None,None],0].transpose(0,2,1)
            truth[pos]=np.where((future<limit)[:,None],y,np.nan)
        if path.name.startswith('train_'):
            residual=(truth[int(native['fit_start']):]-mean[int(native['fit_start']):]).astype('f8');valid=np.isfinite(residual)
            sumx+=np.where(valid,residual,0).sum((0,1));sumxx+=np.where(valid,residual.astype('f8')**2,0).sum((0,1));count+=valid.sum((0,1))
        target=out/path.name;np.savez_compressed(target,condition=condition,mean=mean,summary=summary,truth=truth,query=q,start=start,fit_start=native['fit_start'])
        receipts.append(dict(file=path.name,sha256=sha(target),source_native_sha256=sha(path),source_vmf_feature_sha256=sha(CACHE/f's{seed}'/path.name)))
        print(f'diffusion export seed{seed}: {path.name}',flush=True)
    bias=sumx/count;scale=np.sqrt(np.maximum(sumxx/count-bias*bias,1))
    write_json(out/'statistics.json',dict(bias_mm=bias.tolist(),scale_mm=scale.tolist(),count=count.tolist(),years=[2020,2021,2022,2023],strict_oof=False))
    write_json(HERE/f'diffusion_source_s{seed}.json',dict(seed=seed,selection_sha256=sha(HERE/'selection_2023.json'),mean_weights_sha256=sha(HERE/f'runs/vmf3_s{seed}/best.pt'),joint_weights_sha256=sha(PARENT/f'runs/joint_v2_s{seed}/best.pt'),code_sha256=sha(__file__),condition_dim=218,summary_dim=90,history_records=16,lead_hours=list(range(0,37,6)),statistics_sha256=sha(out/'statistics.json'),files=receipts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal-mean',action='store_true');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.seal_mean:seal_mean()
    else:export(args.seed)
