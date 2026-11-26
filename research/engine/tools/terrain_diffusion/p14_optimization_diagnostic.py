"""Fixed training-season/year checkpoint diagnostics; no weight selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .p14_fusion import FusionData,P14Fusion
from .data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4)
    data=FusionData(a.data,72);model=P14Fusion('p14',72).cuda().eval()
    rows=[];gradients=[];overall=[]
    for step in (0,1000,2000,3000):
        if step==0:model.base.load_state_dict(torch.load(a.out/f'source_weights/s{a.seed}.pt',weights_only=True))
        else:model.load_state_dict(torch.load(a.out/f'p14_s{a.seed}/step_{step}.pt',weights_only=True))
        all_sse=0.;all_count=0
        for year in range(2016,2020):
            for month,season in [(1,'DJF'),(4,'MAM'),(7,'JJA'),(10,'SON')]:
                start=int(np.searchsorted(data.events.hours,np.datetime64(f'{year}-{month:02d}-08','h').astype('int64')))
                sse=0.;count=0
                for offset in (0,12):
                    b=data.batch(np.arange(start+offset,start+offset+12),data.events.layouts[0],False,False,False)
                    with torch.no_grad():err=(model(b)-b['target'])[b['mask']]*data.std[0]
                    sse+=float(err.square().sum());count+=err.numel()
                rows.append(dict(step=step,year=year,season=season,date=f'{year}-{month:02d}-08',
                    micro_rmse_mm=(sse/count)**.5,valid_count=count))
                all_sse+=sse;all_count+=count
        overall.append(dict(step=step,fixed_training_micro_rmse_mm=(all_sse/all_count)**.5))
        grads=[]
        for month in (1,7):
            start=int(np.searchsorted(data.events.hours,np.datetime64(f'2018-{month:02d}-08','h').astype('int64')))
            b=data.batch(np.arange(start,start+12),data.events.layouts[0],False,False,False)
            model.zero_grad(set_to_none=True)
            loss=(model(b)-b['target'])[b['mask']].square().mean()
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach().clone())
        gradients.append(dict(step=step,jan_grad_norm=float(grads[0].norm()),jul_grad_norm=float(grads[1].norm()),
            jan_jul_grad_cosine=float(torch.nn.functional.cosine_similarity(grads[0],grads[1],dim=0))))
        print(json.dumps(dict(step=step,overall=overall[-1],gradient=gradients[-1])),flush=True)
    write_json(a.out/'optimization_diagnostic.json',dict(status='POSTHOC_TRAINING_DIAGNOSTIC_NOT_VALIDATION',
        seed=a.seed,dates='Jan/Apr/Jul/Oct8,24 continuous hours each,2016-2019,layout0',
        overall=overall,by_year_season=rows,gradients=gradients,
        interpretation='Local gradients and fixed training errors diagnose optimization; neither proves generalization or causal seasonal forgetting'))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    steps=[0,1000,2000,3000]
    axes[0].plot(steps,[r['fixed_training_micro_rmse_mm'] for r in overall],marker='o')
    axes[0].set_title('Same 16 training dates')
    for season in ('DJF','MAM','JJA','SON'):
        vals=[np.mean([r['micro_rmse_mm'] for r in rows if r['step']==s and r['season']==season]) for s in steps]
        axes[1].plot(steps,vals,marker='o',label=season)
    axes[1].set_title('Fixed season groups');axes[1].legend()
    for year in range(2016,2020):
        vals=[np.mean([r['micro_rmse_mm'] for r in rows if r['step']==s and r['year']==year]) for s in steps]
        axes[2].plot(steps,vals,marker='o',label=str(year))
    axes[2].set_title('Fixed year groups');axes[2].legend()
    for ax in axes:ax.set(xlabel='Updates',ylabel='Training diagnostic RMSE (mm)')
    fig.tight_layout();fig.savefig(a.out/'optimization_diagnostic.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
