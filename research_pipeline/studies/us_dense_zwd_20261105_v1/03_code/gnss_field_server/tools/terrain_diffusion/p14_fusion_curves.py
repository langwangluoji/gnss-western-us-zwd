"""Post-hoc fixed-population checkpoint curves; never used to select a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .p14_fusion import FusionData,P14Fusion
from .data import write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',default='p14')
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    protocol=json.loads((a.out/'protocol.json').read_text(encoding='utf-8'))
    length=protocol['config']['length']
    data=FusionData(a.data,length)
    folder=a.out/f'{a.kind}_s{a.seed}'
    fit=json.loads((folder/'fit.json').read_text(encoding='utf-8'))
    model=P14Fusion(a.kind,length).cuda().eval()
    batches=[];dates=[]
    for date in ('2018-01-08','2018-07-08'):
        start=int(np.searchsorted(data.events.hours,np.datetime64(date,'h').astype('int64')))
        for offset in (0,12):
            anchors=np.arange(start+offset,start+offset+12)
            batches.append(data.batch(anchors,data.events.layouts[0],False,
                a.kind not in ('p14','features'),a.kind!='p14'))
        dates.append(date)
    points=[]
    for step in (0,1000,2000,3000):
        if step==0:
            model.base.load_state_dict(torch.load(a.out/f'source_weights/s{a.seed}.pt',weights_only=True))
            weight=a.out/f'source_weights/s{a.seed}.pt'
        else:
            weight=folder/f'step_{step}.pt'
            if not weight.exists():continue
            model.load_state_dict(torch.load(weight,weights_only=True))
        sse=0.;count=0
        for b in batches:
            err=(model(b)-b['target'])[b['mask']]*data.std[0]
            sse+=float(err.square().sum());count+=err.numel()
        points.append(dict(step=step,fixed_training_micro_rmse_mm=(sse/count)**.5,weight_sha256=sha(weight)))
    write_json(folder/'fixed_checkpoint_curve.json',dict(status='POSTHOC_TRAINING_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION',
        dates=dates,consecutive_hours_per_date=24,layout=0,points=points,
        note='Same population at every checkpoint; final3000 is primary regardless of this curve'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    trace=fit['trace']
    axes[0].plot([r['step'] for r in trace],[r['training_block_rmse_mm'] for r in trace],color='#43798b',marker='.',lw=1)
    axes[0].axvline(2917,color='#bd7545',ls='--',label='End of first chronological pass')
    axes[0].set(xlabel='Optimizer updates',ylabel='RMSE (mm)',title='Changing consecutive training blocks')
    axes[0].legend(fontsize=7)
    axes[1].plot([r['step'] for r in points],[r['fixed_training_micro_rmse_mm'] for r in points],color='#bd7545',marker='o')
    axes[1].set(xlabel='Optimizer updates',ylabel='RMSE (mm)',title='Same Jan/Jul training diagnostic')
    for r in points:
        axes[1].annotate(f"{r['fixed_training_micro_rmse_mm']:.3f}",(r['step'],r['fixed_training_micro_rmse_mm']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle(f'p14 fusion: {a.kind}, seed {a.seed} — fixed 3000-step endpoint')
    fig.tight_layout();fig.savefig(folder/'training_curves.png',dpi=180);plt.close(fig)
    print(json.dumps(points),flush=True)


if __name__=='__main__':main()
