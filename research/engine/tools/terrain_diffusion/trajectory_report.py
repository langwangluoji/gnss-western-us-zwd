"""24-hour horizon and time-member-shuffle diagnostics from saved predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import Dataset,write_json
from .metrics import evaluate


def run(root):
    root=Path(root);results=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for path in sorted(root.glob('validation24_*/evaluation.npz')):
        config=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        if 'data' not in config:
            import torch
            config['data']=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)['config']['data']
        data=Dataset(config['data'],device='cpu')
        if data.contract['data_sha256']!=config['data_sha256']:raise ValueError('Trajectory metadata data mismatch')
        pack=np.load(path);s=pack['samples'];y=pack['truth'];rng=np.random.default_rng(791)
        shuffle=s.copy()
        for layout in range(len(s)):
            for t in range(s.shape[2]):shuffle[layout,:,t]=s[layout,rng.permutation(s.shape[1]),t]
        original=[];shuffled=[]
        for li in range(3):
            ds=np.concatenate([np.diff(s[li,:,i:i+24],axis=1) for i in range(0,s.shape[2],24)],1)
            dq=np.concatenate([np.diff(shuffle[li,:,i:i+24],axis=1) for i in range(0,s.shape[2],24)],1)
            dy=np.concatenate([np.diff(y[li,i:i+24],axis=0) for i in range(0,y.shape[1],24)],0)
            original.append(evaluate(ds,dy)['fair_crps_mm']);shuffled.append(evaluate(dq,dy)['fair_crps_mm'])
        error=s.mean(1)-y
        horizon=np.sqrt(np.nanmean(error.reshape(3,-1,24,error.shape[-1])**2,axis=(0,1,3)))
        axes[0].plot(np.arange(24),horizon,label=path.parent.name)
        station=0;seq=s[0,:,:24,station];truth=y[0,:24,station];lo,hi=np.quantile(seq,[.05,.95],axis=0)
        axes[1].plot(np.arange(24),seq.mean(0),label=path.parent.name);axes[1].fill_between(np.arange(24),lo,hi,alpha=.15)
        results.append(dict(run=path.parent.name,increment_crps_mm=float(np.mean(original)),
                            shuffled_increment_crps_mm=float(np.mean(shuffled)),shuffle_minus_original_mm=float(np.mean(shuffled)-np.mean(original))))
    axes[0].set(xlabel='Hour since cold start',ylabel='Micro RMSE (mm)',title='24-hour rolling error')
    axes[1].plot(np.arange(24),truth,'k--',label='GNSS');axes[1].set(xlabel='Hour since cold start',ylabel='ZWD (mm)',title=f'Fixed station {data.a["val_ids"][0]} / layout0 / January')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'trajectory24.png',dpi=160);plt.close(fig)
    write_json(root/'time_shuffle.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
