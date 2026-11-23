"""Plot actual saved runs; no synthetic performance curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import Dataset,write_json


def run_data(config):
    data=Dataset(config['data'],device='cpu')
    if data.contract['data_sha256']!=config['data_sha256']:raise ValueError('Run data hash differs from reporting data')
    return data


def report(root):
    root=Path(root);runs=sorted(p.parent for p in root.glob('*/completion.json'))
    if not runs:raise ValueError('No completed runs')
    fig,axes=plt.subplots(1,3,figsize=(15,4));summary=[]
    for run in runs:
        hist=pd.read_csv(run/'history.csv');config=json.loads((run/'config.json').read_text(encoding='utf-8'));metrics=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
        label=run.name
        axes[0].plot(hist.step,hist.loss.rolling(20,min_periods=1).mean(),label=label)
        axes[1].plot(hist.step,hist.train_x0_rmse_mm.rolling(20,min_periods=1).mean(),label=label)
        val=hist.dropna(subset=['validation_macro_rmse_mm']);axes[2].plot(val.step,val.validation_macro_rmse_mm,'o-',label=label)
        for item in metrics['layouts']:
            summary.append(dict(run=label,arm=config['arm'],seed=config['seed'],selected_step=metrics['selected_step'],
                                **{k:v for k,v in item.items() if k!='increment'},increment_rmse_mm=item['increment'].get('station_macro_rmse_mm')))
    for ax,title,y in zip(axes,['Training objective (20-update smoothing)','Denoising / regression training error','Cold-start validation'],['Native objective','ZWD RMSE (mm)','Station-macro RMSE (mm)']):
        ax.set(title=title,xlabel='Updates',ylabel=y);ax.grid(alpha=.2)
    axes[2].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    frame=pd.DataFrame(summary);frame.to_csv(root/'metrics_summary.csv',index=False)
    means=frame.groupby('run',sort=False)[['station_macro_rmse_mm','fair_crps_mm','coverage90','increment_rmse_mm']].mean()
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,col in zip(axes,means.columns):
        ax.bar(np.arange(len(means)),means[col]);ax.set_xticks(np.arange(len(means)),means.index,rotation=35,ha='right',fontsize=7);ax.set_title(col);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    terrain=[]
    for run in runs:
        data=run_data(json.loads((run/'config.json').read_text(encoding='utf-8')))
        pack=np.load(run/'evaluation.npz');samples=pack['samples'];truth=pack['truth'];error=samples.mean(1)-truth
        station_rmse=np.sqrt(np.nanmean(error**2,axis=(0,1)));height=data.a['val_static'][:,2]*3000
        for name,mask in [('below500',height<500),('500to1500',(height>=500)&(height<1500)),('above1500',height>=1500)]:
            terrain.append(dict(run=run.name,group=name,stations=int(mask.sum()),macro_rmse_mm=float(np.nanmean(station_rmse[mask]))))
        fig,axes=plt.subplots(1,3,figsize=(15,4));mean=samples.mean(1);valid=np.isfinite(truth)
        axes[0].scatter(truth[valid],mean[valid],s=3,alpha=.2);axes[0].plot([0,300],[0,300],'k--');axes[0].set(xlabel='Observed ZWD (mm)',ylabel='Ensemble mean (mm)',title='All three layouts')
        axes[1].scatter(height,station_rmse,s=12);axes[1].set(xlabel='Station elevation (m)',ylabel='Station RMSE (mm)',title='Terrain diagnosis')
        # Fixed metadata index, never cherry-picked by error.
        j=0;k=samples[0,:, :4,j];y=truth[0,:4,j];x=np.arange(len(y))
        axes[2].plot(x,k.T,color='steelblue',alpha=.3);axes[2].plot(x,y,'ko-',label='GNSS');axes[2].plot(x,k.mean(0),'r--',label='Mean')
        axes[2].set(xlabel='Hour since cold start',ylabel='ZWD (mm)',title=f'Layout 0 / {data.a["val_ids"][j]}');axes[2].legend()
        fig.suptitle(run.name);fig.tight_layout();fig.savefig(run/'diagnostics.png',dpi=150);plt.close(fig)
        levels=np.array([.1,.3,.5,.7,.9]);coverage=[]
        for level in levels:
            lo,hi=np.quantile(samples,[(1-level)/2,(1+level)/2],axis=1);coverage.append(float(np.mean(((truth>=lo)&(truth<=hi))[valid])))
        fig,ax=plt.subplots(figsize=(5,4));ax.plot(levels,coverage,'o-');ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Nominal central interval',ylabel='Empirical coverage',title=run.name);fig.tight_layout();fig.savefig(run/'calibration.png',dpi=150);plt.close(fig)
    pd.DataFrame(terrain).to_csv(root/'terrain_metrics.csv',index=False)
    write_json(root/'summary.json',means.reset_index().to_dict('records'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');report(p.parse_args().root)
