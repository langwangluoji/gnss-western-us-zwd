"""Verification and charts for completed v11 paired runs."""
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .transport_train import ROOT
from .data import write_json,Dataset
from .metrics import evaluate


def require_full_calendar(config):
    if config.get('rows')!=list(range(12)) or config.get('hours')!=24:
        raise ValueError('Full-calendar report requires all12 months and24 hours')


def run(root):
    root=Path(root);runs=sorted(p.parent for p in root.glob('*/completion.json'));records=[];audit=[];terrain=[];spatial=[]
    data=Dataset(device='cpu');height=data.a['val_geo'][:,2]*1000
    distance=cdist(data.a['val_geo'][:,:2],data.a['val_geo'][:,:2]);ii,jj=np.triu_indices(137,1)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for p in runs:
        h=pd.read_csv(p/'history.csv');conf=json.loads((p/'config.json').read_text(encoding='utf-8'));m=json.loads((p/'metrics.json').read_text())
        selected=h[h.step==m['selected_step']].iloc[0]
        record=dict(run=p.name,variant=conf['variant'],arm=conf['arm'],seed=conf['seed'],selected_step=m['selected_step'],
            macro_rmse_mm=m['macro_rmse_mm'],crps_mm=m['crps_mm'],fixed_train_rmse_mm=float(selected.fixed_train_rmse_mm),
            clip_fraction=float(h.clipped.mean()),median_gradient_norm=float(h.gradient_norm.median()))
        expanded=p/'all_months24/metrics.json'
        if expanded.exists():
            require_full_calendar(json.loads((expanded.parent/'config.json').read_text(encoding='utf-8')))
            e=json.loads(expanded.read_text());record.update(all_months_rmse_mm=e['macro_rmse_mm'],all_months_crps_mm=e['crps_mm'])
            arrays=np.load(p/'all_months24/evaluation.npz');err=arrays['samples'].mean(1)-arrays['truth']
            station_rmse=np.nanmean(np.sqrt(np.nanmean(err**2,axis=1)),axis=0)
            for label,mask in [('below500',height<500),('500to1500',(height>=500)&(height<1500)),('above1500',height>=1500)]:
                terrain.append(dict(run=p.name,group=label,stations=int(mask.sum()),macro_rmse_mm=float(np.nanmean(station_rmse[mask]))))
        records.append(record);pred=np.load(p/'evaluation.npz');diff=0
        for li in range(3):
            new=evaluate(pred['samples'][li],pred['truth'][li])
            for k in new:diff=max(diff,abs(new[k]-m['layouts'][li][k]))
        if diff>1e-5:raise ValueError('Metric recomputation mismatch')
        for label,keep in [('0to50km',(distance[ii,jj]>0)&(distance[ii,jj]<=50)),('50to150km',(distance[ii,jj]>50)&(distance[ii,jj]<=150))]:
            left,right=ii[keep],jj[keep]
            if not len(left):continue
            pair_scores=[]
            for li in range(3):
                s=pred['samples'][li][:,:,left]-pred['samples'][li][:,:,right]
                y=pred['truth'][li][:,left]-pred['truth'][li][:,right]
                pair_scores.append(evaluate(s,y))
            spatial.append(dict(run=p.name,distance_group=label,pairs=len(left),pair_macro_rmse_mm=float(np.mean([x['station_macro_rmse_mm'] for x in pair_scores])),
                                pair_fair_crps_mm=float(np.mean([x['fair_crps_mm'] for x in pair_scores]))))
        exposure=pd.read_csv(p/'exposure.csv');audit.append(dict(run=p.name,metric_difference=diff,
            all_stations_condition_exposed=bool((exposure.context_count>0).all()),all_stations_target_exposed=bool((exposure.target_count>0).all())))
        label=p.name;axes[0,0].plot(h.step,h.loss.rolling(50,min_periods=1).mean(),label=label)
        val=h.dropna(subset=['validation_rmse_mm'])
        axes[0,1].plot(val.step,val.validation_rmse_mm,'o-',label=label)
        axes[1,0].plot(val.step,val.fixed_train_rmse_mm,'o-',label=label)
        axes[1,1].plot(h.step,h.gradient_norm.rolling(50,min_periods=1).median(),label=label)
    for ax,title,y in zip(axes.ravel(),['Training objective (50-update mean)','Fixed validation population','Fixed training reconstruction','Gradient norm before clipping'],['Native objective','Macro RMSE (mm)','Macro RMSE (mm)','L2 norm']):
        ax.set(title=title,ylabel=y,xlabel='Updates');ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    pd.DataFrame(records).to_csv(root/'summary.csv',index=False,lineterminator='\n')
    if terrain:pd.DataFrame(terrain).to_csv(root/'terrain.csv',index=False,lineterminator='\n')
    if spatial:pd.DataFrame(spatial).to_csv(root/'spatial_increments.csv',index=False,lineterminator='\n')
    write_json(root/'summary.json',records);write_json(root/'verification.json',audit)
    if not records:return
    frame=pd.DataFrame(records);fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(frame))
    ax.bar(x,frame.macro_rmse_mm);ax.set_xticks(x,frame.run,rotation=30,ha='right');ax.set(ylabel='Macro RMSE (mm)',title='Matched quarterly selection screen')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)


    spatial_path=root/'spatial_shuffle_all_months24.json'
    if spatial_path.exists():
        spatial_frame=pd.DataFrame(json.loads(spatial_path.read_text())['records'])
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,group in zip(axes,['0to50km','50to150km']):
            sub=spatial_frame[spatial_frame.distance_group==group];x=np.arange(len(sub));original=sub.original_crps_mm.to_numpy()
            ax.bar(x-.18,original,.36,label='Original members')
            ax.bar(x+.18,original+sub.shuffle_mean_minus_original_mm.to_numpy(),.36,label='Spatially shuffled')
            ax.set_xticks(x,sub.run.str.replace('B_','',regex=False),rotation=25,ha='right')
            ax.set(title=group,ylabel='Station-difference fair CRPS (mm)');ax.grid(axis='y',alpha=.2)
        axes[0].legend();fig.suptitle('Frozen 12-month sampled calendar; exploratory spatial test');fig.tight_layout();fig.savefig(root/'spatial_dependence.png',dpi=160);plt.close(fig)
    if 'all_months_rmse_mm' in frame:
        fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(frame))
        ax.bar(x,frame.all_months_rmse_mm);ax.axhline(12.065729,color='black',ls='--',label='Fixed height-IDW')
        ax.set_xticks(x,frame.run,rotation=30,ha='right');ax.set(ylabel='Macro RMSE (mm)',title='Frozen checkpoints: 12 month-start days x 24 hours');ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(root/'expanded_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));run(p.parse_args().root)
