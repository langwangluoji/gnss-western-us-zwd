"""Empirical CRPS alongside fair CRPS for potentially dependent calibrated members."""
import csv
import numpy as np
from .data import write_json
from .residual_experiment import ROOT,load_pack
from .residual_queue import EXPERIMENTS


def crps_fields(samples,truth):
    k=len(samples);s=np.sort(samples.astype('f8'),axis=0);weights=(2*np.arange(1,k+1)-k-1)[:,None,None]
    spread=(weights*s).sum(0);first=np.abs(samples-truth[None]).mean(0)
    return first-spread/(k*k),first-spread/(k*(k-1)) if k>1 else first


def run():
    records=[]
    for eid,_,_ in EXPERIMENTS:
        for seed in (42,17):
            p=load_pack(ROOT/eid/f's{seed}/expanded/evaluation.npz');emp,fair=crps_fields(p['samples'],p['truth']);a=[];b=[]
            for li in range(3):
                keep=p['layout']==li;a.append(float(np.nanmean(np.nanmean(emp[keep],axis=0))));b.append(float(np.nanmean(np.nanmean(fair[keep],axis=0))))
            records.append(dict(experiment=eid,seed=seed,members=len(p['samples']),empirical_crps_mm=float(np.mean(a)),fair_crps_mm=float(np.mean(b)),iid_members_assumed=eid!='E1207_diffusion_xgb'))
    with (ROOT/'ensemble_scores.csv').open('w',newline='') as f:w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
    write_json(ROOT/'ensemble_score_note.json',dict(meaning='Empirical CRPS scores each finite ensemble CDF; fair CRPS is unbiased for an underlying distribution only with iid members. E1207 common correction depends on all16 members, so its fair score is descriptive. E1206-to-E1207 empirical and fair score changes agree because all pair distances within each ensemble are unchanged.'))


def overview():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.read_csv(ROOT/'summary.csv');s=s[s.scope=='expanded'];p=pd.read_csv(ROOT/'ensemble_scores.csv');sp=pd.read_csv(ROOT/'spatial_diagnostics.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5));labels=['MLP','CNN-LSTM','XGBoost','Base+XGB','Base+Gaussian','Base+Diffusion','Diffusion+XGB'];ids=list(s.experiment.unique());x=np.arange(7)
    axes[0].bar(x,[s[s.experiment==eid].macro_rmse_mm.mean() for eid in ids],color=['#64748b']*3+['#0f766e','#0f766e','#2563eb','#b45309'])
    for i,eid in enumerate(ids):axes[0].scatter([i-.08,i+.08],s[s.experiment==eid].macro_rmse_mm,c='black',s=13,zorder=3)
    axes[0].axhline(12.065729,color='black',ls='--',lw=1,label='Height-IDW');axes[0].set_xticks(x,labels,rotation=40,ha='right');axes[0].set(ylabel='Station macro RMSE (mm)',title='Point estimates; dots show two seeds');axes[0].legend(fontsize=8)
    prob=ids[4:];px=np.arange(3)
    for i,eid in enumerate(prob):
        vals=p[p.experiment==eid].empirical_crps_mm;axes[1].bar(i,vals.mean(),color=['#0f766e','#2563eb','#b45309'][i]);axes[1].scatter([i-.06,i+.06],vals,c='black',s=13,zorder=3)
    axes[1].set_xticks(px,['Gaussian','Diffusion','Diffusion+XGB'],rotation=25,ha='right');axes[1].set(ylabel='Empirical CRPS (mm)',title='Same16-member ensemble forecasts')
    for offset,group,color in [(-.18,'0to50km','#0f766e'),(.18,'50to150km','#b45309')]:
        vals=[sp[(sp.experiment==eid)&(sp.group==group)].shuffle_delta_mm.mean() for eid in prob];axes[2].bar(px+offset,vals,.36,label=group,color=color)
    axes[2].axhline(0,color='black',lw=.8);axes[2].set_xticks(px,['Gaussian','Diffusion','Diffusion+XGB'],rotation=25,ha='right');axes[2].set(ylabel='Shuffled minus original CRPS (mm)',title='Positive: useful spatial correspondence');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(ROOT/'final_overview.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    run();overview()
