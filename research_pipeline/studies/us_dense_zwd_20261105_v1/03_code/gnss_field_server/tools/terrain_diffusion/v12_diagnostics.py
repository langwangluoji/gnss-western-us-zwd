"""Frozen v12 spatial, terrain and calibration diagnostics, independent of training."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .data import Dataset,write_json
from .metrics import evaluate
from .residual_experiment import ROOT,load_pack
from .residual_queue import EXPERIMENTS


def run(root=ROOT):
    root=Path(root);data=Dataset(device='cpu');geo=data.a['val_geo'];dist=cdist(geo[:,:2],geo[:,:2]);ii,jj=np.triu_indices(137,1);spatial=[];terrain=[];calibration=[]
    for eid,_,_ in EXPERIMENTS:
        for seed in (42,17):
            folder=root/eid/f's{seed}/expanded';p=load_pack(folder/'evaluation.npz');s=p['samples'];y=p['truth'];err=s.mean(0)-y
            for label,mask in [('below500m',geo[:,2]<.5),('500to1500m',(geo[:,2]>=.5)&(geo[:,2]<1.5)),('above1500m',geo[:,2]>=1.5)]:
                scores=[]
                for layout in range(3):
                    ix=np.flatnonzero(p['layout']==layout);scores.append(evaluate(s[:,ix][:,:,mask],y[ix][:,mask]))
                terrain.append(dict(experiment=eid,seed=seed,group=label,stations=int(mask.sum()),macro_rmse_mm=float(np.mean([v['station_macro_rmse_mm'] for v in scores]))))
            if len(s)>1:
                for label,keep in [('0to50km',(dist[ii,jj]>0)&(dist[ii,jj]<=50)),('50to150km',(dist[ii,jj]>50)&(dist[ii,jj]<=150))]:
                    left,right=ii[keep],jj[keep];original=[];shuffled=[]
                    for layout in range(3):
                        ix=np.flatnonzero(p['layout']==layout);a=s[:,ix];truth=y[ix];delta=a[:,:,left]-a[:,:,right];dy=truth[:,left]-truth[:,right]
                        original.append(evaluate(delta,dy)['fair_crps_mm'])
                        rng=np.random.default_rng(12190+layout);permuted=a.copy()
                        for time in range(a.shape[1]):
                            for station in range(a.shape[2]):permuted[:,time,station]=a[rng.permutation(len(a)),time,station]
                        np.testing.assert_array_equal(np.sort(permuted,axis=0),np.sort(a,axis=0))
                        shuffled.append(evaluate(permuted[:,:,left]-permuted[:,:,right],dy)['fair_crps_mm'])
                    spatial.append(dict(experiment=eid,seed=seed,group=label,pairs=len(left),fair_crps_mm=float(np.mean(original)),shuffle_delta_mm=float(np.mean(shuffled)-np.mean(original))))
                for level in (.5,.8,.9):
                    lo,hi=np.quantile(s,[(1-level)/2,(1+level)/2],axis=0);valid=np.isfinite(y)
                    calibration.append(dict(experiment=eid,seed=seed,nominal=level,coverage=float(np.mean(((y>=lo)&(y<=hi))[valid])),width_mm=float(np.mean((hi-lo)[valid]))))
            rmse=np.sqrt(np.nanmean(err**2,axis=0));fig,ax=plt.subplots(figsize=(7,5));im=ax.scatter(geo[:,0],geo[:,1],c=rmse,cmap='viridis',s=24);fig.colorbar(im,ax=ax,label='Station RMSE (mm)');ax.set(title=f'{eid} seed{seed}: sampled2018',xlabel='Projected x (km)',ylabel='Projected y (km)',aspect='equal');fig.tight_layout();fig.savefig(folder/'station_rmse_map.png',dpi=150);plt.close(fig)
            station=int(np.nanargmax(np.nanstd(y[:24],axis=0)));fig,ax=plt.subplots(figsize=(8,4));ax.plot(y[:24,station],color='black',label='GNSS');ax.plot(s[:,:24,station].mean(0),label='Ensemble mean')
            if len(s)>1:
                lo,hi=np.quantile(s[:,:24,station],[.05,.95],axis=0);ax.fill_between(np.arange(24),lo,hi,alpha=.2,label='90% sample interval')
            ax.set(title=f'{eid} seed{seed}, first day, station {data.a["val_ids"][station]}',xlabel='Hour',ylabel='ZWD (mm)');ax.legend();fig.tight_layout();fig.savefig(folder/'example_sequence.png',dpi=150);plt.close(fig)
    for name,rows in [('terrain',terrain),('spatial',spatial),('calibration',calibration)]:
        with (root/f'{name}_diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    write_json(root/'diagnostics_receipt.json',dict(status='completed',scope='posthoc frozen expanded predictions; one fixed spatial shuffle, not significance test',terrain_rows=len(terrain),spatial_rows=len(spatial),calibration_rows=len(calibration)))


if __name__=='__main__':run()
