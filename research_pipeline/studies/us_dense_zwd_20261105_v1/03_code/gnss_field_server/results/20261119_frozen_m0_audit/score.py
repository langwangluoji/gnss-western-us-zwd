from pathlib import Path
import numpy as np,json,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnss_field.evaluation.metrics import point_metrics
from gnss_field.evaluation.bootstrap import _blocks
d=Path(__file__).resolve().parent
def load(year):
 files=sorted((d/'monthly').glob(f'{year}-*.npz'));assert len(files)==12
 rows=[dict(np.load(f)) for f in files]
 return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
tr=load(2018);ev=load(2019)
v=tr['mask'];e=tr['truth']-tr['mean'];weight=np.broadcast_to(1/np.maximum(v.sum(0),1),v.shape)[v];res=e[v];order=np.argsort(res);ws=np.cumsum(weight[order]);qs=np.interp((np.arange(64)+.5)/64*ws[-1],ws,res[order]);assert np.isfinite(qs).all()
y=ev['truth'];ok=ev['mask'];hours=ev['hours']
assert len(hours)==8760 and ok.sum()==642258
names=['coarse','idw','mean','innovation']
metrics={n:point_metrics(y,ev[n],ok) for n in names}
for n,ref in [('coarse',13.364),('idw',10.279),('mean',10.345)]:assert round(metrics[n]['station_macro_rmse_mm'],3)==ref
delta=y-ev['mean'];crps=np.zeros_like(delta,dtype=float)
for q in qs: crps+=abs(delta-q)/len(qs)
crps-=np.abs(qs[:,None]-qs).mean()/2
def macro(z):return float(np.nanmean(np.divide(np.where(ok,z,0).sum(0),ok.sum(0),out=np.full(ok.shape[1],np.nan),where=ok.sum(0)>0)))
prob={'CRPS_mm':macro(crps),'members':qs.tolist(),'training_stations':int((v.sum(0)>0).sum()),'levels':{}}
for level in (.8,.9,.95):
 lo,hi=np.quantile(qs,[(1-level)/2,(1+level)/2]);prob['levels'][level]={'coverage':macro((delta>=lo)&(delta<=hi)),'width_mm':float(hi-lo)}
cis={}
for days in (7,14):
 blocks=_blocks(hours,days);rng=np.random.default_rng(20261117);active=ok.any(0)
 count=np.stack([ok[ix].sum(0)[active] for ix in blocks])
 sums={n:np.stack([np.where(ok[ix],(ev[n][ix].astype(float)-y[ix])**2,0).sum(0)[active] for ix in blocks]) for n in names}
 scores={n:[] for n in names}
 while len(scores['mean'])<2000:
  mult=np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks));den=mult@count
  if not den.all():continue
  for n in names:scores[n].append(np.sqrt((mult@sums[n])/den).mean())
 cis[days]={f'{a}-{b}':np.quantile(np.array(scores[a])-scores[b],[.025,.975]).tolist() for a,b in [('mean','idw'),('mean','coarse'),('innovation','mean'),('innovation','idw')]}
result=dict(primary=metrics,empirical=prob,paired_ci95=cis,gate='STOP_NO_MATCHING_DECODER')
(d/'baseline_results.json').write_text(json.dumps(result,indent=2))
stations=pd.read_csv('../../02_data/processed/gnss_field_server_v1/stations.csv').iloc[json.loads((d/'station_prediction_manifest.json').read_text())['year2019_station_ids']]
def bystation(n):return np.sqrt(np.where(ok,(ev[n]-y)**2,0).sum(0)/np.maximum(ok.sum(0),1))
fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar(names,[metrics[n]['station_macro_rmse_mm'] for n in names]);ax[0].set_ylabel('Station-macro RMSE (mm)');keep=ok.any(0);vmap=(bystation('innovation')-bystation('mean'))[keep];lim=max(abs(vmap));sc=ax[1].scatter(stations.x.to_numpy()[keep]/1000,stations.y.to_numpy()[keep]/1000,c=vmap,vmin=-lim,vmax=lim,cmap='RdBu_r');ax[1].set(xlabel='Lambert x (km)',ylabel='Lambert y (km)',aspect='equal');fig.colorbar(sc,label='Innovation IDW minus M0 RMSE (mm)');fig.tight_layout();fig.savefig(d/'station_skill.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(9,4));lev=list(prob['levels']);ax[0].plot([.75,1],[.75,1],'k--');ax[0].plot(lev,[prob['levels'][x]['coverage'] for x in lev],'o-');ax[0].set(xlabel='Nominal coverage',ylabel='Observed station-macro coverage');ax[1].bar([str(x) for x in lev],[prob['levels'][x]['width_mm'] for x in lev]);ax[1].set(xlabel='Nominal coverage',ylabel='Interval width (mm)');fig.tight_layout();fig.savefig(d/'marginal_calibration.png',dpi=160);plt.close(fig)
print(json.dumps({n:v['station_macro_rmse_mm'] for n,v in metrics.items()}),prob['CRPS_mm'],prob['levels'],cis,flush=True)

