from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gnss_field.data import PackedStore
d=Path(__file__).resolve().parent
s=PackedStore('../../02_data/processed/gnss_field_server_v1',purpose='development');a=s.ids('A');c=s.ids('C')
res=json.loads((d/'baseline_results.json').read_text());q=np.array(res['empirical']['members'])
files=sorted((d/'monthly').glob('2019-*.npz'));rows=[dict(np.load(p)) for p in files]
def get(k):return np.concatenate([r[k] for r in rows])
hours=get('hours');truth=get('truth');mean=get('mean');mask=get('mask')
val=s.context_values(hours);valid=np.isfinite(val).all(-1);dist=cdist(s.xy[c],s.xy[a])/1000
nearest=np.stack([dist[:,v].min(1) if v.any() else np.full(len(c),np.inf) for v in valid]);z=np.where(valid,val[:,:,0],np.nan)
training=np.asarray(s.hours);training=training[pd.to_datetime(training,unit='h').year==2018];v=s.context_values(training);ztr=np.where(np.isfinite(v).all(-1),v[:,:,0],np.nan)
threshold=np.nanquantile(np.nanmedian(abs(ztr[3:]-ztr[:-3]),axis=1),.95)
change=np.full(len(z),np.nan);change[3:]=np.nanmedian(abs(z[3:]-z[:-3]),axis=1)
groups={'All':mask,'Nearest <=50 km':mask&(nearest<=50),'Nearest >50 km':mask&(nearest>50),'Rapid A change':mask&(change>threshold)[:,None],'Other weather':mask&(change<=threshold)[:,None]}
delta=truth-mean;crps=sum(abs(delta-x)/len(q) for x in q)-np.abs(q[:,None]-q).mean()/2
result={}
for key,g in groups.items():
 counts=g.sum(0);active=counts>0
 def macro(x):return float((np.where(g,x,0).sum(0)[active]/counts[active]).mean())
 entry={'n_stations':int(active.sum()),'n_station_hours':int(g.sum()),'crps_mm':macro(crps),'coverage':[],'width_mm':[]}
 for level in (.8,.9,.95):
  lo,hi=np.quantile(q,[(1-level)/2,(1+level)/2]);entry['coverage'].append(macro((delta>=lo)&(delta<=hi)));entry['width_mm'].append(hi-lo)
 result[key]=entry
(d/'calibration_groups.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(1,2,figsize=(11,4.5));ax[0].plot([.65,1],[.65,1],'k--')
for key,r in result.items():ax[0].plot([.8,.9,.95],r['coverage'],'o-',label=key)
ax[0].set(xlabel='Nominal coverage',ylabel='Observed coverage');ax[0].legend(fontsize=8);ax[1].barh(list(result),[r['crps_mm'] for r in result.values()]);ax[1].set_xlabel('Station-macro CRPS (mm)');fig.tight_layout();fig.savefig(d/'calibration_by_condition.png',dpi=160)
