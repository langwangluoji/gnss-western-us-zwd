from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
S=OUT.parents[1]
INV=S/'01_inventory/pwv_local'
P=S/'02_data/processed'
g=pd.read_csv(INV/'global_station_headers.csv')
r=pd.read_csv(INV/'regional_station_candidates.csv')
files=list((S.parents[1]/'data/pwv_data').glob('*.gz'))
expected={Path(x).name:int(b) for x,b in zip(g.source_path,g.source_bytes)}
actual={x.name:x.stat().st_size for x in files}
summary={'scope':'Fresh raw inventory and source hashes; full processed validity scan. Regional raw completeness CSV is prior audit, not rescanned here.',
 'raw_files':len(files),'raw_bytes':sum(actual.values()),'missing_files':sorted(set(expected)-set(actual)),
 'extra_files':sorted(set(actual)-set(expected)),'size_mismatch':[k for k in expected.keys()&actual.keys() if expected[k]!=actual[k]],
 'global_bounds':{c:[float(g[c].min()),float(g[c].max())] for c in ['lon','lat']},
 'header_time':[str(g.header_start.min()),str(g.header_end.max())], 'regional_candidates':len(r)}
summary['screenshot_box_stations']=int(g.lon.between(-125,-110).mul(g.lat.between(25,50)).sum())
annual=[]; monthly=[]; stationrows=[]; roles={}
with h5py.File(P/'mae_station_archive/station_observations_2000_2021.h5') as f:
    hours=f['hour_id'][:]; dates=pd.to_datetime(hours,unit='h'); ids=[x.decode() if isinstance(x,bytes) else str(x) for x in f['station'][:]]
    mask=np.empty((len(hours),len(ids)),bool)
    for i in range(0,len(hours),4096):mask[i:i+4096]=np.isfinite(f['values'][i:i+4096]).all(2)
    sparse=f['sparse_input'][:].astype(bool)
roles['construction']=(ids,dates,mask)
roles['input39']=([s for s,k in zip(ids,sparse) if k],dates,mask[:,sparse])
for role,name in [('validation42','validation_stations.h5'),('test26','independent_test_2020_2021.h5')]:
    with h5py.File(P/'expanded_2000_2021'/name) as f:
        names=[x.decode() if isinstance(x,bytes) else str(x) for x in f['station'][:]]
        roles[role]=(names,pd.to_datetime(f['hour_id'][:],unit='h'),np.isfinite(f['zwd_mm'][:]))
for role,(names,dt,m) in roles.items():
    meta=g.set_index('station').loc[names]
    summary[role]={'stations':len(names),'bounds':{c:[float(meta[c].min()),float(meta[c].max())] for c in ['lon','lat']},'axis_start':str(dt.min()),'axis_end':str(dt.max()),'duplicate_hours':int(dt.duplicated().sum()),'axis_gaps':int((np.diff(dt.asi8)!=3600*10**9).sum())}
    for y in sorted(set(dt.year)):
        a=m[dt.year==y]; n=len(a)
        annual.append(dict(role=role,year=int(y),axis_hours=n,active_stations=int(a.any(0).sum()),hours_any=int(a.any(1).sum()),hours_all=int(a.all(1).sum()),valid_station_hours=int(a.sum()),coverage=float(a.mean())))
    month_labels=dt.strftime('%Y-%m')
    for ym in sorted(set(month_labels)):
        a=m[month_labels==ym]
        monthly.append(dict(role=role,month=ym,coverage=float(a.mean()),active_stations=int(a.any(0).sum())))
    for j,name in enumerate(names):
        ok=np.flatnonzero(m[:,j]); stationrows.append(dict(role=role,station=name,lon=meta.loc[name,'lon'],lat=meta.loc[name,'lat'],coverage=float(m[:,j].mean()),first=str(dt[ok[0]]) if len(ok) else None,last=str(dt[ok[-1]]) if len(ok) else None))
idx=np.load(S/'04_models/residual_edm_v1/stage1/indices.npz')
summary['splits']={k:int(len(idx[k])) for k in idx.files if k!='domain'}
summary['domain_cells']=int(idx['domain'].sum())
trainyears=pd.Series(dates[idx['train']].year).value_counts().sort_index()
summary['train_complete_by_year']={str(k):int(v) for k,v in trainyears.items()}
# Rehash every source file used by the current construction/validation/test sets.
checks=[]
for name in ['construction','validation','independent_test']:
    t=pd.read_csv(P/f'expanded_2000_2021/{name}_source_manifest.csv')
    for row in t.itertuples():
        digest=hashlib.sha256()
        with open(row.path,'rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
        checks.append(dict(role=name,station=row.station,match=digest.hexdigest()==row.sha256))
summary['source_hashes']={'checked':len(checks),'mismatches':[x for x in checks if not x['match']]}
pd.DataFrame(checks).to_csv(OUT/'source_hash_checks.csv',index=False)
a=pd.DataFrame(annual); a.to_csv(OUT/'annual_coverage.csv',index=False)
pd.DataFrame(monthly).to_csv(OUT/'monthly_coverage.csv',index=False)
st=pd.DataFrame(stationrows); st.to_csv(OUT/'station_coverage.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
ax[0].scatter(g.lon,g.lat,s=2,alpha=.4); ax[0].set(title='Local raw product: 5084 station files',xlabel='Longitude',ylabel='Latitude',xlim=(-180,180),ylim=(-90,90));ax[0].grid(alpha=.2)
for role,color,marker in [('construction','#999999','.'),('input39','#0072b2','o'),('validation42','#e69f00','^'),('test26','#cc79a7','s')]:
    z=st[st.role==role];ax[1].scatter(z.lon,z.lat,s=18 if role!='construction' else 10,c=color,marker=marker,label=role)
ax[1].set(title='Current model station roles (coordinate plot)',xlabel='Longitude',ylabel='Latitude');ax[1].legend();ax[1].grid(alpha=.2)
fig.savefig(OUT/'spatial_distribution.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
for role in roles:
    z=a[a.role==role]; ax[0].plot(z.year,z.active_stations,'o-',label=role);ax[1].plot(z.year,z.coverage*100,'o-',label=role)
ax[0].set(ylabel='Stations with valid data',title='Processed archive coverage (full fixed roster denominator)');ax[0].legend();ax[1].set(ylabel='Valid station-hours (%)',xlabel='Year',ylim=(0,102))
for x in ax:x.grid(alpha=.2)
fig.savefig(OUT/'temporal_coverage.png',dpi=170);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2));print(a[a.year.between(2016,2020)].to_string(index=False))
