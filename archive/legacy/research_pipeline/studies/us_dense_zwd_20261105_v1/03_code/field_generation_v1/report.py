"""Locked baselines, field scale diagnostics and paired evaluation."""
import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from field_generation_v1.prepare import ROOT
from field_generation_v1.geometry import query_grid
from field_generation_v1.model import MeanField,ResidualField
from residual_edm_v1.data import write_json

def grouped_metrics(records,selected):
    a=records[selected]
    if not len(a):return dict(stations=0,station_hours=0,station_macro_rmse_mm=None)
    e=a[:,2]-a[:,3];stations,inverse=np.unique(a[:,1],return_inverse=True)
    counts=np.bincount(inverse);sse=np.bincount(inverse,weights=e*e)
    return dict(stations=len(stations),station_hours=len(a),station_macro_rmse_mm=float(np.mean(np.sqrt(sse/counts))),
        rmse_mm=float(np.sqrt(np.mean(e*e))),mae_mm=float(np.mean(abs(e))),bias_mm=float(np.mean(e)))

def paired_scale_interval(original,coarsened,hours):
    """Synchronized seven-day resampling of pair-hour squared errors."""
    days=np.asarray(hours,dtype=int)//24;days-=days.min();nd=int(days.max()+1)
    counts=np.zeros(nd);sa=counts.copy();sb=counts.copy()
    valid=np.isfinite(original)&np.isfinite(coarsened)
    np.add.at(counts,days,valid.sum(1))
    np.add.at(sa,days,np.where(valid,original*original,0).sum(1))
    np.add.at(sb,days,np.where(valid,coarsened*coarsened,0).sum(1))
    rng=np.random.default_rng(20261117);values=[]
    for _ in range(2000):
        start=rng.integers(0,nd,size=int(np.ceil(nd/7)));take=((start[:,None]+np.arange(7))%nd).ravel()[:nd]
        n=counts[take].sum()
        if n:values.append(float(np.sqrt(sa[take].sum()/n)-np.sqrt(sb[take].sum()/n)))
    return dict(replicates=len(values),delta_original_minus_100km_95ci=np.quantile(values,[.025,.975]).tolist() if values else None)

def support_diagnostics(data,records,year,station_ids):
    from scipy.spatial import ConvexHull
    hours=np.flatnonzero(data.dates.year==year);memo={};land=data.g['region'];land_cells=int(land.sum())
    possible=0;supported_land_hours=0;near=np.empty(len(records));density=near.copy();boundary=near.copy()
    unique,starts,counts=np.unique(records[:,0].astype(int),return_index=True,return_counts=True)
    slices={int(t):slice(int(s),int(s+c)) for t,s,c in zip(unique,starts,counts)}
    for t in hours:
        key=data.available[t].tobytes()
        if key not in memo:
            _,_,_,n,c=data.geometry_for(key,'stations');xy=data.xy[data.inputs[data.available[t]]]
            edge=np.full(len(data.roles),np.nan)
            if len(xy)>=3:
                from scipy.spatial import QhullError
                try:
                    eq=ConvexHull(xy).equations
                    edge=-np.max(data.xy@eq[:,:2].T+eq[:,2],axis=1)
                except QhullError:pass
            memo[key]=(n,c,edge,int(data.support(int(t))[0].sum()))
        n,c,edge,area=memo[key];supported_land_hours+=area
        possible+=int(np.isfinite(data.values[t,station_ids,0]).sum())
        if int(t) in slices:
            sl=slices[int(t)];ids=records[sl,1].astype(int);near[sl]=n[ids]/1000;density[sl]=c[ids];boundary[sl]=edge[ids]/1000
    groups={}
    for label,values,bins in [('input_distance_km',near,[(0,25),(25,50),(50,75),(75,101)]),
                              ('input_count_150km',density,[(0,3),(3,5),(5,10),(10,float('inf'))]),
                              ('input_hull_boundary_km',boundary,[(-float('inf'),25),(25,50),(50,100),(100,float('inf'))])]:
        groups[label]={f'{lo}-{hi}':grouped_metrics(records,(values>=lo)&(values<hi)) for lo,hi in bins}
    return dict(groups=groups,coverage=dict(role_stations=len(station_ids),scored_stations=len(np.unique(records[:,1])),
        valid_observation_station_hours=possible,scored_station_hours=len(records),station_hour_fraction=len(records)/possible if possible else None,
        land_output_cells=land_cells,land_output_cell_hours=land_cells*len(hours),supported_land_cell_hours=supported_land_hours,
        supported_land_fraction=supported_land_hours/(land_cells*len(hours)),cell_area_km2=625,
        area_note='Projected grid-cell coverage; station error is not area-average error.'))

def export_events(predictor,year,destination,name):
    """Fixed calendar events, chosen without observations or model errors."""
    data=predictor.data
    for month in (1,4,7,10):
        ids=np.flatnonzero((data.dates.year==year)&(data.dates.month==month)&(data.dates.day==15)&(data.dates.hour==12))
        if not len(ids):continue
        t=int(ids[0]);path=destination/f'field_{name}_{year}_{month:02d}15T12.npz'
        if path.exists():continue
        with torch.no_grad():
            field,features,offset=predictor.predict_field(t)
            np.savez_compressed(path,z0_mm=field.cpu().numpy(),features=features.cpu().numpy() if features is not None else np.empty(0),
                land=data.g['region'],support=data.support(t)[0],origin=data.g['origin'],spacing_m=25000,hour_id=data.hours[t],offset=offset)
            saved=np.load(path);reloaded=torch.as_tensor(saved['z0_mm'],device=data.device)
            station_ids=data.dev if year==2019 else data.retro
            expected,_=predictor.query(t,field,None,offset,station_ids);actual,_=predictor.query(t,reloaded,None,offset,station_ids)
            torch.testing.assert_close(actual,expected,rtol=0,atol=.001)

def diagnostics(data,path):
    a=np.load(path)['records'];stations=np.unique(a[:,1].astype(int));hours=np.unique(a[:,0].astype(int))
    n=len(stations);pred=np.full((len(hours),n),np.nan);truth=pred.copy();low=pred.copy()
    ti=np.searchsorted(hours,a[:,0]);si=np.searchsorted(stations,a[:,1]);pred[ti,si]=a[:,2];truth[ti,si]=a[:,3];low[ti,si]=a[:,5]
    error=pred-truth
    def macro(e):return float(np.nanmean(np.sqrt(np.nanmean(e*e,axis=0))))
    result=dict(original_macro_rmse=macro(error),coarsened_macro_rmse=macro(low-truth),distance_bins={})
    distance=cdist(data.xy[stations],data.xy[stations])/1000;factor=np.exp(data.height[stations]/3000)
    for left,right in [(0,25),(25,50),(50,100),(100,200)]:
        i,j=np.where(np.triu((distance>left)&(distance<=right),1));rows={}
        for kind,e,le in [('z0',(pred-truth)*factor,(low-truth)*factor),('surface_height100',error,low-truth)]:
            good=np.ones(len(i),bool) if kind=='z0' else np.abs(data.height[stations[i]]-data.height[stations[j]])<=100
            diff=e[:,i[good]]-e[:,j[good]];ldiff=le[:,i[good]]-le[:,j[good]];valid=np.isfinite(diff)
            rows[kind]=dict(pairs=int(good.sum()),pair_hours=int(valid.sum()),rmse_mm=float(np.sqrt(np.nanmean(diff*diff))) if valid.any() else None)
            rows[kind]['coarsened_rmse_mm']=float(np.sqrt(np.nanmean(ldiff*ldiff))) if valid.any() else None
            if valid.any():rows[kind]['paired_scale_bootstrap']=paired_scale_interval(diff,ldiff,hours)
        result['distance_bins'][f'{left}-{right}']=rows
    result['cohorts']={str(c):dict(stations=int((data.roles.iloc[stations].cohort.to_numpy()==c).sum()),macro_rmse=macro(error[:,data.roles.iloc[stations].cohort.to_numpy()==c])) for c in data.roles.iloc[stations].cohort.unique()}
    result['seasonal']={str(season):macro(error[((data.dates[hours].month%12)//3)==season]) for season in range(4)}
    result['height_bins']={f'{lo}-{hi}':macro(error[:,(data.height[stations]>=lo)&(data.height[stations]<hi)]) for lo,hi in [(-500,500),(500,1500),(1500,10000)] if ((data.height[stations]>=lo)&(data.height[stations]<hi)).any()}
    return result

def paired_bootstrap(first,second,data,coarsened=False):
    a=np.load(first)['records'];b=np.load(second)['records']
    if coarsened:b=b.copy();b[:,2]=b[:,5]
    np.testing.assert_array_equal(a[:,:2],b[:,:2])
    stations=np.unique(a[:,1].astype(int));si=np.searchsorted(stations,a[:,1]);days=a[:,0].astype(int)//24;days-=days.min();nd=int(days.max()+1)
    count=np.zeros((nd,len(stations)));sa=count.copy();sb=count.copy()
    np.add.at(count,(days,si),1);np.add.at(sa,(days,si),(a[:,2]-a[:,3])**2);np.add.at(sb,(days,si),(b[:,2]-b[:,3])**2)
    rng=np.random.default_rng(20261117);values=[]
    for _ in range(2000):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/7)));take=((starts[:,None]+np.arange(7))%nd).ravel()[:nd]
        n=count[take].sum(0);ok=n>0
        values.append(float(np.mean(np.sqrt(sa[take].sum(0)[ok]/n[ok])-np.sqrt(sb[take].sum(0)[ok]/n[ok]))))
    frame=data.roles.iloc[stations];dense=frame.cohort.to_numpy()=='dense'
    keys=np.floor(data.xy[stations]/50000).astype(int)
    groups=[np.flatnonzero(dense&(keys==key).all(1)) for key in np.unique(keys[dense],axis=0)]
    groups=[g for g in groups if len(g)]
    grouped=[]
    if groups:
        station_counts=count.sum(0);delta=np.sqrt(sa.sum(0)/np.maximum(station_counts,1))-np.sqrt(sb.sum(0)/np.maximum(station_counts,1))
        for _ in range(2000):
            selected=rng.integers(0,len(groups),size=len(groups));ids=np.concatenate([groups[k] for k in selected]);grouped.append(float(delta[ids].mean()))
    return dict(replicates=2000,days_per_block=7,delta_macro_rmse_95ci=np.quantile(values,[.025,.975]).tolist(),
        dense_group_count=len(groups),dense_group_resample_95ci=np.quantile(grouped,[.025,.975]).tolist() if grouped else None)

def run(data,base,base_hash,scales):
    from field_generation_v1.training import Predictor,evaluate,status,check_pause
    class Baseline(Predictor):
        def __init__(self,kind):super().__init__(data,torch.nn.Identity());self.kind=kind
        def predict_field(self,t,station=None):
            if self.kind=='coarse':
                raw,_=data.coarse(t);v=torch.as_tensor(raw[0:1],device=data.device)
                field=torch.nn.functional.interpolate(v[None],size=data.shape,mode='bilinear',align_corners=False)[0]
            else:
                ids,w,_,_,_=data.geometry_for(data.available[t].tobytes(),'grid')
                values=np.nan_to_num(data.input_values[t,:,0]).copy()
                if self.kind=='innovation':
                    full,_,_=Predictor(data,base).predict_field(t)
                    xy=torch.as_tensor(data.qxy[data.inputs],device=data.device,dtype=torch.float32)
                    values-=query_grid(full,xy)[:,0].cpu().numpy()
                a=(values[ids]*w).sum(1).reshape(data.shape)
                field=torch.as_tensor(a[None],device=data.device,dtype=torch.float32)
                if self.kind=='innovation':field=full+field
            return field,None,np.array([0,0])
    destination=ROOT/'evaluation';destination.mkdir(exist_ok=True)
    models={'coarse':Baseline('coarse'),'IDW':Baseline('idw'),'M0':Predictor(data,base),'innovation_IDW':Baseline('innovation')}
    for arm in ['M2-F','M2-J']:
        chosen=json.loads((ROOT/arm/'selected.json').read_text());m=ResidualField().to(data.device)
        m.load_state_dict(torch.load(chosen['path'],map_location=data.device,weights_only=False)['ema']);m.eval()
        p=Predictor(data,m,base,scales['z0'],base_hash);p.faithful=arm=='M2-F';p.surface_scale=scales['surface'];models[arm]=p
    for year,station_ids in [(2019,data.dev),(2020,data.retro)]:
        hours=np.flatnonzero(data.dates.year==year)
        for name,predictor in models.items():
            check_pause();target=destination/f'{name}_{year}.json';archive=destination/f'{name}_{year}.npz'
            if target.exists():continue
            status(phase='locked_evaluation',arm=name,year=year)
            metrics=evaluate(predictor,hours,station_ids,archive)
            metrics['diagnostics']=diagnostics(data,archive);metrics['scope']=f'{year}_'+('development' if year==2019 else 'retrospective')
            metrics.update(support_diagnostics(data,np.load(archive)['records'],year,station_ids))
            metrics['paired_100km_comparison']=paired_bootstrap(archive,archive,data,coarsened=True)
            export_events(predictor,year,destination,name)
            write_json(target,metrics);print('LOCKED EVALUATION',name,year,metrics['station_macro_rmse_mm'],flush=True)
        for arm in ['M2-F','M2-J']:
            write_json(destination/f'{arm}_vs_M0_{year}.json',paired_bootstrap(destination/f'{arm}_{year}.npz',destination/f'M0_{year}.npz',data))
