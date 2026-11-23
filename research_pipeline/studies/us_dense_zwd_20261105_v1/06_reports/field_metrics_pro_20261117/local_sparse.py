"""A-only local input-thinning probe; B labels never fit interpolator parameters."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from local_pilot import OUT,interpolate,query,covariance_fit,metrics,pair_metrics

def main():
    sys.path.insert(0,str(OUT.parents[1]/'03_code'))
    from field_generation_v1.data import FieldData
    d=FieldData('cpu');manifest=json.loads((OUT/'local_pilot_manifest.json').read_text());rng=np.random.default_rng(20261118)
    plans=[]
    for region in manifest['regions']:
        case=next(c for c in manifest['cases'] if c['region']['name']==region['name'])
        sources=np.array([s for s in case['sources'] if s in d.inputs]);rng.shuffle(sources)
        y,x=region['y'],region['x'];center=d.g['origin']+np.array([x+12,y+12])*25000
        single=int(sources[np.argmin(np.linalg.norm(d.xy[sources]-center,axis=1))])
        targets=d.train[(d.qxy[d.train,0]>=x)&(d.qxy[d.train,0]<=x+23)&(d.qxy[d.train,1]>=y)&(d.qxy[d.train,1]<=y+23)]
        for level in ['100%','50%','25%','one','zero']:
            keep=sources if level=='100%' else sources[:int(np.ceil(len(sources)*(0.5 if level=='50%' else .25)))] if level in ['50%','25%'] else np.array([single]) if level=='one' else np.array([],int)
            assert not np.intersect1d(keep,targets).size
            plans.append(dict(region=region,level=level,sources=keep.tolist(),targets=targets.tolist(),events=case['events'],all_local_a=sources.tolist()))
    (OUT/'local_sparse_manifest.json').write_text(json.dumps(dict(seed=20261118,optimizer_updates=0,source_role='A only',target_role='B training-period physical stations',retained='Local A fractions, one static source per domain, zero-source A-training climatology fallback',plans=plans),indent=2),encoding='utf-8')
    records=[]
    for plan in plans:
        region=plan['region'];y,x=region['y'],region['x'];origin=d.g['origin']+np.array([x,y])*25000
        yy,xx=np.meshgrid(np.arange(24),np.arange(24),indexing='ij');grid=np.c_[xx.ravel(),yy.ravel()]*25+origin/1000+12.5
        targets=np.array(plan['targets']);xy=d.qxy[targets]-np.array([x,y]);height=d.height[targets];sources=np.array(plan['sources'],int)
        local_a=np.array(plan['all_local_a']);climate=float(np.nanmean(d.values[:d.train_stop,local_a,0]))
        fit=covariance_fit(d,sources,[e['t'] for e in plan['events']]) if len(sources)>=4 else None
        for e in plan['events']:
            t=e['t'];truth=np.array(d.values[t,targets,0])*np.exp(-height/3000)
            values=np.array(d.values[t,sources,0]);ok=np.isfinite(values);ids=sources[ok];values=values[ok]
            distance=cdist(d.xy[targets]/1000,d.xy[ids]/1000) if len(ids) else None
            near=distance.min(1) if len(ids) else np.full(len(targets),np.inf);count=(distance<=150).sum(1) if len(ids) else np.zeros(len(targets),int)
            for method in ['IDW','kriging']:
                fallback=len(ids)<4
                if not len(ids):field=np.full((24,24),climate)
                elif len(ids)==1:field=np.full((24,24),values[0])
                elif fit is None or len(ids)<4:field=interpolate(d.xy[ids]/1000,values,grid,100,.05,'IDW').reshape(24,24)
                else:field=interpolate(d.xy[ids]/1000,values,grid,fit['range_km'],fit['nugget_ratio'],method).reshape(24,24)
                prediction=query(field,xy)*np.exp(-height/3000)
                assert np.isfinite(prediction).all()
                for i,s in enumerate(targets):
                    if not np.isfinite(truth[i]):continue
                    records.append(dict(region=region['name'],level=plan['level'],method=method,fold=0,time_index=t,station=int(s),prediction_mm=float(prediction[i]),target_mm=float(truth[i]),
                        visible_sources=len(ids),nearest_km=float(near[i]),sources150km=int(count[i]),distance_density_support=bool(near[i]<=100 and count[i]>=3),explicit_fallback=fallback))
    frame=pd.DataFrame(records);frame.to_csv(OUT/'local_sparse_predictions.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for keys,f in frame.groupby(['region','level','method']):
        pair,count=pair_metrics(f,d.xy,d.height)
        summaries.append(dict(region=keys[0],level=keys[1],method=keys[2],**metrics(f),difference_0_50km_rmse_mm=pair,pair_hours=count,
            distance_density_support_fraction=float(f.distance_density_support.mean()),fallback_fraction=float(f.explicit_fallback.mean()),
            min_visible_sources=int(f.visible_sources.min()),max_visible_sources=int(f.visible_sources.max())))
    result=dict(optimizer_updates=0,training_years_only=True,development_or_retrospective_labels_used=False,summary=summaries,
        limitations=['Two regions,12 events each,one fixed nested thinning pattern; no robust yearly generalization claim','Only reference baselines tested; neural model thinning remains untested','Distance-density support omits hull and training-label-distance conditions and is explicitly named','Zero/one cases are declared climate/constant fallbacks, not native kriging solutions'])
    (OUT/'local_sparse_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(summaries).to_csv(OUT/'local_sparse_summary.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(summaries)[['region','level','method','station_macro_rmse_mm','distance_density_support_fraction','min_visible_sources','max_visible_sources']].to_string(index=False))

if __name__=='__main__':main()
