"""Bounded, training-period reference CV. Zero neural-network optimizer updates."""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

OUT=Path(__file__).parent

def select_rapid_events(surface):
    # Use the same valid station intersection at both ends: no composition jumps.
    change=np.full(len(surface),np.nan)
    current=surface[3:];previous=surface[:-3];valid=np.isfinite(current)&np.isfinite(previous)
    eligible=valid.sum(1)>=4
    increments=np.where(valid,current-previous,np.nan)
    change[3:][eligible]=np.abs(np.nanmedian(increments[eligible],axis=1))
    change[:18]=np.nan;rapid=[]
    for t in np.argsort(np.nan_to_num(change,nan=-1))[::-1]:
        if len(rapid)==6:break
        if t>=18 and np.isfinite(change[t]) and all(abs(int(t)-v)>72 for v in rapid):rapid.append(int(t))
    return sorted(rapid)

def query(field,xy):
    return map_coordinates(field,[xy[:,1],xy[:,0]],order=1,mode='nearest',prefilter=False)

def lowpass(field):
    h,w=field.shape
    coarse=field.reshape(h//4,4,w//4,4).mean((1,3))
    yy,xx=np.meshgrid((np.arange(h)+.5)/4-.5,(np.arange(w)+.5)/4-.5,indexing='ij')
    return map_coordinates(coarse,[yy,xx],order=1,mode='nearest',prefilter=False)

def interpolate(source_xy,values,target_xy,range_km,nugget_ratio,method):
    distance=cdist(target_xy,source_xy)
    if method=='IDW':
        idx=np.argsort(distance,axis=1)[:,:min(4,len(values))]
        dist=np.take_along_axis(distance,idx,axis=1)
        weight=np.where(dist<=150,1/np.maximum(dist,.001)**2,0)
        empty=weight.sum(1)==0;weight[empty,0]=1
        return (values[idx]*weight).sum(1)/weight.sum(1)
    # A plug-in planar trend plus ordinary kriging of residuals; no held-out fit.
    center=source_xy.mean(0);design=np.c_[np.ones(len(values)),(source_xy-center)/100]
    coef=np.linalg.lstsq(design,values,rcond=None)[0];residual=values-design@coef
    c=(1-nugget_ratio)*np.exp(-cdist(source_xy,source_xy)/range_km)
    c.flat[::len(values)+1]+=nugget_ratio+1e-8
    factor=cho_factor(c,lower=True,check_finite=False)
    cross=(1-nugget_ratio)*np.exp(-distance.T/range_km)
    weights=cho_solve(factor,cross,check_finite=False)
    one=cho_solve(factor,np.ones(len(values)),check_finite=False)
    weights+=one[:,None]*(1-weights.sum(0))/one.sum()
    return np.c_[np.ones(len(target_xy)),(target_xy-center)/100]@coef+residual@weights

def covariance_fit(data,source_ids,hours):
    xy=data.xy[source_ids]/1000;distance=cdist(xy,xy)
    squared=np.zeros_like(distance);count=np.zeros_like(distance)
    for t in hours:
        y=np.array(data.values[t,source_ids,0],dtype=float);valid=np.isfinite(y)
        if valid.sum()<5:continue
        design=np.c_[np.ones(valid.sum()),(xy[valid]-xy[valid].mean(0))/100]
        r=y[valid]-design@np.linalg.lstsq(design,y[valid],rcond=None)[0]
        r/=max(np.std(r),1e-6);i=np.flatnonzero(valid)
        squared[np.ix_(i,i)]+=.5*(r[:,None]-r[None,:])**2
        count[np.ix_(i,i)]+=1
    hs=[];gamma=[];pairs=[]
    for lo,hi in [(0,25),(25,50),(50,100),(100,150),(150,250),(250,400)]:
        use=np.triu((distance>lo)&(distance<=hi)&(count>0),1)
        if use.sum()<3:continue
        hs.append(distance[use].mean());gamma.append((squared[use]/count[use]).mean());pairs.append(int(use.sum()))
    if len(hs)<3:raise ValueError('Insufficient within-fold pairs for covariance fit')
    hs=np.array(hs);gamma=np.array(gamma)
    fit=least_squares(lambda v:v[1]+(1-v[1])*(1-np.exp(-hs/v[0]))-gamma,[100,.1],bounds=([5,.01],[500,.8]))
    return dict(range_km=float(fit.x[0]),nugget_ratio=float(fit.x[1]),distance_centers_km=hs.tolist(),empirical_normalized_semivariogram=gamma.tolist(),pair_counts=pairs,fit_success=bool(fit.success))

def metrics(frame,pred='prediction_mm',truth='target_mm'):
    e=frame[pred]-frame[truth]
    per=(e*e).groupby(frame.station).mean()
    return dict(stations=len(per),station_hours=len(frame),station_macro_rmse_mm=float(np.sqrt(per).mean()),
        micro_rmse_mm=float(np.sqrt(np.mean(e*e))),mae_mm=float(np.mean(abs(e))),bias_mm=float(np.mean(e)))

def pair_metrics(frame,xy,height):
    sse=count=0
    # Both members must be hidden together and predicted by the same fitted field.
    for _,hour in frame.groupby(['fold','time_index']):
        ids=hour.station.to_numpy(int);dist=cdist(xy[ids],xy[ids])/1000;i,j=np.where(np.triu((dist>0)&(dist<=50),1))
        error=(hour.prediction_mm.to_numpy()-hour.target_mm.to_numpy())*np.exp(height[ids]/3000)
        sse+=float(((error[i]-error[j])**2).sum());count+=len(i)
    return (float(np.sqrt(sse/count)) if count else None),count

def main():
    sys.path.insert(0,str(OUT.parents[1]/'03_code'))
    import pandas as pd
    from field_generation_v1.data import FieldData
    data=FieldData('cpu');rng=np.random.default_rng(20261117);started=time.time()
    groups=np.floor(data.xy[data.train]/50000).astype(int)
    unique=np.unique(groups,axis=0);rng.shuffle(unique)
    group_folds={tuple(g):i%3 for i,g in enumerate(unique)}
    folds={int(s):group_folds[tuple(g)] for s,g in zip(data.train,groups)}
    candidates=[]
    for patch,(y,x) in enumerate(data.origins):
        inside=(data.qxy[data.train,0]>=x)&(data.qxy[data.train,0]<=x+23)&(data.qxy[data.train,1]>=y)&(data.qxy[data.train,1]<=y+23)
        terrain=data.g['height'][y:y+24,x:x+24]
        candidates.append(dict(patch=patch,y=int(y),x=int(x),count=int(inside.sum()),roughness_proxy=float(np.std(terrain))))
    first=max(candidates,key=lambda c:(c['count'],-c['patch']))
    second=max([c for c in candidates if c['count']>=25 and np.hypot(c['x']-first['x'],c['y']-first['y'])>=26],key=lambda c:(c['roughness_proxy'],-c['patch']))
    regions=[dict(first,name='dense_training_geometry'),dict(second,name='terrain_proxy_training_geometry')]
    planned=[]
    for region in regions:
        y,x=region['y'],region['x'];center=data.g['origin']+np.array([x+12,y+12])*25000
        pool=np.r_[data.inputs,data.train];near=np.linalg.norm(data.xy[pool]-center,axis=1)
        pool=pool[near<=600000]
        pool_a=np.intersect1d(pool,data.inputs);pool_b=np.intersect1d(pool,data.train)
        pool_b=pool_b[np.argsort(np.linalg.norm(data.xy[pool_b]-center,axis=1))[:192-len(pool_a)]]
        pool=np.r_[pool_a,pool_b]
        local_a=np.intersect1d(pool,data.inputs)
        if len(local_a)<3:raise ValueError('Insufficient local input stations')
        surface=np.array(data.values[:data.train_stop,local_a,0])*np.exp(-data.height[local_a]/3000)
        rapid=select_rapid_events(surface)
        calendar=[]
        for year in (2016,2017,2018):
            for month in (1,7):calendar.append(int(np.flatnonzero((data.dates.year==year)&(data.dates.month==month)&(data.dates.day==15)&(data.dates.hour==12))[0]))
        events=[dict(t=t,type='calendar') for t in calendar]+[dict(t=t,type='rapid_input_A_proxy') for t in sorted(rapid)]
        interior=data.train[(data.qxy[data.train,0]>=x)&(data.qxy[data.train,0]<=x+23)&(data.qxy[data.train,1]>=y)&(data.qxy[data.train,1]<=y+23)]
        for fold in range(3):
            targets=[int(s) for s in interior if folds[int(s)]==fold]
            if not targets:raise ValueError('Empty spatial fold')
            fold_pool=[int(s) for s in pool if s in data.inputs or folds[int(s)]!=fold]
            b_groups=sorted({tuple(np.floor(data.xy[s]/50000).astype(int)) for s in fold_pool if s in data.train})
            rng.shuffle(b_groups);keep=set(b_groups[:int(np.ceil(len(b_groups)/2))])
            for fraction in (1.,.5):
                sources=[s for s in fold_pool if fraction==1 or s in data.inputs or tuple(np.floor(data.xy[s]/50000).astype(int)) in keep]
                assert not set(sources)&set(targets)
                assert all(s in data.inputs or folds[s]!=fold for s in sources)
                planned.append(dict(region=region,fold=fold,b_fraction=fraction,sources=sources,targets=targets,events=events))
    manifest=dict(seed=20261117,data_contract=data.contract,optimizer_updates=0,years=[2016,2017,2018],development_used=False,retrospective_used=False,
        folds=3,station_fold=folds,regions=regions,cases=planned,source_cap=192,
        info_condition='Reference construction A plus trainingB excluding entire hidden50km groups; B density changes only, A fixed. Not comparable to A-only neural model skill.',
        interpolation='IDW nearest4,p2,radius150km with flagged nearest fallback; plug-in plane+ordinary residual kriging exponential covariance fitted within each source fold/density.',
        event_note='Six fixed calendar hours plus six inputA median3h-change maxima separated72h; rapid proxy is not an independent weather classification.',
        spatial_note='24x24 patch25km, grid query before station-height restore. Terrain proxy uses25km aggregatedDEM, not raw-DEM slope. No statistical significance claimed.')
    (OUT/'local_pilot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records=[];covariances=[]
    for index,case in enumerate(planned):
        region=case['region'];y,x=region['y'],region['x'];origin=data.g['origin']+np.array([x,y])*25000
        yy,xx=np.meshgrid(np.arange(24),np.arange(24),indexing='ij');grid=np.c_[xx.ravel(),yy.ravel()]*25+(origin/1000+12.5)
        source_ids=np.array(case['sources']);targets=np.array(case['targets']);xy=data.qxy[targets]-np.array([x,y]);height=data.height[targets]
        fit=covariance_fit(data,source_ids,[e['t'] for e in case['events']]);covariances.append(dict(region=region['name'],fold=case['fold'],b_fraction=case['b_fraction'],**fit))
        train_b=np.intersect1d(source_ids,data.train)
        training_surface=np.array(data.values[:data.train_stop,train_b,0])*np.exp(-data.height[train_b]/3000)
        wet_threshold=float(np.nanquantile(training_surface,.95))
        for event in case['events']:
            predictions={};geometries={}
            for t in [event['t'],event['t']-3]:
                values=np.array(data.values[t,source_ids,0],float);valid=np.isfinite(values);ids=source_ids[valid]
                assert not np.intersect1d(ids,targets).size
                if len(ids)<4:raise ValueError('Too few sources')
                distance=cdist(data.xy[targets]/1000,data.xy[ids]/1000);nearest=distance.min(1);count=(distance<=150).sum(1)
                geometries[t]=(nearest,count)
                for method in ['IDW','kriging']:
                    field=interpolate(data.xy[ids]/1000,values[valid],grid,fit['range_km'],fit['nugget_ratio'],method).reshape(24,24)
                    if not np.isfinite(field).all():raise ValueError('Nonfinite predicted field')
                    for scale,f in [('25km',field),('100km_returned',lowpass(field))]:
                        predictions[(t,method,scale)]=query(f,xy)*np.exp(-height/3000)
                    if index==0 and event==case['events'][0]:np.savez_compressed(OUT/f'pilot_field_{method}.npz',z0_mm=field,coarsened_z0_mm=lowpass(field),target_ids=targets,query_xy=xy,origin=origin)
            t=event['t'];truth=np.array(data.values[t,targets,0])*np.exp(-height/3000);previous=np.array(data.values[t-3,targets,0])*np.exp(-height/3000)
            nearest,count=geometries[t]
            terrain=data.g['height'][y:y+24,x:x+24];deltaheight=height-query(terrain,xy)
            for method in ['IDW','kriging']:
                for scale in ['25km','100km_returned']:
                    pred=predictions[(t,method,scale)];increment=pred-predictions[(t-3,method,scale)]
                    for i,s in enumerate(targets):
                        if not np.isfinite(truth[i]):continue
                        records.append(dict(region=region['name'],fold=case['fold'],b_fraction=case['b_fraction'],time_index=t,event_type=event['type'],station=int(s),height_m=float(height[i]),height_difference_m=float(deltaheight[i]),
                            method=method,scale=scale,prediction_mm=float(pred[i]),target_mm=float(truth[i]),increment_prediction_mm=float(increment[i]),increment_target_mm=float(truth[i]-previous[i]),
                            nearest_source_km=float(nearest[i]),sources_150km=int(count[i]),outside_idw_radius=bool(nearest[i]>150),high_wet=bool(truth[i]>wet_threshold)))
        print('PILOT',index+1,len(planned),region['name'],'fold',case['fold'],'B',case['b_fraction'],flush=True)
    frame=pd.DataFrame(records);frame.to_csv(OUT/'local_pilot_station_predictions.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for keys,f in frame.groupby(['region','b_fraction','method','scale']):
        r=dict(zip(['region','b_fraction','method','scale'],keys));r.update(metrics(f));r['outside_idw_radius_fraction']=float(f.outside_idw_radius.mean())
        inc=f[np.isfinite(f.increment_target_mm)]
        r['increment_3h_rmse_mm']=float(np.sqrt(np.mean((inc.increment_prediction_mm-inc.increment_target_mm)**2)))
        score,pair_count=pair_metrics(f,data.xy,data.height)
        r['z0_difference_0_50km_rmse_mm']=score;r['pair_hours_0_50km']=pair_count;summary.append(r)
    conditions=[]
    for keys,f in frame.groupby(['method','scale','b_fraction']):
        for label,mask in [('near_le50km',f.nearest_source_km<=50),('far_gt50km',f.nearest_source_km>50),('height_diff_le100m',abs(f.height_difference_m)<=100),('height_diff_gt100m',abs(f.height_difference_m)>100),('high_wet',f.high_wet),('rapid_A_event',f.event_type=='rapid_input_A_proxy')]:
            if mask.any():conditions.append(dict(method=keys[0],scale=keys[1],b_fraction=keys[2],condition=label,**metrics(f[mask])))
    result=dict(synthetic=False,exploratory=True,optimizer_updates=0,seconds=time.time()-started,source_case_count=len(planned),domain_hours=len(planned)*12,station_prediction_rows=len(frame),
        summary=summary,conditions=conditions,covariance_fits=covariances,restrictions=manifest['info_condition'],limitations=['Two selected regions,12 events each,3folds; not annual accuracy or statistical proof','Current high terrain group uses coarseDEM representation proxy','Kriging uncertainty not evaluated here','No C2019 or D2020 observations used','Same paired field queries used across methods/scales'])
    (OUT/'local_pilot_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(summary).to_csv(OUT/'local_pilot_summary.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(seconds=result['seconds'],rows=len(frame),summary=summary),indent=2),flush=True)

if __name__=='__main__':main()
