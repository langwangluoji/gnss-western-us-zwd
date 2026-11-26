"""CPU-only terrain-path audit of frozen seed42 exposed development predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
CRS='+proj=lcc +lat_0=37.5 +lon_0=-117.5 +lat_1=30 +lat_2=45 +datum=WGS84 +units=m'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def bilinear_dem(height,xy_m,origin_m,spacing_m):
    uv=(np.asarray(xy_m)-origin_m)/spacing_m-.5
    clipped=((uv<0)|(uv>np.array([height.shape[1]-1,height.shape[0]-1]))).any(-1)
    uv=np.clip(uv,0,np.array([height.shape[1]-1,height.shape[0]-1]))
    ix=np.floor(uv[...,0]).astype(int);iy=np.floor(uv[...,1]).astype(int)
    jx=np.minimum(ix+1,height.shape[1]-1);jy=np.minimum(iy+1,height.shape[0]-1)
    fx=uv[...,0]-ix;fy=uv[...,1]-iy
    value=(height[iy,ix]*(1-fx)*(1-fy)+height[iy,jx]*fx*(1-fy)
        +height[jy,ix]*(1-fx)*fy+height[jy,jx]*fx*fy)
    return value,clipped


def path_features(height,query_m,condition_m,query_height_m,condition_height_m,origin_m,spacing_m,points=31):
    t=np.linspace(0,1,points)
    positions=query_m[None]*(1-t[:,None])+condition_m[None]*t[:,None]
    z,clipped=bilinear_dem(height,positions,origin_m,spacing_m)
    chord=query_height_m*(1-t)+condition_height_m*t
    dem_chord=z[0]*(1-t)+z[-1]*t
    return dict(barrier_m=float(np.maximum(z-chord,0).max()),
        above_max_station_m=float(z.max()-max(query_height_m,condition_height_m)),
        dem_chord_barrier_m=float(np.maximum(z-dem_chord,0).max()),
        clipped=bool(clipped.any()),clipped_points=int(clipped.sum()),
        query_dem_minus_station_m=float(z[0]-query_height_m),
        condition_dem_minus_station_m=float(z[-1]-condition_height_m),
        max_dem_m=float(z.max()),path_samples=points)


def nearest_visible(query_km,condition_km,visible,condition_ids):
    distance=np.linalg.norm(query_km[:,None]-condition_km[None],axis=-1)
    order=np.argsort(condition_ids,kind='stable')
    available=np.where(visible[:,None,:],distance[None],np.inf)[:,:,order]
    chosen=np.argmin(available,axis=-1);value=np.take_along_axis(available,chosen[...,None],axis=-1)[...,0]
    index=order[chosen];index[~np.isfinite(value)]=-1
    return index,value


def controlled_contrast(frame):
    usable=frame[(frame.n>0)&frame.barrier_group.isin([0,1])]
    sums=usable.groupby(['cell','barrier_group'])[['n','signed','square']].sum()
    stations=usable.groupby(['cell','barrier_group']).station.nunique()
    supported=sums.index[(sums.n>=48)&(stations>=2)]
    low=set(supported[supported.get_level_values(1)==0].get_level_values(0))
    high=set(supported[supported.get_level_values(1)==1].get_level_values(0))
    cells=sorted(low&high)
    if not cells:return dict(supported_cells=0,bias_high_minus_low_mm=np.nan,rmse_high_minus_low_mm=np.nan)
    lo=sums.loc[[(c,0) for c in cells]];hi=sums.loc[[(c,1) for c in cells]]
    weight=np.minimum(lo.n.to_numpy(),hi.n.to_numpy());den=weight.sum()
    def score(part):
        return dict(bias_mm=float(np.sum(weight*part.signed.to_numpy()/part.n.to_numpy())/den),
            rmse_mm=float(np.sqrt(np.sum(weight*part.square.to_numpy()/part.n.to_numpy())/den)),
            valid_hours=float(part.n.sum()))
    l=score(lo);h=score(hi)
    return dict(supported_cells=len(cells),cells=cells,low=l,high=h,
        low_station_count=int(usable[(usable.cell.isin(cells))&(usable.barrier_group==0)].station.nunique()),
        high_station_count=int(usable[(usable.cell.isin(cells))&(usable.barrier_group==1)].station.nunique()),
        bias_high_minus_low_mm=h['bias_mm']-l['bias_mm'],rmse_high_minus_low_mm=h['rmse_mm']-l['rmse_mm'])


def contrast(frame,error_name,draws=250):
    table=frame.assign(signed=frame[error_name],square=frame[error_name]**2).groupby(
        ['cell','barrier_group','q','month'],as_index=False).agg(n=('signed','count'),signed=('signed','sum'),square=('square','sum'))
    table=table.rename(columns={'q':'station'});result=controlled_contrast(table)
    rng=np.random.default_rng(261126);values=[]
    stations=table.station.unique();months=table.month.unique()
    for _ in range(draws):
        sw=pd.Series(rng.choice(stations,len(stations),replace=True)).value_counts()
        mw=pd.Series(rng.choice(months,len(months),replace=True)).value_counts()
        weight=table.station.map(sw).fillna(0).to_numpy()*table.month.map(mw).fillna(0).to_numpy()
        boot=table.copy();boot[['n','signed','square']]=boot[['n','signed','square']].mul(weight,axis=0)
        current=controlled_contrast(boot)
        if current['supported_cells']:
            values.append([current['bias_high_minus_low_mm'],current['rmse_high_minus_low_mm']])
    result['bootstrap_unit']='station and month crossed resampling; all layouts/hours retained; exploratory'
    result['bootstrap_draws']=draws
    if values:
        ci=np.quantile(values,[.025,.975],axis=0)
        result['bias_difference_ci95_mm']=ci[:,0].tolist();result['rmse_difference_ci95_mm']=ci[:,1].tolist()
    return result,table


def summaries(frame,keys):
    rows=[]
    for group,part in frame.groupby(keys,observed=True):
        group=group if isinstance(group,tuple) else (group,)
        row=dict(zip(keys,group));row.update(valid_hours=len(part),stations=part.q.nunique(),
            distance_median_km=part.distance_km.median(),barrier_median_m=part.barrier_m.median())
        for model in ['I1','TF']:
            error=part['error_'+model]
            row[model]=dict(signed_bias_mm=error.mean(),rmse_mm=np.sqrt(np.mean(error**2)),
                station_rmse_mean_mm=np.mean(np.sqrt(part.assign(square=error**2).groupby('q').square.mean())))
        rows.append(row)
    return rows


def cv_r2(X,y,fold):
    prediction=np.zeros(len(y))
    for f in np.unique(fold):
        train=fold!=f;test=~train;mean=X[train].mean(0);std=X[train].std(0);std[std<1e-8]=1
        a=(X[train]-mean)/std;b=(X[test]-mean)/std;intercept=y[train].mean()
        beta=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(y[train]-intercept))
        prediction[test]=intercept+b@beta
    return float(1-np.sum((prediction-y)**2)/np.sum((y-y.mean())**2))


def regression_diagnostic(frame,geometry):
    columns=['distance_km','q_height_m','height_difference_m','height_difference_abs_m','rough_q_m','rough_c_m',
        'slope_q','slope_c','local_height_q_m','local_height_c_m','coast_q_km','coast_delta_km',
        'barrier_m','dem_chord_barrier_m']
    table=frame.groupby(['q','layout','month'],as_index=False)[columns+['error_I1','error_TF']].mean()
    rms=frame.assign(square_I1=frame.error_I1**2,square_TF=frame.error_TF**2).groupby(['q','layout','month'])[['square_I1','square_TF']].mean()
    table=table.merge(np.sqrt(rms).reset_index(),on=['q','layout','month'])
    baseline=columns[:-2]
    X=table[baseline].to_numpy();X[:,0]=np.log1p(X[:,0])
    X=np.c_[X,pd.get_dummies(table.month,dtype=float).to_numpy()[:,1:],pd.get_dummies(table.layout,dtype=float).to_numpy()[:,1:]]
    extra=np.log1p(table[['barrier_m','dem_chord_barrier_m']].to_numpy())
    xy=geometry['dev_xy'][table.q.to_numpy()];block=np.floor(xy/100).astype('int64')
    fold=((block[:,0]*73856093)^(block[:,1]*19349663))%5
    results=dict(units=len(table),query_stations=int(table.q.nunique()),fold_definition='100km coordinate blocks hashed to five folds; same folds for all diagnostic regressions',
        baseline_features=baseline+['month dummies','layout dummies'],added_features=['log1p(barrier_m)','log1p(dem_chord_barrier_m)'],
        ridge=1.,outcome_definition='station x layout x month mean error / current RMSE; diagnostic only, not corrected ZWD')
    for model in ['I1','TF']:
        results[model]={}
        for outcome,column in [('signed_bias','error_'+model),('current_rmse','square_'+model)]:
            y=table[column].to_numpy();base=cv_r2(X,y,fold);full=cv_r2(np.c_[X,extra],y,fold)
            results[model][outcome]=dict(baseline_spatial_cv_r2=base,with_barriers_spatial_cv_r2=full,incremental_r2=full-base)
    results['existing_features_predict_log_barrier_spatial_cv_r2']=cv_r2(X,extra[:,0],fold)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    dem_path=ENGINE.parents[1]/'02_data/processed/gnss_field_server_v1/geometry.npz'
    geo_path=ENGINE/'results/20261125_temporal_continuous_data/geometry.npz'
    cache=ENGINE/'results/20261126_native_weather_cache_v2/calendar8_s42'
    emb=ENGINE/'results/20261126_native_weather_embeddings/transformer72_physics_aux_last_dev_s42'
    geometry=dict(np.load(geo_path));dem=dict(np.load(dem_path))
    dem_manifest=dem_path.parent/'manifest.json';height_metadata=dem_path.parent/'statistics.json'
    geo_contract=geo_path.parent/'contract.json'
    if sha(dem_path)!=json.loads(dem_manifest.read_text())['shared']['geometry.npz']:raise ValueError('registered DEM hash differs')
    if sha(geo_path)!=json.loads(geo_contract.read_text(encoding='utf-8'))['files']['geometry.npz']:raise ValueError('registered station geometry differs')
    height_datum=json.loads(height_metadata.read_text())['height_datum']
    if height_datum!='EGM96':raise ValueError('height datum differs')
    cache_manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    embedding_manifest=json.loads((emb/'manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((cache/'protocol.json').read_text(encoding='utf-8'))
    if protocol['config']['calendar_day']!=8 or protocol['config']['seed']!=42:raise ValueError('exposed seed42 calendar8 only')
    if embedding_manifest['cache_manifest_sha256']!=sha(cache/'manifest.json'):raise ValueError('prediction/cache source mismatch')
    file_names=['history','context_ids','query_ids','origins','nearest','mean','truth']
    source_hashes={str(dem_path):sha(dem_path),str(geo_path):sha(geo_path),str(cache/'protocol.json'):sha(cache/'protocol.json'),
        str(cache/'manifest.json'):sha(cache/'manifest.json'),str(emb/'manifest.json'):sha(emb/'manifest.json'),
        str(emb/'mean.npy'):sha(emb/'mean.npy'),str(Path(__file__)):sha(Path(__file__))}
    for path in [dem_manifest,height_metadata,geo_contract,ENGINE/'tools/terrain_diffusion/data.py',ENGINE/'tools/terrain_diffusion/coast_features.py']:
        source_hashes[str(path)]=sha(path)
    for name in file_names:
        actual=sha(cache/f'{name}.npy');source_hashes[str(cache/f'{name}.npy')]=actual
        if actual!=cache_manifest['files'][f'{name}.npy']:raise ValueError('cache hash changed: '+name)
    if sha(emb/'mean.npy')!=embedding_manifest['files']['mean.npy']:raise ValueError('frozen prediction hash changed')
    a={name:np.load(cache/f'{name}.npy',mmap_mode='r') for name in file_names}
    prediction=np.load(emb/'mean.npy',mmap_mode='r')
    spacing=float(np.diff(dem['x'][0]).mean());origin=dem['origin']
    np.testing.assert_allclose(dem['x'],np.broadcast_to(origin[0]+(np.arange(dem['height'].shape[1])+.5)*spacing,dem['x'].shape))
    np.testing.assert_allclose(dem['y'],np.broadcast_to((origin[1]+(np.arange(dem['height'].shape[0])+.5)*spacing)[:,None],dem['y'].shape))
    assert spacing==25000
    tx,ty=Transformer.from_crs(4326,CRS,always_xy=True).transform(dem['lon'],dem['lat'])
    crs_error=float(max(np.max(np.abs(tx-dem['x'])),np.max(np.abs(ty-dem['y']))))
    if crs_error>1:raise ValueError('DEM CRS fails meter-coordinate numerical verification')
    coord_error=max(float(np.max(np.abs(geometry[p+'static'][:,:2]*1000-geometry[p+'xy']))) for p in ('','dev_'))
    height_error=max(float(np.max(np.abs(geometry[p+'static'][:,2]*3000-geometry[p+'height']))) for p in ('','dev_'))
    if coord_error>.001 or height_error>.001:raise ValueError('static coordinate/height unit contract differs')
    units=dict(dem_shape=dem['height'].shape,dem_origin_m=origin,grid_spacing_m=spacing,height_datum=height_datum,
        grid_center_rule='origin + (index+0.5)*25000 meters; bilinear on pixel centers',crs=CRS,
        crs_forward_roundtrip_max_error_m=crs_error,static_xy_times1000_vs_registered_xy_max_error_km=coord_error,
        static_height_times3000_vs_registered_height_max_error_m=height_error,
        dem_height_range_m=[float(np.nanmin(dem['height'])),float(np.nanmax(dem['height']))])
    frames=[];max_nearest_difference=0.;visibility_changed=0;all_missing=0
    for b in range(len(a['mean'])):
        c=np.asarray(a['context_ids'][b],int);q=np.asarray(a['query_ids'][b],int)
        visible=np.asarray(a['history'][b,-12:,:,3])>0
        nearest,distance=nearest_visible(geometry['dev_xy'][q],geometry['xy'][c],visible,geometry['station_ids'][c])
        cached=np.asarray(a['nearest'][b],float)
        if not np.array_equal(np.isfinite(cached),np.isfinite(distance)):raise ValueError('cached nearest visibility differs')
        finite=np.isfinite(cached);max_nearest_difference=max(max_nearest_difference,float(np.max(np.abs(cached[finite]-distance[finite]),initial=0)))
        all_missing+=int((nearest<0).sum())
        static_nearest,_=nearest_visible(geometry['dev_xy'][q],geometry['xy'][c],np.ones((12,len(c)),bool),geometry['station_ids'][c])
        visibility_changed+=int((static_nearest!=nearest).sum())
        station=np.where(nearest>=0,c[np.maximum(nearest,0)],-1)
        hours=np.asarray(a['origins'][b]);month=(hours.astype('datetime64[h]').astype('datetime64[M]').astype(int)%12)+1
        truth=np.asarray(a['truth'][b,:,:,0],float)
        frames.append(pd.DataFrame(dict(q=np.broadcast_to(q,distance.shape).ravel(),c=station.ravel(),
            layout=b//72,month=np.broadcast_to(month[:,None],distance.shape).ravel(),distance_km=distance.ravel(),
            error_I1=(np.asarray(a['mean'][b],float)-truth).ravel(),error_TF=(np.asarray(prediction[b,:,:,0],float)-truth).ravel(),
            error_F6=(np.asarray(prediction[b,:,:,6],float)-np.asarray(a['truth'][b,:,:,6],float)).ravel())))
    if max_nearest_difference>.001:raise ValueError(f'nearest difference {max_nearest_difference}km')
    records=pd.concat(frames,ignore_index=True);pairs=[]
    for q,c in records.loc[records.c>=0,['q','c']].drop_duplicates().itertuples(index=False,name=None):
        qs=geometry['dev_static'][q];cs=geometry['static'][c]
        features=path_features(dem['height'],geometry['dev_xy'][q]*1000,geometry['xy'][c]*1000,
            geometry['dev_height'][q],geometry['height'][c],origin,spacing)
        features.update(q=q,c=c,query_station=geometry['dev_ids'][q],condition_station=geometry['station_ids'][c],
            q_height_m=geometry['dev_height'][q],c_height_m=geometry['height'][c],
            height_difference_m=geometry['dev_height'][q]-geometry['height'][c],
            height_difference_abs_m=abs(geometry['dev_height'][q]-geometry['height'][c]),
            rough_q_m=float(qs[5]*500),rough_c_m=float(cs[5]*500),
            slope_q=float(np.linalg.norm(qs[3:5])*.1),slope_c=float(np.linalg.norm(cs[3:5])*.1),
            local_height_q_m=float(qs[6]*1000),local_height_c_m=float(cs[6]*1000),
            coast_q_km=float(geometry['dev_coast'][q]),coast_c_km=float(geometry['coast'][c]),
            coast_delta_km=float(geometry['dev_coast'][q]-geometry['coast'][c]))
        features['coastal_to_inland']=features['coast_q_km']>=50 and features['coast_delta_km']>=25
        pairs.append(features)
    pair_table=pd.DataFrame(pairs);records=records.merge(pair_table,on=['q','c'],how='left')
    clipped_count=int(records.clipped.fillna(True).sum())
    valid=records[(records.c>=0)&(~records.clipped.fillna(True))&np.isfinite(records.error_I1)&np.isfinite(records.error_TF)].copy()
    valid['distance_bin']=np.digitize(valid.distance_km,[50,100,150,200,300])
    valid['height_bin']=np.digitize(valid.q_height_m,[250,750,1500,2500])
    valid['cell']=((valid.month-1)*6+valid.distance_bin)*5+valid.height_bin
    valid['barrier_bin']=pd.cut(valid.barrier_m,[-.001,100,300,700,1500,np.inf],right=False,labels=['<100','100-300','300-700','700-1500','>=1500'])
    valid['barrier_group']=np.where(valid.barrier_m<100,0,np.where(valid.barrier_m>=300,1,-1))
    controls={};tables=[]
    for model in ['I1','TF']:
        controls[model],table=contrast(valid,'error_'+model);table['model']=model;tables.append(table)
    pd.concat(tables).to_csv(out/'controlled_cells.csv',index=False)
    pair_table.to_csv(out/'pair_features.csv',index=False)
    station_rows=[]
    for q,part in valid.groupby('q'):
        row=dict(station=geometry['dev_ids'][q],q=int(q),valid_hours=len(part),height_m=part.q_height_m.iloc[0],coast_km=part.coast_q_km.iloc[0],
            barrier_median_m=part.barrier_m.median(),barrier_p90_m=part.barrier_m.quantile(.9),
            dem_chord_barrier_median_m=part.dem_chord_barrier_m.median(),high_barrier_fraction=(part.barrier_m>=300).mean(),
            distance_median_km=part.distance_km.median(),query_dem_minus_station_m=part.query_dem_minus_station_m.median(),
            TF_current_rmse_mm=np.sqrt(np.mean(part.error_TF**2)),TF_current_bias_mm=part.error_TF.mean(),
            I1_current_rmse_mm=np.sqrt(np.mean(part.error_I1**2)),TF_F6_rmse_mm=np.sqrt(np.mean(part.error_F6**2)))
        station_rows.append(row)
    diagnostic=regression_diagnostic(valid,geometry)
    sensitivity={}
    for name in ['dem_chord_barrier_m','above_max_station_m']:
        alternate=valid.copy();alternate['barrier_group']=np.where(alternate[name]<100,0,np.where(alternate[name]>=300,1,-1))
        sensitivity[name],_=contrast(alternate,'error_TF',draws=100)
    layout_control=valid.copy();layout_control['cell']=layout_control.cell*3+layout_control.layout
    strict_layout,_=contrast(layout_control,'error_TF',draws=100)
    direction_controls={}
    for direction,part in valid.groupby('coastal_to_inland'):
        direction_controls[str(bool(direction))],_=contrast(part,'error_TF',draws=100)
    result=dict(scope='seed42; exposed2020 calendar8; 137development stations; 3layouts; geometry/error audit, not causal or new model precision',
        units=units,nearest=dict(max_absolute_difference_km=max_nearest_difference,query_origins_where_visibility_changes_nearest=visibility_changed,
            no_visible_condition_query_origins=all_missing,mask='history[-12:,:,3]>0 at each of12current origins; context_ids select410training coordinates'),
        sample=dict(total_query_origins=len(records),valid_current_label_and_unclipped_query_origins=len(valid),
            query_stations=valid.q.nunique(),unique_nearest_pairs=len(pair_table),clipped_query_origins=clipped_count,
            clipped_pairs=int(pair_table.clipped.sum()),calendar='2020monthly days8-10,72origins x3layouts',
            missing_current_label_query_origins=int((~np.isfinite(records.error_I1)).sum()),
            registered_stations_without_current_labels=[str(geometry['dev_ids'][q]) for q in range(137) if q not in set(valid.q)],
            record_unit='layout x origin x query; same observed station-hours repeated across3layouts, not independent hours',
            barrier_zero_fraction=float((valid.barrier_m==0).mean()),query_dem_station_absolute_mismatch_median_m=float(valid.query_dem_minus_station_m.abs().median())),
        strata_definition=dict(distance_edges_km=[0,50,100,150,200,300,'inf'],query_height_edges_m=['-inf',250,750,1500,2500,'inf'],
            barrier_edges_m=[0,100,300,700,1500,'inf'],contrast='low<100m vs high>=300m; same distance-bin/month/query-height-bin; min48hours and2stations pergroup'),
        barrier_strata=summaries(valid,['barrier_bin']),coastal_direction_strata=summaries(valid,['coastal_to_inland','barrier_bin']),
        detailed_strata=summaries(valid,['distance_bin','month','height_bin','barrier_bin']),
        matched_controls=controls,matched_additional_layout_control_TF=strict_layout,
        matched_coastal_direction_TF=direction_controls,endpoint_sensitivity_TF=sensitivity,diagnostic_regression=diagnostic,
        stations=station_rows,source_sha256=source_hashes,strict_oof=False,confirmation_C_D_read=False,external_downloads=False)
    (out/'audit.json').write_text(json.dumps(plain(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    make_plot(result,out);write_report(result,out)
    print(json.dumps(plain(dict(sample=result['sample'],nearest=result['nearest'],controls=controls,diagnostic=diagnostic)),ensure_ascii=False))


def make_plot(result,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Terrain paths and frozen current-field errors | exposed 2020 development',fontsize=15)
    rows=result['barrier_strata'];x=np.arange(len(rows));width=.36
    for j,(model,color) in enumerate([('I1','#8d9ca9'),('TF','#246b91')]):
        axes[0,0].bar(x+(j-.5)*width,[r[model]['rmse_mm'] for r in rows],width,color=color,label=model)
    axes[0,0].set_xticks(x,[f"{r['barrier_bin']}\n{r['stations']} station"+('s' if r['stations']!=1 else '') for r in rows]);axes[0,0].set(xlabel='Path barrier (m)',ylabel='Current RMSE (mm)',title='Unadjusted groups (different station populations)');axes[0,0].legend()
    for j,(model,color) in enumerate([('I1','#8d9ca9'),('TF','#246b91')]):
        c=result['matched_controls'][model]
        if c['supported_cells']:
            axes[0,1].plot([0,1],[c['low']['rmse_mm'],c['high']['rmse_mm']],marker='o',color=color,label=f"{model}: {c['supported_cells']} common cells")
    axes[0,1].set_xticks([0,1],['Barrier <100m','Barrier >=300m']);axes[0,1].set(ylabel='Standardized current RMSE (mm)',title='Matched distance / month / query-height bins');axes[0,1].legend()
    c=result['matched_controls']['TF'];ci=c['rmse_difference_ci95_mm']
    axes[0,1].text(.55,.04,f"TF high-low = {c['rmse_high_minus_low_mm']:+.3f} mm\nStation x month bootstrap95% [{ci[0]:+.3f}, {ci[1]:+.3f}]",transform=axes[0,1].transAxes,fontsize=9,ha='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    labels=[];values=[]
    for model in ['I1','TF']:
        for outcome in ['signed_bias','current_rmse']:
            labels.append(model+' '+outcome.replace('_',' '));values.append(result['diagnostic_regression'][model][outcome]['incremental_r2'])
    axes[1,0].barh(labels,values,color=['#8d9ca9','#8d9ca9','#246b91','#246b91']);axes[1,0].axvline(0,color='#555',lw=.8)
    axes[1,0].set(xlabel='Added spatial-CV R2 from barriers',title='Error-association diagnostic; not ZWD improvement')
    stations=result['stations'];xx=[s['barrier_median_m'] for s in stations];yy=[s['TF_current_rmse_mm'] for s in stations]
    scatter=axes[1,1].scatter(xx,yy,c=[min(s['coast_km'],300) for s in stations],cmap='viridis',s=24,alpha=.75)
    for s in stations:
        if s['station'] in ['PLTX','P475','PTEX','P510','P786']:
            axes[1,1].annotate(s['station'],(s['barrier_median_m'],s['TF_current_rmse_mm']),xytext=(4,5),textcoords='offset points',fontsize=9)
    axes[1,1].set(xlabel='Station median nearest-path barrier (m)',ylabel='TF current RMSE (mm)',title='Station view; named stations were highlighted by F6')
    fig.colorbar(scatter,ax=axes[1,1],label='Distance to coast (km; clipped300)')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'terrain_path_audit.png',dpi=160);plt.close(fig)


def write_report(result,out):
    sample=result['sample'];near=result['nearest'];diag=result['diagnostic_regression']
    lines=['# Query–条件站路径地形屏障审计','',
        'seed42，2020每月8–10日已暴露开发日历，137开发站、3布局。I1与冻结Transformer physics_aux均使用既有预测；本报告解释当前重建误差的空间关联，不新增ZWD精度或因果结论。','',
        '## 单位、来源与可见性','',
        f"使用144×96、25km中心格网DEM，覆盖{sample['unique_nearest_pairs']}条实际最近站路径；origin以米表示，站xy以km表示，static[:,:2]×1000恢复km。网格中心与Lambert投影数值核验通过，投影误差{result['units']['crs_forward_roundtrip_max_error_m']:.3g}m；注册高程基准为EGM96。",
        f"从history最后12帧逐origin的GNSS mask和context_ids重建可见站。cached nearest最大复算差{near['max_absolute_difference_km']:.7f}km；忽略缺测会改变最近站的query-origin数为{near['query_origins_where_visibility_changes_nearest']}。",
        f"有效当前标签且路径未截边：{sample['valid_current_label_and_unclipped_query_origins']:,}条；截边{sample['clipped_query_origins']}条（主分析排除），{sample['clipped_pairs']}个站对。路径31点双线性插值；25km DEM衡量区域地形，不解析细窄山脊。",'',
        f"每条记录为布局×起点×查询站，3布局复用观测，不是独立小时。注册137开发站中，{sample['query_stations']}站有当前有效标签；无当前标签站为{', '.join(sample['registered_stations_without_current_labels'])}，缺测标签记录共{sample['missing_current_label_query_origins']:,}条。",'',
        '主proxy=max(DEM路径高程−GNSS两端站高线性连线,0)。同时记录maxDEM−max(两端站高)，并用DEM端点连线作敏感性核查，以区分路径起伏与DEM/站点高程支持差。', '',
        '## 分层结果','',
        '| 屏障/m | 有效小时 | 站数 | I1 bias/mm | I1 RMSE/mm | TF bias/mm | TF RMSE/mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['barrier_strata']:
        lines.append(f"| {r['barrier_bin']} | {r['valid_hours']} | {r['stations']} | {r['I1']['signed_bias_mm']:.3f} | {r['I1']['rmse_mm']:.3f} | {r['TF']['signed_bias_mm']:.3f} | {r['TF']['rmse_mm']:.3f} |")
    lines+=['','表内小时数指含3布局的站时记录。指标为记录微RMSE；JSON另给子集站均RMSE，不能与原主日历站宏RMSE混排。','',
        '## 相同距离、月份与query高程范围','',
        '距离[0,50,100,150,200,300,+∞)km，高程[-∞,250,750,1500,2500,+∞)m，月份固定。低屏障<100m、高屏障≥300m；每格每组≥2站且≥48小时。按两组较小样本量加权，保留共同支撑。区间按站和月份交叉重采样，仅作探索。','']
    for model,c in result['matched_controls'].items():
        if c['supported_cells']:
            lines.append(f"- {model}：{c['supported_cells']}个共同格；低/高组{c['low_station_count']}/{c['high_station_count']}站、{int(c['low']['valid_hours'])}/{int(c['high']['valid_hours'])}小时。高−低bias={c['bias_high_minus_low_mm']:.3f}mm，95%区间{np.round(c['bias_difference_ci95_mm'],3).tolist()}；RMSE差={c['rmse_high_minus_low_mm']:.3f}mm，区间{np.round(c['rmse_difference_ci95_mm'],3).tolist()}。")
    lc=result['matched_additional_layout_control_TF']
    lines.append(f"- 再固定布局的TF敏感性：{lc['supported_cells']}个共同格，RMSE高−低差{lc['rmse_high_minus_low_mm']:.3f}mm，区间{np.round(lc['rmse_difference_ci95_mm'],3).tolist()}。")
    lines+=['','## 与现有局地地形信息比较','',
        '固定5折100km空间块分组的诊断回归，以站×布局×月份的当前bias或RMSE为解释对象。基准包含距离、高程/高差、两端roughness/坡度/站高−局地DEM、海岸距离、月和布局；新增两个屏障量，ridge固定1，无选参。这里的R²衡量开发误差的空间关联，不代表ZWD预测改善。','',
        '| 模型/诊断对象 | 现有特征空间CV R² | 加屏障后 | 增量 |','|---|---:|---:|---:|']
    for model in ['I1','TF']:
        for outcome in ['signed_bias','current_rmse']:
            r=diag[model][outcome];lines.append(f"| {model}/{outcome} | {r['baseline_spatial_cv_r2']:.4f} | {r['with_barriers_spatial_cv_r2']:.4f} | {r['incremental_r2']:+.4f} |")
    lines += ['',f"现有特征预测log屏障的空间CV R²={diag['existing_features_predict_log_barrier_spatial_cv_r2']:.4f}。query端DEM与站高绝对差中位数={sample['query_dem_station_absolute_mismatch_median_m']:.1f}m；原始站高连线proxy包含该支持差。",'',
        '## 误差重点站','',
        '下列站此前因F6较高受到关注；本表分别列当前R与F6，二者不互代。F6采用本审计有效当前子集内可用未来标签的微RMSE，用于定位。屏障为各origin实际最近可见条件站路径的中位数。','',
        '| 站 | 离海/km | 最近距离/km | 屏障/m | DEM端点连线屏障/m | 高屏障占比 | 当前TF RMSE/mm | 当前bias/mm | F6 RMSE/mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in result['stations']:
        if s['station'] in ['PLTX','P475','PTEX','P510','P786']:
            lines.append(f"| {s['station']} | {s['coast_km']:.1f} | {s['distance_median_km']:.1f} | {s['barrier_median_m']:.1f} | {s['dem_chord_barrier_median_m']:.1f} | {s['high_barrier_fraction']:.1%} | {s['TF_current_rmse_mm']:.3f} | {s['TF_current_bias_mm']:.3f} | {s['TF_F6_rmse_mm']:.3f} |")
    lines += ['', '## 方向与端点敏感性','',
        '“条件站更近海、query在内陆”固定为query离海≥50km且比条件站离海多≥25km；分层完整指标见audit.json的coastal_direction_strata。屏障两种端点敏感性的共同支撑对照见endpoint_sensitivity_TF。','',
        '| TF对照定义 | 共同格 | RMSE高−低差/mm | 探索95%区间/mm |','|---|---:|---:|---|']
    for name,c in result['endpoint_sensitivity_TF'].items():
        lines.append(f"| {name} | {c['supported_cells']} | {c['rmse_high_minus_low_mm']:.3f} | {np.round(c['rmse_difference_ci95_mm'],3).tolist()} |")
    for direction,c in result['matched_coastal_direction_TF'].items():
        lines.append(f"| coastal_to_inland={direction} | {c['supported_cells']} | {c['rmse_high_minus_low_mm']:.3f} | {np.round(c['rmse_difference_ci95_mm'],3).tolist()} |")
    lines += ['',
        'PLTX是反例：当前误差较高，但实际最近站路径屏障中位数为0。PTEX的站高连线屏障与DEM端点连线屏障都较高，支持把它保留为个例分析。P786的站高连线屏障约410m，同时query端DEM高于站高约410m；换成DEM端点连线后屏障中位数约78m，说明原proxy会混入25km地形支持与站高的差异。', '',
        '## 对下一条3k对照的判断','',
        '本次证据强度偏弱，不建议现在为路径屏障新增独立3k训练臂。原始分层显示高屏障与较大误差共现；固定距离、月份和query高程后，TF差值区间跨零。加入现有局地高差、roughness、坡度和海岸信息的固定空间块诊断中，两个屏障量没有增加当前bias或RMSE的预测解释力。这个结果适用于本次25km DEM、最近可见站proxy与开发日历，不推广为所有地形机制无效。', '',
        '保留可复用的站对特征及重点站反例，先完成已批准C/H/S同源对照；后续若开展一条地形臂，应只检验一个预先冻结的路径特征，继续与同源继续训练控制比较。', '',
        '## 复查','',
        '`python maintenance/terrain_path_audit_20261126/audit.py`；全部DEM、几何、缓存和冻结预测SHA256见audit.json。只读取当前410训练/137开发轨道；确认/C/D标签未读取；无新外部数据。','',
        '![路径审计](terrain_path_audit.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
