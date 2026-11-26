"""Build observation-supported hourly 30 km ZWD reference fields, product time."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree, Delaunay, QhullError
import h5py

STUDY = Path(__file__).resolve().parents[1]
PIPE = STUDY.parents[1]
INV = STUDY/'01_inventory/pwv_local/selection_2007_2011'
OUT = STUDY/'02_data/processed/reference_30km_v1'
SPACING_KM = 30

def grid_centers(spacing_km):
    if spacing_km not in (25,30):
        raise ValueError('Supported grid spacing is 25 or 30 km')
    step=spacing_km*1000
    return np.arange(-300000+step/2,300000,step)

def height_reduce(z, h, scale):
    return z * np.exp(h/scale)

def weights(xy, query, k, power, radius_m=60000, nearest_max_m=30000):
    n = len(xy)
    if n < 3:
        return np.zeros((len(query),n)), np.zeros(len(query),bool), np.full(len(query),np.inf), np.zeros(len(query),int)
    tree = cKDTree(xy)
    d, idx = tree.query(query, k=min(k,n))
    count = tree.query_ball_point(query, radius_m, return_length=True)
    try:
        hull = Delaunay(xy).find_simplex(query)>=0
    except QhullError:
        hull = np.zeros(len(query),bool)
    mask = (d[:,0]<=nearest_max_m) & (count>=3) & hull
    w = np.where(d<=radius_m, np.maximum(d,1e-6)**(-power), 0.)
    exact = d[:,0]<1e-6
    w[exact] = 0
    w[exact,0] = 1
    w /= np.maximum(w.sum(axis=1,keepdims=True),1e-300)
    full = np.zeros((len(query),n))
    np.put_along_axis(full,idx,w,axis=1)
    return full, mask, d[:,0], count

def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def prepare():
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import shape, Point
    OUT.mkdir(parents=True,exist_ok=True)
    stations = pd.read_csv(INV/'export_station_inventory.csv')
    # Freeze roles using coordinates/groups only; main cohort is assigned first.
    groups = stations.groupby('geometry_group_id').agg(x_km=('x_km','mean'),y_km=('y_km','mean'),main=('in_main_2008_2010','max'))
    rng = np.random.default_rng(42)
    groups['role'] = 'reference'
    for main in [True,False]:
        subset = groups[groups.main==main]
        bins = subset.assign(bx=np.floor(subset.x_km/100),by=np.floor(subset.y_km/100))
        for _, block in bins.groupby(['bx','by']):
            ids = rng.permutation(block.index.to_numpy())
            # Local largest remainder allocation; small bins keep reference support.
            hold = int(np.floor(len(ids)*.30+.5)) if len(ids)>=3 else 0
            for j, gid in enumerate(ids[:hold]):
                groups.loc[gid,'role'] = 'validation' if j%2==0 else 'independent_test'
    stations['role'] = stations.geometry_group_id.map(groups.role)
    if SPACING_KM==25:
        frozen=pd.read_csv(STUDY/'02_data/processed/reference_30km_v1/station_roles.csv')
        assert frozen.set_index('station').role.equals(stations.set_index('station').role), '25/30 km roles differ'
    role_path = OUT/'station_roles.csv'
    if role_path.exists():
        previous = pd.read_csv(role_path)
        assert previous.set_index('station').role.equals(stations.set_index('station').role), 'Frozen roles changed'
    stations.to_csv(role_path,index=False)
    with (PIPE/'data/external/egm96_15.gtx').open('rb') as f:
        lat,lon,dy,dx,ny,nx=struct.unpack('>4d2i',f.read(40))
        g=np.frombuffer(f.read(),dtype='>f4').reshape(ny,nx)
    geoid=RegularGridInterpolator((lat+dy*np.arange(ny),lon+dx*np.arange(nx)),g)
    stations['orthometric_height_m']=stations.ellipsoidal_height_m-geoid(stations[['lat','lon']].values)
    if SPACING_KM==30:
        grid=pd.read_csv(PIPE/'studies/us_dense_zwd_20261105_plan/expanded_region_v2/northwest_600_grid.csv')[['x_km','y_km','lon','lat','us_land']]
    else:
        xx,yy=np.meshgrid(grid_centers(SPACING_KM),grid_centers(SPACING_KM))
        inverse=Transformer.from_crs('+proj=aeqd +lat_0=35 +lon_0=-118 +datum=WGS84 +units=m',4326,always_xy=True)
        longitude,latitude=inverse.transform(xx.ravel(),yy.ravel())
        features=json.loads((PIPE/'studies/us_dense_zwd_20261105_plan/evidence/countries.geojson').read_text(encoding='utf-8'))['features']
        usa=shape(next(f['geometry'] for f in features if f['properties']['ADMIN']=='United States of America'))
        grid=pd.DataFrame(dict(x_km=xx.ravel()/1000,y_km=yy.ravel()/1000,lon=longitude,lat=latitude,us_land=[usa.covers(Point(a,b)) for a,b in zip(longitude,latitude)]))
    terrain=STUDY/'02_data/processed/terrain/srtm_cgiar_v41_90m_support_123W_113W_31N_39N.tif'
    with rasterio.open(terrain) as src:
        h=np.array([v[0] for v in src.sample(grid[['lon','lat']].values)],float)
        h[h==src.nodata]=np.nan
    grid['height_egm96_m']=h
    grid['geoid_undulation_m']=geoid(grid[['lat','lon']].values)
    grid['eligible']=grid.us_land & np.isfinite(h)
    grid.to_csv(OUT/'grid_geometry.csv',index=False)
    stations.to_csv(role_path,index=False)
    return stations,grid

def read_reference(stations):
    # Held-out files are never opened by the field builder or parameter selection.
    start=int(np.datetime64('2007-01-01','h').astype(int))
    end=int(np.datetime64('2012-01-01','h').astype(int))
    refs=stations[stations.role=='reference'].sort_values('station').reset_index(drop=True)
    z=np.full((end-start,len(refs)),np.nan,dtype=np.float64)
    for j,row in enumerate(refs.itertuples()):
        df=pd.read_parquet(STUDY/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',columns=['hour_id','zwd_mm','joint_valid'])
        df=df[df.joint_valid & df.hour_id.between(start,end-1)]
        z[df.hour_id.to_numpy()-start,j]=df.zwd_mm.to_numpy()
    return refs,z,start

def select_parameters(refs,z,start):
    # Fixed 7-day/6-hour sample in 2008; identical target observations across candidates.
    offset=int(np.datetime64('2008-01-01','h').astype(int))-start
    sample=offset+np.arange(0,366*24,7*24)[:,None]+np.array([0,6,12,18])[None,:]
    zz=z[sample.ravel()].copy()
    zz[:,~refs.in_main_2008_2010.to_numpy()]=np.nan
    xy=refs[['x_km','y_km']].to_numpy()*1000
    h=refs.orthometric_height_m.to_numpy()
    records=[]
    for scale in [1500,2000,3000,4000]:
        for k in [4,8,12]:
            for p in [1,2]:
                errors=[]
                for target in np.flatnonzero(refs.in_main_2008_2010):
                    available=np.flatnonzero(refs.in_main_2008_2010.to_numpy() & (np.arange(len(refs))!=target))
                    # Recompute support for every distinct missingness pattern in sample.
                    patterns, inverse=np.unique(np.isfinite(zz[:,available]),axis=0,return_inverse=True)
                    for pattern_id,valid in enumerate(patterns):
                        src=available[valid]
                        w,mask,_,_=weights(xy[src],xy[target:target+1],k,p)
                        rows=np.flatnonzero((inverse==pattern_id)&np.isfinite(zz[:,target]))
                        if mask[0] and len(rows):
                            pred=height_reduce(zz[np.ix_(rows,src)],h[src],scale)@w[0]*np.exp(-h[target]/scale)
                            errors.extend((pred-zz[rows,target]).tolist())
                e=np.asarray(errors)
                records.append(dict(scale_height_m=scale,neighbors=k,power=p,n=len(e),rmse_mm=float(np.sqrt(np.mean(e*e))),mae_mm=float(np.mean(abs(e)))))
    cv=pd.DataFrame(records).sort_values('rmse_mm')
    cv.to_csv(OUT/'reference_internal_2008_loo_cv.csv',index=False)
    best=cv.iloc[0].to_dict()
    dump(OUT/'selected_parameters.json',best)
    return best

def build():
    refs_all,grid=prepare()
    refs,z,start=read_reference(refs_all)
    parameter_path=OUT/'selected_parameters.json'
    if SPACING_KM==25:
        best=json.loads((STUDY/'02_data/processed/reference_30km_v1/selected_parameters.json').read_text(encoding='utf-8'))
        dump(parameter_path,best)
    best=json.loads(parameter_path.read_text(encoding='utf-8')) if parameter_path.exists() else select_parameters(refs,z,start)
    print('Selected',best,flush=True)
    xy=refs[['x_km','y_km']].to_numpy()*1000
    query=grid[['x_km','y_km']].to_numpy()*1000
    h=refs.orthometric_height_m.to_numpy()
    scale=best['scale_height_m']
    dates=pd.date_range('2007-01-01','2012-01-01',freq='h',inclusive='left')
    stats=[]
    hourly_stats=[]
    side=len(grid_centers(SPACING_KM))
    for period in dates.to_period('M').unique():
        rows=np.flatnonzero(dates.to_period('M')==period)
        year=period.year
        cohort=refs['in_annual_2007' if year==2007 else 'in_annual_2011' if year==2011 else 'in_main_2008_2010'].to_numpy()
        valid=np.isfinite(z[rows]) & cohort
        # One observation per canonical 1 km group; prefer first available alias.
        for _, aliases in refs.groupby('geometry_group_id').groups.items():
            aliases=np.asarray(list(aliases))
            if len(aliases)>1:
                block=valid[:,aliases]
                valid[:,aliases]=block & (np.cumsum(block,axis=1)==1)
        patterns,inv=np.unique(valid,axis=0,return_inverse=True)
        field=np.full((len(rows),len(grid)),np.nan,dtype=np.float32)
        near=np.full_like(field,np.nan)
        counts=np.zeros(field.shape,dtype=np.uint16)
        for pid,pattern in enumerate(patterns):
            src=np.flatnonzero(pattern)
            times=np.flatnonzero(inv==pid)
            w,mask,d,n=weights(xy[src],query,int(best['neighbors']),best['power'])
            mask &= grid.eligible.to_numpy()
            values=height_reduce(z[np.ix_(rows[times],src)],h[src],scale)@w.T
            values[:,~mask]=np.nan
            field[times]=values
            near[times]=d/1000
            counts[times]=n
        surface=field*np.exp(-grid.height_egm96_m.to_numpy()[None,:]/scale)
        path=OUT/f'zwd_reference_{SPACING_KM}km_{period}.h5'
        with h5py.File(path,'w') as f:
            f.attrs.update(time_basis='GPAC product nominal calendar hour; UTC/GPST unspecified',height_datum='EGM96 orthometric',reference_height_m=0.,scale_height_m=scale,neighbors=int(best['neighbors']),power=best['power'],grid_spacing_m=SPACING_KM*1000,projection='+proj=aeqd +lat_0=35 +lon_0=-118 +datum=WGS84 +units=m',cv_rmse_surface_mm=best['rmse_mm'],cv_interpretation='2008 reference-station LOO tuning diagnostic, not independent accuracy or calibrated uncertainty')
            f.create_dataset('hour_id',data=rows+start).attrs['units']='nominal hours since 1970-01-01'
            for name,data,unit in [('zwd_h0_mm',field,'mm'),('zwd_surface_mm',surface,'mm'),('support_mask',np.isfinite(field),'boolean'),('nearest_station_km',near,'km'),('stations_within_60km',counts,'count')]:
                f.create_dataset(name,data=data.reshape(-1,side,side),compression='gzip',shuffle=True).attrs['units']=unit
            for col in ['lon','lat','height_egm96_m','eligible']:
                f.create_dataset(col,data=grid[col].to_numpy().reshape(side,side))
            f.create_dataset('x_m',data=grid_centers(SPACING_KM))
            f.create_dataset('y_m',data=grid_centers(SPACING_KM))
        # Reopen each product and verify masks, height round-trip and time axis.
        with h5py.File(path,'r') as f:
            a=f['zwd_h0_mm'][:]; b=f['zwd_surface_mm'][:]
            assert np.array_equal(np.isfinite(a),f['support_mask'][:])
            np.testing.assert_allclose(b*np.exp(grid.height_egm96_m.to_numpy().reshape(side,side)/scale),a,rtol=2e-6,equal_nan=True)
            assert np.array_equal(f['hour_id'][:],rows+start)
        stats.append(dict(month=str(period),hours=len(rows),mean_supported_cells=float(np.isfinite(field).sum(axis=1).mean()),minimum_supported_cells=int(np.isfinite(field).sum(axis=1).min()),maximum_supported_cells=int(np.isfinite(field).sum(axis=1).max()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(str(period),stats[-1]['mean_supported_cells'],flush=True)
        hourly_stats.append(pd.DataFrame(dict(hour_id=rows+start,epoch_product=dates[rows],year=year,eligible_reference_stations=int(cohort.sum()),available_geometry_groups=valid.sum(axis=1),supported_cells=np.isfinite(field).sum(axis=1))))
    pd.DataFrame(stats).to_csv(OUT/'monthly_verification.csv',index=False)
    pd.concat(hourly_stats,ignore_index=True).to_csv(OUT/'hourly_availability.csv',index=False)
    dump(OUT/'build_summary.json',dict(hours=len(dates),months=len(stats),grid_shape=[side,side],land_cells=int(grid.eligible.sum()),reference_stations=len(refs),roles=refs_all.role.value_counts().to_dict(),parameters=best,mean_supported_cells=float(np.average([s['mean_supported_cells'] for s in stats],weights=[s['hours'] for s in stats])),verified=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spacing-km',type=int,choices=[25,30],default=30)
    args=parser.parse_args()
    SPACING_KM=args.spacing_km
    OUT=STUDY/f'02_data/processed/reference_{SPACING_KM}km_v1'
    build()
