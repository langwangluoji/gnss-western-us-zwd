"""Independent expanded-region source audit; no development label statistics."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import struct
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pwv_local import read_station
from residual_edm_v1.data import write_json,sha256

STUDY=Path(__file__).resolve().parents[2]
PIPE=STUDY.parents[1]
ROOT=STUDY/'04_models/field_generation_v1'
DATA=STUDY/'02_data/processed/field_generation_v1'
CRS='+proj=lcc +lat_0=37.5 +lon_0=-117.5 +lat_1=30 +lat_2=45 +datum=WGS84 +units=m'
START=int(np.datetime64('2016-01-01','h').astype('i8'))
STOP=int(np.datetime64('2021-01-01','h').astype('i8'))

def station_job(row):
    path=DATA/'stations'/f"{row['station']}.npz"
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old['source_sha256']!=sha256(row['source_path']): raise ValueError('Source changed')
        return old
    frame,metadata=read_station(row['source_path'])
    frame=frame[frame.hour_id.between(START,STOP-1)]
    cols=['zwd_mm','temperature_k','pressure_hpa','tm_k','era5_pwv_mm']
    values=frame[cols].to_numpy(dtype='f4')
    # Retain existing parser QC and explicitly validate all five input variables.
    valid=frame.joint_valid.to_numpy()&np.isfinite(values).all(1)&(values[:,4]>=0)&(values[:,4]<=120)
    values[~valid]=np.nan
    ids=frame.hour_id.to_numpy()-START
    np.savez_compressed(path,indices=ids,values=values)
    years=frame.epoch_product.dt.year.to_numpy()
    coverage={str(y):float(valid[years==y].sum()/ (8784 if y%4==0 else 8760)) for y in range(2016,2021)}
    result=dict(station=row['station'],source_sha256=metadata['source_sha256'],source_path=row['source_path'],coverage=coverage,
                valid_hours=int(valid.sum()),cache_sha256=sha256(path))
    write_json(meta,result)
    return result

def main():
    (DATA/'stations').mkdir(parents=True,exist_ok=True)
    inventory=pd.read_csv(STUDY/'01_inventory/pwv_local/global_station_headers.csv')
    inv=inventory[inventory.lat.between(25,50)&inventory.lon.between(-125,-110)].copy()
    # Timestamp-only audit safely prefilters impossible >=90% candidates.
    counts=pd.read_csv(STUDY/'06_reports/data_distribution_20261116/full_raw_time/station_year_record_coverage.csv')
    counts=counts[counts.year.between(2016,2018)].assign(fraction=lambda x:x.record_hours/x.calendar_hours)
    candidates=counts.groupby('station').fraction.min()
    inv=inv[inv.station.isin(candidates[candidates>=.9].index)].sort_values('station')
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(station_job,row) for row in inv.to_dict('records')]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%20==0: print(f'AUDIT {len(results)}/{len(inv)}',flush=True)
    by={r['station']:r for r in results}
    inv['minimum_train_coverage']=[min(by[s]['coverage'][str(y)] for y in (2016,2017,2018)) for s in inv.station]
    inv=inv[inv.minimum_train_coverage>=.9].copy().reset_index(drop=True)
    transform=Transformer.from_crs(4326,CRS,always_xy=True)
    inv['x'],inv['y']=transform.transform(inv.lon.to_numpy(),inv.lat.to_numpy())
    tree=cKDTree(inv[['x','y']]); pairs=tree.query_pairs(1000,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(inv),len(inv)))
    _,groups=connected_components(graph,directed=False)
    inv['physical_group']=groups
    inv=inv.sort_values(['minimum_train_coverage','station'],ascending=[False,True]).groupby('physical_group').head(1).copy()
    inv['cellx']=np.floor(inv.x/100000).astype(int);inv['celly']=np.floor(inv.y/100000).astype(int)
    inv['center_distance']=np.hypot(inv.x-(inv.cellx+.5)*100000,inv.y-(inv.celly+.5)*100000)
    inv['preferred']=inv.minimum_train_coverage>=.95
    source=inv.sort_values(['preferred','minimum_train_coverage','center_distance','station'],ascending=[False,False,True,True]).groupby(['cellx','celly']).head(1).index
    inv['role']='train';inv.loc[source,'role']='input';inv['cohort']='dispersed'
    rest=inv[inv.role=='train'].copy()
    rest['gx']=np.floor(rest.x/50000).astype(int);rest['gy']=np.floor(rest.y/50000).astype(int)
    blocks=[list(g.index) for _,g in rest.groupby(['gx','gy']) if len(g)>=4]
    dense_members={i for ids in blocks for i in ids}
    inv.loc[list(dense_members),'cohort']='dense'
    rng=np.random.default_rng(20261117);rng.shuffle(blocks)
    target=round(len(rest)*.075)
    used=set()
    for role in ['development','retrospective']:
        count=0
        for ids in blocks:
            if any(i in used for i in ids): continue
            if abs(count+len(ids)-target)>abs(count-target): continue
            inv.loc[ids,'role']=role;inv.loc[ids,'cohort']='dense';used.update(ids);count+=len(ids)
    # Dispersed holdouts must never split any remaining dense group.
    remaining=inv.index[(inv.role=='train')&~inv.index.isin(dense_members)].to_numpy();rng.shuffle(remaining)
    cursor=0
    for role in ['development','retrospective']:
        count=max(0,round(len(rest)*.15)-int((inv.role==role).sum()))
        if cursor+count>len(remaining):raise ValueError('Insufficient dispersed stations for registered role proportions')
        ids=remaining[cursor:cursor+count];inv.loc[ids,'role']=role;cursor+=count
    with (PIPE/'data/external/egm96_15.gtx').open('rb') as f:
        lat,lon,dy,dx,ny,nx=struct.unpack('>4d2i',f.read(40));g=np.frombuffer(f.read(),dtype='>f4').reshape(ny,nx)
    geoid=RegularGridInterpolator((lat+dy*np.arange(ny),lon+dx*np.arange(nx)),g)
    inv['height']=inv.ellipsoidal_height_m-geoid(inv[['lat','lon']].values)
    inv=inv.sort_values('station').reset_index(drop=True)
    noninput=inv[inv.role!='input'].copy()
    noninput['gx']=np.floor(noninput.x/50000);noninput['gy']=np.floor(noninput.y/50000)
    for _,group in noninput.groupby(['gx','gy']):
        if len(group)>=4 and group.role.nunique()!=1:raise ValueError('Dense group split across roles')
    inv.to_csv(DATA/'station_roles.csv',index=False)
    write_json(DATA/'audit.json',dict(sources=results,roles=inv.role.value_counts().to_dict(),crs=CRS,
        station_roles_sha256=sha256(DATA/'station_roles.csv'),candidate_numeric_qc='existing parser plus finite5 and ERA5PWV0..120',
        physical_group_policy='within1km connected components; highest train coverage representative',
        cohort_counts=inv.groupby(['role','cohort']).size().unstack(fill_value=0).to_dict('index'),
        dense_group_policy='All non-input 50km cells with >=4 stations indivisible; dispersed holdouts only from other cells',
        start=START,stop=STOP))
    print('AUDIT COMPLETE',inv.role.value_counts().to_dict(),flush=True)

if __name__=='__main__':main()
