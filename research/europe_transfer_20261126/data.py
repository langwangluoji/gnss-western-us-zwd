"""European Model4 adapter. Only explicitly allowed columns enter the new model."""
from pathlib import Path
import hashlib
import json
import urllib.request

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'联合信号分析与深度学习的高精度对流层建模/Data/model/Model4/Model4_FFT_SH_Patent'
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
CACHE=HERE/'cache'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dry_pressure(pressure_hpa,temperature_kelvin,height_difference_m):
    return pressure_hpa*np.exp(-9.80665*height_difference_m/(287.05*temperature_kelvin))


def neural_cutoff(hours,version=2):
    if version not in (1,2):raise ValueError('unknown training protocol')
    end=np.datetime64('2024-01-01' if version==2 else '2023-01-01','h').astype('i8')
    return int(np.searchsorted(hours,end))


def dense_values(hours,station_index,values,nstations):
    hours=np.asarray(hours,dtype=np.int64)
    if np.any(hours%6):raise ValueError('off six-hour UTC grid')
    ti=(hours-hours.min())//6
    keys=ti*nstations+station_index
    if len(np.unique(keys))!=len(keys):raise ValueError('duplicate station epoch')
    out=np.full((int(ti.max())+1,nstations,values.shape[-1]),np.nan,dtype=np.float32)
    out[ti,station_index]=values
    return out


def station_roles(xy):
    # Morton spatial order; randomization occurs only inside neighboring groups.
    scale=(xy-xy.min(0))/(np.ptp(xy,axis=0)+1e-9)
    coords=np.minimum((scale*65535).astype(np.uint64),65535)
    codes=np.zeros(len(xy),dtype=np.uint64)
    for bit in range(16):
        codes|=((coords[:,0]>>bit)&1)<<(2*bit)
        codes|=((coords[:,1]>>bit)&1)<<(2*bit+1)
    order=np.argsort(codes,kind='stable');result=np.full(len(xy),'train',dtype='<U8')
    rng=np.random.default_rng(20261126)
    for group in np.array_split(order,20):
        selected=rng.permutation(group);nd=int(round(len(group)*.15))
        result[selected[:nd]]='dev';result[selected[nd:2*nd]]='confirm'
    return result


def site_roles(xy,old):
    parent=np.arange(len(xy))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in sorted(cKDTree(xy).query_pairs(1.0)):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    groups=np.array([find(i) for i in range(len(xy))])
    registered=np.unique(groups[old]);centers=np.stack([xy[groups==g].mean(0) for g in registered])
    gr=station_roles(centers);mapping=dict(zip(registered,gr))
    # A 2024-only station stays outside training, even if it aliases an old site.
    roles=np.array([mapping.get(g,'new2024') if o else 'new2024' for g,o in zip(groups,old)])
    return roles,groups


def read_columns(path,cols):
    with h5py.File(path,'r') as f:
        return pd.DataFrame({c:f[c].asstr()[:] if f[c].dtype.kind=='O' else f[c][:] for c in cols})


def coast_distances(points,transformer):
    url='https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson'
    path=CACHE/'ne_50m_coastline.geojson'
    if not path.exists():
        # Reuse byte-identical source if an existing US cache is available.
        cached=list((ENGINE/'results/20261125_temporal_continuous_data').glob('coast_source/ne_50m_coastline.geojson'))
        if cached:path.write_bytes(cached[0].read_bytes())
        else:
            with urllib.request.urlopen(url,timeout=60) as response:path.write_bytes(response.read())
    lines=[]
    for feature in json.loads(path.read_text(encoding='utf-8'))['features']:
        geom=feature['geometry'];coordinates=[geom['coordinates']] if geom['type']=='LineString' else geom['coordinates']
        for line in coordinates:
            a=np.asarray(line)
            if ((a[:,0]>-40)&(a[:,0]<60)&(a[:,1]>10)&(a[:,1]<85)).any():
                x,y=transformer.transform(a[:,0],a[:,1]);lines.append(np.c_[x,y]/1000)
    # Exact projected segment distance, not nearest vertex.
    a=np.concatenate([line[:-1] for line in lines]);b=np.concatenate([line[1:] for line in lines]);d=b-a
    length2=np.maximum((d*d).sum(1),1e-12);answer=[]
    for p in points:
        t=np.clip(((p-a)*d).sum(1)/length2,0,1)
        answer.append(np.sqrt(((p-a-t[:,None]*d)**2).sum(1)).min())
    return np.asarray(answer,dtype='f4'),dict(url=url,sha256=sha(path),crs='EPSG:3035',distance='minimum projected segment km',license='Natural Earth public domain')


def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'data.npz').exists():raise FileExistsError('use a new version for repacking')
    paths=[SOURCE/'data/euref_train_2020_2023.h5',SOURCE/'data/euref_test_2024.h5']
    metadata=['station_phys','datetime','Latitude','Longitude','Height','era5_orography_m']
    frames=[read_columns(p,metadata) for p in paths]
    meta=pd.concat(frames).groupby('station_phys',sort=True)[metadata[2:]].median().reset_index()
    old=np.isin(meta.station_phys,frames[0].station_phys.unique())
    transformer=Transformer.from_crs(4326,3035,always_xy=True)
    x,y=transformer.transform(meta.Longitude.to_numpy(),meta.Latitude.to_numpy());xy=np.c_[x,y]/1000
    role,groups=site_roles(xy,old)
    meta['role']=role;meta['site_group']=groups;meta['x_km']=xy[:,0];meta['y_km']=xy[:,1]
    # Freeze metadata split before reading any ZWD labels.
    meta.to_csv(HERE/'stations.csv',index=False,lineterminator='\n')
    write_json(HERE/'split_lock.json',dict(version=2,seed=20261126,source_metadata=[sha(p) for p in paths],stations_sha256=sha(HERE/'stations.csv'),roles=meta.role.value_counts().to_dict(),split_uses_labels=False,labels_previously_loaded_for_audit=True,site_group_radius_km=1))
    coordinates=xy-xy.mean(0)
    # Coarse terrain is reconstructed only from ERA5 static orography.
    lo=coordinates.min(0)-100;hi=coordinates.max(0)+100
    gx=np.linspace(lo[0],hi[0],32);gy=np.linspace(lo[1],hi[1],32)
    xx,yy=np.meshgrid(gx,gy);gridxy=np.c_[xx.ravel(),yy.ravel()]
    topo=RBFInterpolator(coordinates[old],meta.era5_orography_m.to_numpy()[old],neighbors=20,smoothing=1000)(gridxy).reshape(32,32)
    topo=np.clip(topo,-100,4000);dy,dx=np.gradient(topo,(gy[1]-gy[0])*1000,(gx[1]-gx[0])*1000)
    rough=np.sqrt(np.maximum(uniform_filter(topo*topo,5)-uniform_filter(topo,5)**2,0))
    _,ix=cKDTree(gridxy).query(coordinates)
    heights=meta.Height.to_numpy();orog=meta.era5_orography_m.to_numpy()
    static=np.c_[coordinates/1000,heights/3000,dx.ravel()[ix]/.1,dy.ravel()[ix]/.1,rough.ravel()[ix]/500,(heights-orog)/1000]
    gridstatic=np.c_[gridxy/1000,topo.ravel()/3000,dx.ravel()/.1,dy.ravel()/.1,rough.ravel()/500,np.zeros(len(gridxy))]
    coasts,receipt=coast_distances(np.r_[xy,gridxy+xy.mean(0)],transformer)
    static=np.c_[static,np.log1p(coasts[:len(meta)])/6,coasts[:len(meta)]/500]
    gridstatic=np.c_[gridstatic,np.log1p(coasts[len(meta):])/6,coasts[len(meta):]/500]
    pieces=[];allhours=[];counts=[]
    for path in paths:
        d=read_columns(path,['station_phys','datetime','zwd','temperature','surface_pressure_era5_hpa','height_diff_m'])
        hours=pd.to_datetime(d.datetime).to_numpy().astype('datetime64[h]').astype('i8')
        index=pd.Index(meta.station_phys).get_indexer(d.station_phys)
        t=d.temperature.to_numpy()+273.15
        values=np.c_[d.zwd.to_numpy()*1000,t,dry_pressure(d.surface_pressure_era5_hpa.to_numpy(),t,d.height_diff_m.to_numpy())]
        a=dense_values(hours,index,values,len(meta));pieces.append(a)
        allhours.append(np.arange(hours.min(),hours.max()+1,6));counts.append(dict(rows=len(d),epochs=len(a),source=str(path),sha256=sha(path)))
    hours=np.concatenate(allhours);values=np.concatenate(pieces)
    if not np.all(np.diff(hours)==6):raise ValueError('files not contiguous')
    np.savez_compressed(CACHE/'data.npz',values=values,hours=hours,static=static.astype('f4'),grid_static=gridstatic.astype('f4'),xy=coordinates.astype('f4'),grid_xy=gridxy.astype('f4'),grid_shape=[32,32],lo=lo,hi=hi,role=role,groups=groups,stations=meta.station_phys.to_numpy(dtype='<U4'),height=heights.astype('f4'),grid_height=topo.ravel().astype('f4'))
    nearest=cKDTree(coordinates).query(coordinates,k=2)[0][:,1]
    audit=dict(sources=counts,stations=len(meta),shared_stations=int(len(set(frames[0].station_phys)&set(frames[1].station_phys))),roles=meta.role.value_counts().to_dict(),coast=receipt,terrain='ERA5 static orography sampled at stations, RBF20-neighbor smoothing1000 on32x32; coarse, not SRTM',source_label='Model4 zwd[m] converted to mm; inherits humidity-informed label preparation and row selection',allowed_columns=['zwd','temperature','surface_pressure_era5_hpa','height_diff_m',*metadata],pressure='dry static exp(-g*dz/Rd/T) without WVP',temperature='C+273.15',hours=6,history_span_hours=90,nearest_station_min_km=float(nearest.min()),close_pairs_under1km=int((nearest<1).sum()),pack_sha256=sha(CACHE/'data.npz'),missing_fraction=np.isnan(values).mean((0,1)).tolist())
    write_json(HERE/'data_audit.json',audit);print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':prepare()
