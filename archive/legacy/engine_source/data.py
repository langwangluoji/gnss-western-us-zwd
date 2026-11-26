"""Allowed-column raw QC and role-isolated small/full calendar preparation."""
from pathlib import Path
import argparse
import calendar
import gzip
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist

PROJECT=Path(__file__).resolve().parents[2]
ROOT=next(p for p in PROJECT.parents if (p/'AGENTS.md').exists())
STUDY=PROJECT.parents[1]
ASSETS=ROOT/'docs/superpowers/specs/assets/20261123_diffusion_v2'
DEFAULT_DATA=PROJECT/'results/20261123_terrain_diffusion_v10_data'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    Path(path).write_bytes(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))


def require_role(role):
    if role not in ('training','validation'):
        raise PermissionError(f'{role} labels remain locked during development')


def parse_solution(section,station):
    # Forbidden variables are not converted or used for record selection.
    f=pd.read_csv(io.StringIO(section),sep=r'\s+',comment='*',header=None,
                  usecols=[0,1,5,9,10],names=['station','epoch','zwd','temperature','pressure'])
    if f.empty or not f.station.str.upper().eq(station.upper()).all():raise ValueError('Station identity mismatch')
    parts=f.epoch.str.extract(r'^(\d{2}|\d{4}):(\d{3}):(\d{5})$')
    if parts.isna().any().any():raise ValueError('Invalid epoch')
    year=parts[0].astype(int).to_numpy();year=np.where(year<80,year+2000,np.where(year<100,year+1900,year))
    day=parts[1].astype(int).to_numpy();sec=parts[2].astype(int).to_numpy()
    on=(day>=1)&(day<=np.array([366 if calendar.isleap(int(y)) else 365 for y in year]))&(sec>=0)&(sec<86400)&(sec%3600==0)
    base={y:int(np.datetime64(f'{y:04d}-01-01','h').astype('i8')) for y in np.unique(year)}
    hours=np.array([base[y] for y in year])+(day-1)*24+sec//3600
    values=f[['zwd','temperature','pressure']].apply(pd.to_numeric,errors='coerce').to_numpy(dtype='f4')
    before=np.isfinite(values).sum(0)
    for j,(lo,hi) in enumerate(((0,700),(180,340),(400,1100))):
        ok=on&np.isfinite(values[:,j])&(values[:,j]>=lo)&(values[:,j]<=hi);values[~ok,j]=np.nan
    frame=pd.DataFrame(values,index=hours)
    duplicate=frame.index.duplicated(keep=False);conflicts=0
    if duplicate.any():
        for h,g in frame[duplicate].groupby(level=0):
            bad=g.nunique(dropna=False).to_numpy()>1
            if bad.any():frame.loc[h,np.flatnonzero(bad)]=np.nan;conflicts+=1
    frame=frame[~frame.index.duplicated(keep='first')].sort_index()
    return frame.index.to_numpy(),frame.to_numpy(dtype='f4'),dict(records=len(f),non_hour=int((~on).sum()),
        duplicate_conflicts=conflicts,finite_before=before.tolist(),finite_after=np.isfinite(frame).sum(0).tolist())


def read_raw(station,wanted):
    path=ROOT/'research_pipeline/data/pwv_data'/f'{station.lower()}.pwv.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:text=f.read() # EOF checks gzip CRC.
    prefix,body=text.split('+TROP/SOLUTION',1)
    section=body.split('-TROP/SOLUTION',1)[0]
    header=next((l for l in section.splitlines() if l.startswith('*') and 'TROTOT' in l),'')
    expected=['SITE','EPOCH','TROTOT','STDDEV','PWV','TROWET','TRODRY','ERAZTD','ERAPWV','TEMP','PRESS','TM']
    if [v.strip('_') for v in header.lstrip('*').split()]!=expected:raise ValueError(f'Unknown columns {station}')
    h,v,meta=parse_solution(section,station)
    idx=np.searchsorted(h,wanted);ok=(idx<len(h))&(h[np.minimum(idx,len(h)-1)]==wanted)
    out=np.full((*wanted.shape,3),np.nan,dtype='f4');out[ok]=v[idx[ok]]
    time_system=next((l.strip() for l in prefix.splitlines() if 'TIME SYSTEM' in l),'UNSPECIFIED')
    return out,dict(station=station,sha256=sha(path),time_system=time_system,**meta)


def calendar_hours(years,days):
    return np.array([int(np.datetime64(f'{y}-{m:02d}-{d:02d}','h').astype('i8'))+np.arange(-12,24)
                     for y in years for m in range(1,13) for d in days])


def static_geometry(frame):
    g=np.load(STUDY/'02_data/processed/gnss_field_server_v1/geometry.npz')
    z=g['height'];gy,gx=np.gradient(z,25000.);rough=np.sqrt(np.maximum(uniform_filter(z*z,5)-uniform_filter(z,5)**2,0))
    xy=frame[['x','y']].to_numpy();uv=(xy-g['origin'])/25000-.5
    # Terrain at boundary uses explicit nearest grid support; record boundary flag.
    ix=np.clip(np.rint(uv[:,0]).astype(int),0,z.shape[1]-1);iy=np.clip(np.rint(uv[:,1]).astype(int),0,z.shape[0]-1)
    h=frame.height.to_numpy();r=rough[iy,ix]
    static=np.c_[xy/1e6,h/3000,gx[iy,ix]/.1,gy[iy,ix]/.1,r/500,(h-z[iy,ix])/1000].astype('f4')
    geo=np.c_[xy/1000,h/1000,r/500].astype('f4')
    return static,geo


def prepare(out=DEFAULT_DATA,workers=4,train_days=(8,18,25)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'contract.json').exists():raise FileExistsError('Use a new data directory; existing contract is immutable')
    allstations=pd.read_csv(ASSETS/'station_assignment_proposal.csv')
    tr=allstations[allstations.new_role=='training'].reset_index(drop=True)
    va=allstations[allstations.new_role=='validation'].reset_index(drop=True)
    # The historical contract uses three anchor days.  The expanded A+B
    # training pack uses every valid day while keeping the 2018 validation
    # contract unchanged; this is controlled explicitly by train_days.
    if train_days == 'all':
        th=np.concatenate([np.array([int(np.datetime64(f'{y}-{m:02d}-{d:02d}','h').astype('i8'))+np.arange(-12,24)
                                    for y in (2016,2017) for m in range(1,13)
                                    for d in range(1,calendar.monthrange(y,m)[1]+1)])], axis=0)
    else:
        th=calendar_hours([2016,2017],train_days)
    vh=calendar_hours([2018],[1])
    wanted=np.unique(np.r_[th.ravel(),vh.ravel()]);tidx=np.searchsorted(wanted,th);vidx=np.searchsorted(wanted,vh)
    train=np.full((*th.shape,len(tr),3),np.nan,dtype='f4')
    vc=np.full((*vh.shape,len(tr),3),np.nan,dtype='f4');vy=np.full((*vh.shape,len(va)),np.nan,dtype='f4')
    sources=[]
    jobs=[('training',i,s) for i,s in enumerate(tr.station)]+[('validation',i,s) for i,s in enumerate(va.station)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(read_raw,s,wanted if role=='training' else np.unique(vh)):(role,i,s) for role,i,s in jobs}
        for count,f in enumerate(as_completed(fs),1):
            role,i,s=fs[f];require_role(role);v,meta=f.result();sources.append(dict(role=role,**meta))
            if role=='training':train[:,:,i]=v[tidx];vc[:,:,i]=v[vidx]
            else:vy[:,:,i]=v[np.searchsorted(np.unique(vh),vh),0]
            if count%20==0:print(json.dumps(dict(stage='raw_qc',done=count,total=len(jobs))),flush=True)
    mean=np.nanmean(train,(0,1,2));std=np.maximum(np.nanstd(train,(0,1,2)),1.)
    st,ge=static_geometry(tr);vs,vg=static_geometry(va)
    np.savez_compressed(out/'data.npz',train=train,val_context=vc,val_y=vy,train_hours=th,val_hours=vh,
                        mean=mean,std=std,static=st,geo=ge,val_static=vs,val_geo=vg,
                        station_ids=tr.station.to_numpy(dtype='U'),val_ids=va.station.to_numpy(dtype='U'),
                        groups=tr.physical_group.to_numpy(),val_groups=va.physical_group.to_numpy())
    write_json(out/'contract.json',dict(status='ACTIVE_LOCAL_SCREENING_ALLOWED_RECORD_QC',split_sha256=sha(ASSETS/'station_assignment_proposal.csv'),
        data_sha256=sha(out/'data.npz'),code_sha256=sha(__file__),sources=sorted(sources,key=lambda x:x['station']),
        inputs=['surface_ZWD_mm','temperature_K','pressure_hPa'],excluded_columns=['PWV','ERAPWV','TM'],
        training_stations=len(tr),validation_stations=len(va),confirmation_labels_read=False,C_D_labels_read=False,
        station_cohort='Inherited old A+B station selection; historical PWV-completeness selection persists at station level',
        record_qc='Independent ZWD0..700, T180..340, P400..1100; hourly timestamps; per-variable conflicting duplicates removed',
        time='nominal product hour; per-source header recorded',train_sequences=len(th),validation_sequences=len(vh),
        train_finite_fraction=np.isfinite(train).mean((0,1,2)).tolist(),val_label_fraction=float(np.isfinite(vy).mean()),
        normalisation=dict(mean=mean.tolist(),std=std.tolist(),scope='training410, 2016/17 selected hours'),
        terrain='25km nearest-node terrain; station height retained; no fine DEM claim'))
    print(json.dumps(dict(stage='data_ready',path=str(out),train_shape=train.shape)),flush=True)


class Dataset:
    def __init__(self,path=DEFAULT_DATA,device='cuda'):
        self.path=Path(path);self.contract=json.loads((self.path/'contract.json').read_text(encoding='utf-8'))
        if sha(self.path/'data.npz')!=self.contract['data_sha256']:raise ValueError('Data hash changed')
        a=np.load(self.path/'data.npz');self.a={k:a[k] for k in a.files};self.device=device
        self.mean=self.a['mean'];self.std=self.a['std']

    def layouts(self,fraction=.2):
        return [np.sort(np.random.default_rng(8120+i).choice(len(self.a['static']),round(len(self.a['static'])*fraction),replace=False)) for i in range(3)]

    def batch(self,row,t,context,query,role='training',teacher=False,query_geometry=None):
        require_role(role);a=self.a;c=np.asarray(context);q=np.asarray(query)
        if role=='training':
            assert not set(a['groups'][c])&set(a['groups'][q]),'Context target physical overlap'
            if query_geometry is not None:raise ValueError('External queries are inference-only')
            raw=a['train'][row,t-11:t+1][:,c]
            y=a['train'][row,t,q,0];qs=a['static'][q];qg=a['geo'][q];hours=a['train_hours'][row,t-11:t+1]
        else:
            raw=a['val_context'][row,t-11:t+1][:,c];hours=a['val_hours'][row,t-11:t+1]
            if query_geometry is None:y=a['val_y'][row,t,q];qs=a['val_static'][q];qg=a['val_geo'][q]
            else:
                qs,qg=query_geometry;y=np.full(len(qs),np.nan)
                if qs.shape!=(len(q),7) or qg.shape!=(len(q),4):raise ValueError('External geometry dimensions')
        assert len(raw)==12
        cs=a['static'][c];cg=a['geo'][c];valid=np.isfinite(raw);norm=np.where(valid,(raw-self.mean)/self.std,0.)
        age=np.zeros_like(raw)
        for j in range(12):age[j]=np.where(valid[j],0,(age[j-1]+1 if j else 12))
        def fourier(st):
            phase=st[:,:2,None]*1000/np.array([50,100,200,400])*2*np.pi
            return np.concatenate([np.sin(phase),np.cos(phase)],-1).reshape(len(st),16)
        times=np.stack([np.sin(hours%24/24*2*np.pi),np.cos(hours%24/24*2*np.pi),np.sin(hours/(24*365.25)*2*np.pi),np.cos(hours/(24*365.25)*2*np.pi)],-1)
        dist=cdist(cg[:,:2],cg[:,:2]);np.fill_diagonal(dist,np.inf)
        support=np.c_[np.minimum(dist.min(1),600)/100,np.log1p((dist<150).sum(1))]
        cf=np.concatenate([norm,valid.astype('f4'),np.log1p(age),np.broadcast_to(cs,(12,*cs.shape)),
                           np.broadcast_to(fourier(cs),(12,len(cs),16)),np.broadcast_to(times[:,None],(12,len(cs),4)),
                           np.broadcast_to(support,(12,*support.shape))],-1)
        d=cdist(qg[:,:2],cg[:,:2]);weather=[]
        for radius in (50,150,300):
            w=np.exp(-.5*(d/radius)**2)*(d<=3*radius)
            den=w@valid[-1,:,1:].astype('f4');avg=(w@(norm[-1,:,1:]*valid[-1,:,1:]))/np.maximum(den,1e-8)
            weather.extend([avg,np.log1p(den)])
        active=valid[-1,:,0];near=np.min(np.where(active[None],d,np.inf),1)
        weather.append(np.c_[np.minimum(near,600)/100,np.log1p(((d<150)&active).sum(1))])
        qf=np.concatenate([qs,fourier(qs),np.broadcast_to(times[-1],(len(q),4)),*weather],-1)
        hist=np.zeros((6,len(q),3),dtype='f4')
        if teacher and role=='training':
            h=a['train'][row,t-6:t][:,q,0];m=np.isfinite(h);hist[...,0]=np.where(m,(h-self.mean[0])/self.std[0],0);hist[...,1]=m;hist[...,2]=1
        arrays=dict(context=cf,cgeo=cg,query=qf,qgeo=qg,history=hist)
        out={k:torch.tensor(v,dtype=torch.float32,device=self.device)[None] for k,v in arrays.items()}
        out['target']=torch.tensor(np.nan_to_num((y-self.mean[0])/self.std[0]),dtype=torch.float32,device=self.device)[None]
        out['mask']=torch.tensor(np.isfinite(y),device=self.device)[None]
        return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_DATA);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--all-train-days',action='store_true',help='expand A+B training windows to every valid day in 2016-2017')
    args=p.parse_args();prepare(args.out,args.workers,'all' if args.all_train_days else (8,18,25))
