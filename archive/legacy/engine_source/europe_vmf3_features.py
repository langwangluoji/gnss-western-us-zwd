"""VMF3 interpolation and reference-only background feature construction."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from maintenance.europe_transfer_20261126.data import HERE as PARENT,CACHE as OLD_CACHE,SOURCE,sha,write_json
from maintenance.europe_transfer_20261126.native import inputs,i1_features

HERE=Path(__file__).resolve().parent
CACHE=HERE/'cache'
sys.path.insert(0,str(SOURCE/'src'))
from model4_fft_pipeline import vmf3_background as vmf


def exact_epochs(requested,available,values):
    index=np.searchsorted(available,requested);safe=np.minimum(index,len(available)-1)
    valid=(index<len(available))&(available[safe]==requested)
    result=np.full((len(requested),*values.shape[1:]),np.nan,dtype='f4');result[valid]=values[index[valid]]
    return result


def reference_pressure(raw,cxy,cheight,qxy,qheight,fallback):
    """Use reference T/P only; vertically transfer before spatial averaging."""
    valid=np.isfinite(raw[...,1])&np.isfinite(raw[...,2])&(raw[...,1]>0)&(raw[...,2]>0)
    temperature=np.where(valid,raw[...,1],280)
    pressure=np.where(valid,raw[...,2],0)
    distance=np.sqrt(((qxy[:,None]-cxy[None])**2).sum(-1))
    answer=np.asarray(fallback,dtype='f8').copy();present=np.zeros(answer.shape,dtype=bool)
    for begin in range(0,len(raw),64):
        end=min(begin+64,len(raw));t=temperature[begin:end]
        lifted=pressure[begin:end,None,:]*np.exp(9.80665*(cheight[None,None]-qheight[None,:,None])/(287.05*t[:,None]))
        for radius in (150,300):
            w=np.exp(-.5*(distance/radius)**2)[None]*valid[begin:end,None]
            den=w.sum(-1);take=(~present[begin:end])&(den>1e-8)
            estimated=(w*lifted).sum(-1)/np.maximum(den,1e-8)
            answer[begin:end]=np.where(take,estimated,answer[begin:end]);present[begin:end]|=take
    return answer.astype('f4'),present


def background_features(zhd,zwd,dry_zhd,field_mm):
    native=zwd*1000;aligned=(zhd+zwd-dry_zhd)*1000
    valid=np.isfinite(native)&np.isfinite(aligned)
    value=np.stack([native/100,aligned/100,(aligned-field_mm)/20,valid.astype('f4')],-1)
    return np.where(valid[...,None],value,0).astype('f4')


def interpolate():
    CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/'vmf3_sites.npz'
    if path.exists():raise FileExistsError(path)
    source=SOURCE/'paper_experiments/E17_vmf3_grid_interp/results/_cache';extra=SOURCE/'data/VMF3_grid_ref/pipeline_cache';oro=SOURCE/'data/VMF3_grid_ref/orography_ell_1x1'
    cache=vmf.Vmf3GridCache(source,extra,oro);meta=pd.read_csv(PARENT/'stations.csv');a=np.load(OLD_CACHE/'data.npz')
    idx,wl,wo=vmf.four_neighbour_indices(meta.Latitude.to_numpy(),meta.Longitude.to_numpy());local=cache.local_indices(idx);oro4=cache.orography[idx]
    zhd=[];zwd=[];height=meta.Height.to_numpy()[None,:,None];latitude=meta.Latitude.to_numpy()[None,:,None]
    for start in range(0,len(cache.epochs),128):
        data=cache.values[start:start+128];h=vmf.lift_zhd(data[:,:,0][:,local],latitude,height,oro4[None]);w=vmf.lift_zwd(data[:,:,1][:,local],height,oro4[None])
        zhd.append(vmf._bilinear_3d(h,wl,wo));zwd.append(vmf._bilinear_3d(w,wl,wo))
    grid=np.stack([np.concatenate(zhd),np.concatenate(zwd)],-1)
    values=exact_epochs(a['hours'],cache.epochs.astype('datetime64[h]').astype('i8'),grid)
    np.savez_compressed(path,values=values,hours=a['hours'])
    files=[f for root in (source,extra) for f in root.glob('*.npy')]+[oro,SOURCE/'src/model4_fft_pipeline/vmf3_background.py']
    write_json(HERE/'vmf3_data_receipt.json',dict(product='VMF3_FC1x1 forecast archive',source_url='https://vmf.geo.tuwien.ac.at/trop_products/GRID/1x1/VMF3/VMF3_FC',epochs=len(values),stations=len(meta),available_epochs=int(np.isfinite(values).all((1,2)).sum()),missing_valid_times=[str(x) for x in a['hours'][~np.isfinite(values).all((1,2))].astype('datetime64[h]')],missing_policy='NaN and mask; no temporal interpolation',sources={str(f):sha(f) for f in files},pack_sha256=sha(path),method='original per-gridpoint vertical lift then four-point bilinear; meters retained',future_valid_times_used=False))
    print('VMF3 station interpolation complete',flush=True)


def prepare(seed):
    a=dict(np.load(OLD_CACHE/'data.npz'));meta=pd.read_csv(PARENT/'stations.csv');values=np.load(CACHE/'vmf3_sites.npz')['values']
    old=OLD_CACHE/f'native_v2_s{seed}';coef=np.load(old/'i1.npy');out=CACHE/f's{seed}';out.mkdir(exist_ok=False);receipts=[]
    paths=[*sorted(old.glob('train_*.npz')),old/'dev.npz',old/'test.npz']
    for path in paths:
        cache=dict(np.load(path));c=cache['context'];start=int(cache['start']);stop=start+len(cache['base'])
        pool=np.flatnonzero(a['role']=='train')
        q=(pool[~np.isin(a['groups'][pool],a['groups'][c])] if path.name.startswith('train_') else np.flatnonzero(a['role']=='dev' if path.stem=='dev' else a['role']!='train'))
        field=(cache['base']+i1_features(cache['innovation'])@coef)[:,q]
        zhd=values[start:stop][:,q,0];zwd=values[start:stop][:,q,1]
        lat=meta.Latitude.to_numpy()[q][None];height=meta.Height.to_numpy()[q]
        factor=1-.00266*np.cos(2*np.deg2rad(lat))-.00028*height[None]/1000
        fallback=zhd*factor/.0022768
        pressure,supported=reference_pressure(a['values'][start:stop][:,c],a['xy'][c],a['height'][c],a['xy'][q],a['height'][q],fallback)
        dry=vmf.saastamoinen_zhd_m(pressure,lat,height[None]);feature=background_features(zhd,zwd,dry,field)
        condition=np.zeros((len(field),len(q),111),dtype='f4')
        for i in range(15,len(field),64):
            pos=np.arange(i,min(i+64,len(field)));condition[pos]=inputs(cache,a,pos,q,coef)['condition']
        np.savez_compressed(out/path.name,condition=condition,vmf=feature,field=field.astype('f4'),query=q,start=start,fit_start=cache['fit_start'])
        receipts.append(dict(file=path.name,source_sha256=sha(path),query_count=len(q),epochs=len(field),reference_pressure_support_fraction=float(supported.mean())))
        print(f'VMF3 feature cache seed{seed}: {path.name}',flush=True)
    write_json(HERE/f'features_s{seed}.json',dict(seed=seed,parent_field_sha256=sha(PARENT/f'runs/base_v2_s{seed}/best.pt'),parent_i1_sha256=sha(old/'i1.npy'),vmf3_sha256=sha(CACHE/'vmf3_sites.npz'),feature_order=['native_mm/100','aligned_referenceP_mm/100','aligned_minus_I1_mm/20','vmf3_valid'],aligned_pressure='reference-only T/P, dry vertical transfer,150km kernel then300km fallback; if no support VMF3 own hydrostatic term',target_observations_read_as_conditions=False,segments=receipts))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--interpolate',action='store_true');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.interpolate:interpolate()
    else:prepare(args.seed)
