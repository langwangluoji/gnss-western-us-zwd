"""Geometry-selected contrasts and station diagnostics; no fitting or selection."""
import json
import numpy as np
import pandas as pd
from .features import HERE,CACHE,PARENT,OLD_CACHE,write_json
from .prob_evaluate import KINDS
from tools.terrain_diffusion.native_weather_evaluation import fair_crps


def run():
    a=np.load(OLD_CACHE/'data.npz');points=pd.read_csv(HERE/'point_station_rmse.csv');meta=pd.read_csv(PARENT/'stations.csv')
    source=np.load(CACHE/'diffusion_s42/test.npz');q=source['query'];confirm=np.flatnonzero(a['role'][q]=='confirm');xy=a['xy'][q]
    left,right=np.triu_indices(len(confirm),1);left,right=confirm[left],confirm[right];distance=np.sqrt(((xy[left]-xy[right])**2).sum(-1))
    pairs={};rng=np.random.default_rng(261126)
    for lo,hi in ((0,50),(50,150),(150,300)):
        choices=np.flatnonzero((distance>=lo)&(distance<hi));choices=np.sort(rng.choice(choices,min(len(choices),64),replace=False))
        pairs[f'{lo}-{hi}']=(left[choices],right[choices])
    payload={};station_rows=[]
    for view,kinds in KINDS.items():
        for kind in kinds:
            key=f'{view}_{kind}';pair_scores={name:{'sum':np.zeros(2),'count':np.zeros(2,dtype='i8')} for name in pairs};station_sum=np.zeros((len(q),2));station_count=np.zeros((len(q),2),dtype='i8')
            for seed in (42,17):
                with np.load(HERE/f'evaluation/s{seed}/{view}/{kind}/evaluation.npz') as d:
                    samples=d['samples'];truth=d['truth'];valid=d['primary'];crps=d['crps'].astype('f8')
                    for j,h in enumerate((0,1)):
                        station_sum[:,j]+=np.where(valid[:,:,h],crps[:,:,h],0).sum(0);station_count[:,j]+=valid[:,:,h].sum(0)
                        for label,(l,r) in pairs.items():
                            mask=valid[:,l,h]&valid[:,r,h]
                            y=np.where(mask,truth[:,l,h]-truth[:,r,h],np.nan)
                            score=fair_crps(samples[:,:,l,h]-samples[:,:,r,h],y)
                            pair_scores[label]['sum'][j]+=np.nansum(score);pair_scores[label]['count'][j]+=np.isfinite(score).sum()
            payload[key]={name:dict(registered_pairs=len(pairs[name][0]),fair_crps_mm_by_now_f6=np.divide(v['sum'],v['count'],out=np.full(2,np.nan),where=v['count']>0).tolist(),valid_pairs_by_now_f6=v['count'].tolist()) for name,v in pair_scores.items()}
            for i in np.flatnonzero(station_count[:,0]>0):
                station_rows.append(dict(station_phys=meta.station_phys.iloc[q[i]],model=key,current_crps_mm=station_sum[i,0]/station_count[i,0],f6_crps_mm=station_sum[i,1]/station_count[i,1] if station_count[i,1]>0 else None))
            print(f'diagnostics: {key}',flush=True)
    # JSON permits no NaN: an empty geometric bin has no score.
    for model in payload.values():
        for v in model.values():v['fair_crps_mm_by_now_f6']=[None if not np.isfinite(x) else x for x in v['fair_crps_mm_by_now_f6']]
    write_json(HERE/'spatial_contrasts.json',dict(models=payload,pair_selection='at most64fixed seeded geometry-only registered confirm pairs per distance bin',no_future_mask_used_for_pair_selection=True))
    pd.DataFrame(station_rows).to_csv(HERE/'station_probability_scores.csv',index=False,lineterminator='\n')
    names=pd.read_csv(PARENT/'priority_stations.csv').station_phys.tolist()
    points[points.station_phys.isin(names)].to_csv(HERE/'priority_station_comparison.csv',index=False,lineterminator='\n')


if __name__=='__main__':run()
