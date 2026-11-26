"""Fixed-reference point results and paired probability attribution summaries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json
from .model import BackgroundCorrection,batch
from .prob_evaluate import KINDS


def score(mean,target,mask):
    e=(mean-target)[mask].astype('f8');counts=mask.sum(0);squared=np.where(mask,(mean-target)**2,0).sum(0);active=counts>0
    return dict(records=int(mask.sum()),stations=int(active.sum()),micro_rmse_mm=float(np.sqrt(np.mean(e*e))),station_macro_rmse_mm=float(np.sqrt(squared[active]/counts[active]).mean()))


@torch.no_grad()
def point():
    torch.set_num_threads(3);old=np.load(OLD_CACHE/'comparison_arrays.npz');a=np.load(OLD_CACHE/'data.npz');target=old['target'];base={};q=None
    for seed in (42,17):
        cache=dict(np.load(CACHE/f's{seed}/test.npz'));first=int(cache['fit_start']);q=cache['query']
        control=BackgroundCorrection(False).cuda().eval();control.load_state_dict(torch.load(HERE/f'runs/control_s{seed}/best.pt',weights_only=True));control.time.flatten_parameters();pred=[]
        for p in range(first,len(cache['field']),16):pred.append(control(*batch(cache,np.arange(p,min(p+16,len(cache['field']))))).cpu().numpy())
        control_values=np.concatenate(pred);source=np.load(CACHE/f'diffusion_s{seed}/test.npz');values=source['mean'][first:,:,0]
        for name,x in [('control',control_values),('vmf_field',values),('vmf_native',cache['vmf'][first:,:,0]*100),('vmf_aligned',cache['vmf'][first:,:,1]*100)]:
            dense=np.full(target.shape,np.nan);dense[:,q]=x.astype('f8');base[f'{name}_s{seed}']=dense
    models=dict(old_cnn=old['original'],old_I1=old['ours'])
    for name in ('control','vmf_field','vmf_native','vmf_aligned'):models[name]=(base[f'{name}_s42']+base[f'{name}_s17'])/2
    cohorts=dict(confirm=a['role']=='confirm',dev=a['role']=='dev',new=a['role']=='new2024',all_holdout=a['role']!='train');report={}
    for cohort,role in cohorts.items():
        mask=old['valid']&role[None]
        for value in models.values():
            if not np.isfinite(value[mask]).all():raise ValueError('point model misses frozen valid records')
        report[cohort]={name:score(value,target,mask) for name,value in models.items()}
    write_json(HERE/'point_results.json',dict(cohorts=report,mean_model='new frozen VMF3 field; before diffusion samples',cnn_reference='previous fixed273-station CNN; no continued training',source_probability_lock_sha256=sha(HERE/'probability_lock.json')))
    meta=pd.read_csv(PARENT/'stations.csv');rows=[];role=a['role'];mask=old['valid']
    for j in np.flatnonzero(mask.any(0)):
        row=dict(station_phys=meta.station_phys.iloc[j],role=role[j],Latitude=meta.Latitude.iloc[j],Longitude=meta.Longitude.iloc[j],records=int(mask[:,j].sum()))
        for name,value in models.items():row[name]=float(np.sqrt(np.mean((value[:,j][mask[:,j]]-target[:,j][mask[:,j]])**2)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'point_station_rmse.csv',index=False,lineterminator='\n')
    np.savez_compressed(CACHE/'point_delivery.npz',hours=old['hours'],target=target,valid=old['valid'],**models)
    print(json.dumps(report['confirm']),flush=True)


def probability():
    a=np.load(OLD_CACHE/'data.npz');months=pd.to_datetime(np.load(OLD_CACHE/'comparison_arrays.npz')['hours'].astype('datetime64[h]')).month.to_numpy()
    metrics={};monthly={};rows=[]
    for view,kinds in KINDS.items():
        for kind in kinds:
            key=f'{view}_{kind}';values=[];sums=[];counts=[]
            for seed in (42,17):
                folder=HERE/'evaluation'/f's{seed}'/view/kind;m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'));values.append(m)
                with np.load(folder/'evaluation.npz') as d:
                    crps=d['crps'].astype('f8');q=d['query'];keep=a['role'][q]=='confirm';crps=np.where(keep[None,:,None],crps,np.nan)
                    sums.append(np.stack([np.nansum(crps[months==month],(0,1)) for month in range(1,13)]));counts.append(np.stack([np.isfinite(crps[months==month]).sum((0,1)) for month in range(1,13)]))
            metrics[key]={name:np.mean([v[name] for v in values],axis=0).tolist() if isinstance(values[0][name],list) else float(np.mean([v[name] for v in values])) for name in values[0]}
            monthly[key]=dict(sum=np.sum(sums,axis=0),count=np.sum(counts,axis=0))
            for h in range(7):rows.append(dict(model=key,lead_hours=h*6,fair_crps_mm=metrics[key]['fair_crps_mm_by_lead'][h],rank3_30_coverage=metrics[key]['rank3_30_coverage'][h],point_rmse_mm=metrics[key]['point_rmse_mm_by_lead'][h]))
    contrasts={};rng=np.random.default_rng(261126);ix=rng.integers(0,12,(2000,12))
    for left,right in [('base_local','vmf_local'),('vmf_gaussian','vmf_gp'),('vmf_gp','vmf_local'),('vmf_affine','vmf_local')]:
        if not np.array_equal(monthly[left]['count'],monthly[right]['count']):raise ValueError('paired probability masks differ')
        diff=monthly[right]['sum']-monthly[left]['sum'];counts=monthly[left]['count'];estimate=diff.sum(0)/counts.sum(0)
        boot=diff[ix].sum(1)/counts[ix].sum(1);ci=np.quantile(boot,[.025,.975],axis=0)
        contrasts[f'{right}_minus_{left}']=dict(delta_fair_crps_mm_by_lead=estimate.tolist(),month_block95_low=ci[0].tolist(),month_block95_high=ci[1].tolist(),sign='negative favors right model',interval_scope='fixed two training seeds and one K32 noise stream; paired month resampling; retrospective2024')
    write_json(HERE/'probability_results.json',dict(models=metrics,contrasts=contrasts,aggregation='mean of two fixed training-seed scores; not pooled64-member ensemble',coverage_rank_reference=27/33,point_mean_unchanged_across_probability_models=True))
    pd.DataFrame(rows).to_csv(HERE/'probability_scores.csv',index=False,lineterminator='\n')
    write_json(HERE/'monthly_probability_scores.json',{k:{n:v.tolist() for n,v in value.items()} for k,value in monthly.items()})
    print(json.dumps(dict(models=metrics,contrasts=contrasts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['point','probability']);args=p.parse_args()
    (point if args.stage=='point' else probability)()
