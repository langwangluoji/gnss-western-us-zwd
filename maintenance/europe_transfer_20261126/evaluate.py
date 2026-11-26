"""Frozen2023 selection and common-key full-year2024 system RMSE comparison."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from .data import HERE,CACHE,write_json,sha
from .native import make_model,inputs,monitor,i1_features,tensor_inputs


def common_mask(target,predictions,station_mask):
    valid=np.isfinite(target)&np.asarray(station_mask)[None]
    for prediction in predictions.values():valid&=np.isfinite(prediction)
    return valid


def common_rmse(target,predictions,station_mask):
    valid=common_mask(target,predictions,station_mask);count=valid.sum(0);active=count>0
    result=dict(records=int(valid.sum()),stations=int(active.sum()),models={})
    for name,prediction in predictions.items():
        error=np.where(valid,prediction-target,0).astype('f8');square=(error*error).sum(0)
        result['models'][name]=dict(micro_rmse_mm=float(np.sqrt(square.sum()/count.sum())) if count.sum() else None,station_macro_rmse_mm=float(np.sqrt(square[active]/count[active]).mean()) if active.any() else None)
    return result


def select():
    if (HERE/'selection_2023_v2.json').exists():raise FileExistsError('selection already frozen')
    torch.set_num_threads(4);a=dict(np.load(CACHE/'data.npz'));q=np.flatnonzero(a['role']=='dev');records=[]
    for seed in (42,17):
        cache=dict(np.load(CACHE/f'native_v2_s{seed}/dev.npz'));coef=np.load(CACHE/f'native_v2_s{seed}/i1.npy')
        model=make_model(len(a['static'])).cuda();model.load_state_dict(torch.load(HERE/f'runs/joint_v2_s{seed}/best.pt',weights_only=True));model.eval()
        joint=monitor(model,cache,a,coef)[0]
        origins=np.linspace(int(cache['fit_start'])+16,len(cache['base'])-8,24).astype(int)
        pos=np.concatenate([np.arange(p,p+4) for p in origins]);target=a['values'][pos[:,None]+int(cache['start']),q[None],0]
        p14=cache['base'][pos][:,q];i1=(cache['base']+i1_features(cache['innovation'])@coef)[pos][:,q]
        rmse=common_rmse(target,{'p14_gru':p14,'i1':i1},np.ones(len(q),dtype=bool))
        scores={name:v['station_macro_rmse_mm'] for name,v in rmse['models'].items()};scores['joint']=joint
        records.append(dict(seed=seed,dev_station_macro_rmse_mm=scores,joint_sha256=sha(HERE/f'runs/joint_v2_s{seed}/best.pt'),base_sha256=sha(HERE/f'runs/base_v2_s{seed}/best.pt'),i1_sha256=sha(CACHE/f'native_v2_s{seed}/i1.npy')))
    means={name:float(np.mean([r['dev_station_macro_rmse_mm'][name] for r in records])) for name in ('p14_gru','i1','joint')}
    selected=min(means,key=means.get)
    write_json(HERE/'selection_2023_v2.json',dict(version=2,year=2023,metric='24 fixed4-record blocks, spatially held-out dev station macro RMSE;2023 is inside training time pool',seeds=records,mean_scores_mm=means,selected_architecture=selected,test_2024_labels_used=False))
    print(json.dumps(dict(stage='selection_2023',selected=selected,scores=means)),flush=True)


@torch.no_grad()
def predict(seed):
    torch.set_num_threads(4);a=dict(np.load(CACHE/'data.npz'));root=CACHE/f'native_v2_s{seed}'
    path=root/'test_full.npz' if (root/'test_full.npz').exists() else root/'test.npz'
    cache=dict(np.load(path));coef=np.load(root/'i1.npy');q=np.flatnonzero(a['role']!='train');c=cache['context']
    model=make_model(len(a['static'])).cuda();model.load_state_dict(torch.load(HERE/f'runs/joint_v2_s{seed}/best.pt',weights_only=True));model.eval()
    cs=torch.tensor(a['static'][c],device='cuda');ids=torch.tensor(c,device='cuda');qs=torch.tensor(a['static'][q],device='cuda')
    first=int(cache['fit_start']);joint=[]
    for p in range(first,len(cache['base']),8):
        pos=np.arange(p,min(p+8,len(cache['base'])));b=tensor_inputs(cache,a,pos,q,coef)
        correction,_=model(b['condition'],b['history'],cs,ids,qs)
        joint.append((b['background'][:,:,None]+correction).cpu().numpy())
    pos=np.arange(first,len(cache['base']));base=cache['base'][pos][:,q];i1=(cache['base']+i1_features(cache['innovation'])@coef)[pos][:,q]
    path=root/'final_predictions.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,hours=a['hours'][pos+int(cache['start'])],query=q,p14_gru=base,i1=i1,joint=np.concatenate(joint))
    write_json(HERE/f'runs/joint_v2_s{seed}/evaluation_receipt.json',dict(prediction_sha256=sha(path),selection_sha256=sha(HERE/'selection_2023_v2.json'),test_cache_sha256=sha(root/('test_full.npz' if (root/'test_full.npz').exists() else 'test.npz')),current_epochs=len(pos),query_station_count=len(q),target_station_current_and_historical_GNSS_used_as_input=False))
    print(json.dumps(dict(stage='prediction',seed=seed,epochs=len(pos),stations=len(q))),flush=True)


def compare():
    a=dict(np.load(CACHE/'data.npz'));selection=json.loads((HERE/'selection_2023_v2.json').read_text(encoding='utf-8'));chosen=selection['selected_architecture']
    start=int(np.searchsorted(a['hours'],np.datetime64('2024-01-01','h').astype('i8')));hours=a['hours'][start:];target=a['values'][start:,:,0].astype('f8');N=len(a['static']);predictions={};receipts={}
    for seed in (42,17):
        oldpath=CACHE/f'original/predictions_s{seed}.npz';old=dict(np.load(oldpath));ti=np.searchsorted(hours,old['hours']);sj=old['station']
        if np.any(ti>=len(hours)) or np.any(hours[ti]!=old['hours']):raise ValueError('wrong old-model timestamp')
        if not np.allclose(target[ti,sj],old['target'],atol=5e-5,rtol=0):raise ValueError('target definitions differ')
        olddense=np.full(target.shape,np.nan);olddense[ti,sj]=old['pred'];predictions[f'original_s{seed}']=olddense
        path=CACHE/f'native_v2_s{seed}/final_predictions.npz';new=dict(np.load(path));ix=np.searchsorted(hours,new['hours']);q=new['query']
        if len(ix)!=len(hours) or np.any(hours[ix]!=new['hours']):raise ValueError('new prediction misses full-year epochs')
        for kind in ('p14_gru','i1','joint'):
            dense=np.full(target.shape,np.nan);value=new[kind][:,:,0] if kind=='joint' else new[kind];dense[ix[:,None],q[None]]=value
            predictions[f'{kind}_s{seed}']=dense
        receipts[str(seed)]=dict(original_sha256=sha(oldpath),new_sha256=sha(path))
    predictions['original_ensemble']=(predictions['original_s42']+predictions['original_s17'])/2
    predictions['ours_ensemble']=(predictions[f'{chosen}_s42']+predictions[f'{chosen}_s17'])/2
    role=a['role'];groups=a['groups'];train_groups=np.unique(groups[role=='train'])
    cohorts=dict(confirm=role=='confirm',dev=role=='dev',new_unseen_site=(role=='new2024')&~np.isin(groups,train_groups),all_spatial_holdout=(role!='train')&~np.isin(groups,train_groups))
    summary={name:common_rmse(target,predictions,mask) for name,mask in cohorts.items()}
    main=summary['confirm'];p=main['models'];old=p['original_ensemble']['micro_rmse_mm'];new=p['ours_ensemble']['micro_rmse_mm']
    summary=dict(protocol='same273training stations and2020-2023training pool; own feature pipelines; full2024 retrospective; primary63registered confirm stations, common valid keys',selection=chosen,cohorts=summary,primary_improvement_percent=(old-new)/old*100,source_receipts=receipts,selection_sha256=sha(HERE/'selection_2023_v2.json'))
    write_json(HERE/'comparison.json',summary)
    valid=common_mask(target,predictions,cohorts['all_spatial_holdout']);meta=pd.read_csv(HERE/'stations.csv');rows=[]
    for j in np.flatnonzero(valid.any(0)):
        count=int(valid[:,j].sum());r=dict(station_phys=meta.station_phys.iloc[j],role=role[j],records=count,Latitude=meta.Latitude.iloc[j],Longitude=meta.Longitude.iloc[j],Height=meta.Height.iloc[j],coast_km=float(np.expm1(a['static'][j,7]*6)))
        for name,pred in predictions.items():r[name]=float(np.sqrt(np.mean((pred[:,j][valid[:,j]]-target[:,j][valid[:,j]])**2)))
        rows.append(r)
    pd.DataFrame(rows).to_csv(HERE/'station_rmse.csv',index=False,lineterminator='\n')
    months=pd.to_datetime(hours.astype('datetime64[h]')).month.to_numpy();monthly=[]
    for month in range(1,13):
        metric=common_rmse(target[months==month],predictions={k:v[months==month] for k,v in predictions.items()},station_mask=cohorts['confirm'])
        for name,score in metric['models'].items():monthly.append(dict(month=month,model=name,records=metric['records'],**score))
    pd.DataFrame(monthly).to_csv(HERE/'monthly_rmse.csv',index=False,lineterminator='\n')
    np.savez_compressed(CACHE/'comparison_arrays.npz',target=target,valid=valid,hours=hours,original=predictions['original_ensemble'],ours=predictions['ours_ensemble'])
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','predict','compare']);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.action=='select':select()
    elif args.action=='predict':predict(args.seed)
    else:compare()
