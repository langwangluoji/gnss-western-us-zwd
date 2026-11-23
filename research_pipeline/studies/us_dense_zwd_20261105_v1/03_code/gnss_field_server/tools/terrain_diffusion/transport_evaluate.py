"""Frozen calendar expansion, matched IDW and paired gate reporting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from .data import Dataset,write_json,sha
from .transport import TransportModel
from .transport_train import ROOT,validation
from .metrics import evaluate


def calendar_name(all_months,hours):
    return ('all_months' if all_months else 'quarterly')+str(hours)


def idw(data,rows,hours):
    out=[]
    for li,c in enumerate(data.layouts()):
        dist=cdist(data.a['val_geo'][:,:2],data.a['geo'][c,:2]);rank=dist.argsort(1).argsort(1)
        weight=(rank<16)/np.maximum(dist,1.)**2;ys=[];preds=[[],[]]
        for row in rows:
            for t in range(12,12+hours):
                z=data.a['val_context'][row,t,c,0];w=weight*np.isfinite(z)[None];den=w.sum(1)
                if not (den>0).all():raise ValueError('IDW has no available support in fixed nearest16')
                z=np.nan_to_num(z);preds[0].append(w@z/den)
                preds[1].append((w@(z*np.exp(data.a['geo'][c,2]/3))/den)*np.exp(-data.a['val_geo'][:,2]/3))
                ys.append(data.a['val_y'][row,t])
        for i,name in enumerate(('surface_idw','height3000_idw')):
            out.append(dict(layout=li,method=name,**evaluate(np.stack(preds[i])[None],np.stack(ys))))
    return out


def run(a):
    torch.set_num_threads(4);data=Dataset();root=Path(a.root);rows=list(range(12)) if a.all_months else [0,3,6,9]
    if a.label is None:a.label=calendar_name(a.all_months,a.hours)
    for name in a.runs:
        p=root/name;out=p/a.label
        if out.exists():raise FileExistsError(out)
        out.mkdir();source=p/'best.pt';pack=torch.load(source,map_location='cuda',weights_only=False);conf=pack['config']
        if conf['data_sha256']!=data.contract['data_sha256']:raise ValueError('Data mismatch')
        model=TransportModel(conf['variant'],data.mean[0],data.std[0]).cuda().eval();model.load_state_dict(pack['model'])
        scores,pred,truth=validation(model,data,conf['arm'],rows,a.hours,a.members,19011,progress=True)
        np.savez_compressed(out/'evaluation.npz',samples=pred,truth=truth)
        write_json(out/'metrics.json',scores);write_json(out/'config.json',dict(checkpoint_sha256=sha(source),data_sha256=data.contract['data_sha256'],
            rows=rows,hours=a.hours,members=1 if conf['arm']=='deterministic' else a.members,seed=19011,selection_used=False))
        print(json.dumps(dict(run=name,calendar=a.label,rmse=scores['macro_rmse_mm'],crps=scores['crps_mm'])),flush=True)
    dest=root/f'idw_{a.label}.json'
    if not dest.exists():write_json(dest,idw(data,rows,a.hours))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--label');p.add_argument('--all-months',action='store_true');p.add_argument('--hours',type=int,default=24)
    p.add_argument('--members',type=int,default=16);run(p.parse_args())
