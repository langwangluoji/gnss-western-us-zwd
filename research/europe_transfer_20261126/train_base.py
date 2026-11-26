"""Chronological continuous-block training of the migrated actual p14 carrier."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import CACHE,HERE,sha,write_json,neural_cutoff
from .field import EuropeBase,FieldData


@torch.no_grad()
def monitor(model,d,origins,context,queries):
    model.eval();square=np.zeros(len(queries));count=np.zeros(len(queries))
    for origin in origins:
        b=d.batch(np.arange(origin,origin+4),context,queries);p=model(b)
        mask=b['mask'];error=torch.where(mask,p-b['target'],0)
        square+=error.square().sum(0).cpu().numpy();count+=mask.sum(0).cpu().numpy()
    active=count>0
    return float(np.sqrt(square[active]/count[active]).mean())


def run(seed,steps,smoke=False,version=1):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.manual_seed(seed)
    a=dict(np.load(CACHE/'data.npz'));d=FieldData(a)
    train=np.flatnonzero(a['role']=='train');dev=np.flatnonzero(a['role']=='dev')
    cutoff=neural_cutoff(a['hours'],version)
    devstart=int(np.searchsorted(a['hours'],np.datetime64('2023-01-01','h').astype('i8')))
    stop=int(np.searchsorted(a['hours'],np.datetime64('2024-01-01','h').astype('i8')))
    origins=np.linspace(devstart+16,stop-8,24).astype(int)
    out=HERE/('smoke' if smoke else 'runs')/f'base{("_v2" if version==2 else "")}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    model=EuropeBase().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    ema=copy.deepcopy(model);rng=np.random.default_rng(seed);rows=[];best=float('inf');start=time.monotonic()
    blocks=list(range(16,cutoff-4,4))
    write_json(out/'lock.json',dict(model='original P14Fusion(gru16) + current transported spatial prior',version=version,seed=seed,steps=steps,block_records=4,block_hours=24,chronological=True,context_count_range=[96,min(224,len(train)-32)],train_stations=len(train),dev_stations=len(dev),neural_years=[2020,2021,2022,2023] if version==2 else [2020,2021,2022],selection_year=2023,selection_is_spatial_holdout=version==2,data_sha256=sha(CACHE/'data.npz'),code_sha256=sha(__file__),feature_sha256=sha(Path(__file__).with_name('field.py')),test_2024_used_for_training_or_selection=False))
    for step in range(1,steps+1):
        origin=blocks[(step-1)%len(blocks)]
        ncontext=int(rng.choice([96,160,min(224,len(train)-32)]));context=np.sort(rng.choice(train,ncontext,replace=False))
        available=train[~np.isin(a['groups'][train],a['groups'][context])]
        query=np.sort(rng.choice(available,min(96,len(available)),replace=False))
        if len(query)==0:raise ValueError('no noncolocated targets')
        b=d.batch(np.arange(origin,origin+4),context,query)
        model.train();opt.zero_grad(set_to_none=True);pred=model(b);mask=b['mask']
        loss=((pred[mask]-b['target'][mask])/10).square().mean()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite training')
        opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.01)
        if step==1 or step%100==0 or step==steps:
            row=dict(step=step,origin=int(origin),seconds=time.monotonic()-start,batch_rmse_mm=float(loss.detach().sqrt()*10),gradient_norm=float(norm))
            if step%500==0 or step==steps:
                row['dev_station_macro_rmse_mm']=monitor(ema,d,origins[:2] if smoke else origins,train,dev)
                if row['dev_station_macro_rmse_mm']<best:
                    best=row['dev_station_macro_rmse_mm'];torch.save(ema.state_dict(),out/'best.pt')
            rows.append(row);write_json(out/'progress.json',row)
            with (out/'curve.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(stage=out.name,**row)),flush=True)
    torch.save(model.state_dict(),out/'last.pt');torch.save(ema.state_dict(),out/'ema.pt')
    write_json(out/'complete.json',dict(steps=steps,seconds=time.monotonic()-start,best_dev_station_macro_rmse_mm=best,best_sha256=sha(out/'best.pt'),parameters=sum(p.numel() for p in model.parameters())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=3000);p.add_argument('--smoke',action='store_true');p.add_argument('--version',type=int,choices=[1,2],default=1);args=p.parse_args()
    run(args.seed,args.steps,args.smoke,args.version)
