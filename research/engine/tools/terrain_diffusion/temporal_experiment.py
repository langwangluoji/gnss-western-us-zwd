"""Paired local forecast screen: causal features, inner selection, frozen 2020 dev."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F

from .data import sha, write_json
from .temporal_data import anchors_for_period, spatial_history
from .temporal_encoders import TemporalFoundation


def score(prediction, truth):
    if not np.isfinite(prediction).all():
        raise ValueError('nonfinite predictions')
    def mean(x, axis):
        count = np.isfinite(x).sum(axis=axis)
        return np.divide(np.nansum(x,axis=axis), count,
                         out=np.full(np.shape(count),np.nan,dtype=float), where=count>0)
    error = np.where(np.isfinite(truth), prediction-truth, np.nan)
    finite_station = np.isfinite(error).any(axis=(0, 1))
    station_rmse = np.sqrt(mean(error**2, (0, 1)))
    increments = np.diff(prediction, axis=1)-np.diff(truth, axis=1)
    return dict(macro_rmse_mm=float(np.nanmean(station_rmse)),
                micro_rmse_mm=float(np.sqrt(np.nanmean(error**2))),
                mae_mm=float(np.nanmean(np.abs(error))), bias_mm=float(np.nanmean(error)),
                p95_absolute_mm=float(np.nanpercentile(np.abs(error), 95)),
                station_count=int(finite_station.sum()), valid_count=int(np.isfinite(error).sum()),
                increment_macro_rmse_mm=float(np.nanmean(np.sqrt(mean(increments**2, (0, 1))))),
                horizon_macro_rmse_mm=[float(np.nanmean(np.sqrt(mean(error[:, h]**2, 0)))) for h in (0, 2, 5)])


def calendar_events(hours, year):
    anchors = []
    for month in range(1, 13):
        for day in (8, 22):
            for hour in (6, 18):
                epoch = int(np.datetime64(f'{year}-{month:02d}-{day:02d}T{hour:02d}', 'h').astype('int64'))
                anchors.append(int(np.searchsorted(hours, epoch)))
    return anchors


class Events:
    def __init__(self, data, side=8, max_history=72):
        self.data, self.side, self.max_history = Path(data), side, max_history
        self.contract = json.loads((self.data/'contract.json').read_text(encoding='utf-8'))
        for name, expected in self.contract['files'].items():
            if sha(self.data/name) != expected:
                raise ValueError(f'Data hash mismatch: {name}')
        self.hours = np.load(self.data/'hours.npy')
        self.dev_hours = np.load(self.data/'dev_hours.npy')
        self.values = np.load(self.data/'values.npy', mmap_mode='r')
        self.dev_values = np.load(self.data/'dev_values.npy', mmap_mode='r')
        self.geo = dict(np.load(self.data/'geometry.npz'))
        xy = self.geo['xy']
        self.lo, self.hi = xy.min(0)-25, xy.max(0)+25
        gx, gy = np.meshgrid(np.linspace(self.lo[0], self.hi[0], side), np.linspace(self.lo[1], self.hi[1], side))
        self.grid = np.c_[gx.ravel(), gy.ravel()]
        self.layouts = []
        for seed in (8120, 8121, 8122):
            c = np.sort(np.random.default_rng(seed).choice(len(xy), round(len(xy)*.2), replace=False))
            self.layouts.append(c)

    def make(self, anchor, layout, development=False):
        g = self.geo
        c = self.layouts[layout]
        if development:
            qxy, qheight, static = g['dev_xy'], g['dev_height'], g['dev_static']
            coast = g['dev_coast']
            epoch = self.hours[anchor]
            target_index = np.searchsorted(self.dev_hours, epoch)
            y = np.array(self.dev_values[target_index+1:target_index+7, :, 0])
        else:
            qxy, qheight, static = g['xy'], g['height'], g['static']
            coast = g['coast']
            y = np.array(self.values[anchor+1:anchor+7, :, 0])
            # Hold out every target physical group from all history hours.
            hidden = ~np.isin(g['groups'], g['groups'][c])
            y[:, ~hidden] = np.nan
        history = spatial_history(self.values, self.hours, anchor, self.max_history,
                                  c, g['xy'], g['height'], self.grid)
        distance = cdist(qxy, g['xy'][c])
        latest = np.array(self.values[anchor, c, 0])
        valid = np.isfinite(latest)
        weights = 1/np.maximum(distance, 5)**2
        weights *= valid[None]
        z0 = np.nan_to_num(latest)*np.exp(g['height'][c]/3000)
        base = (weights@z0/np.maximum(weights.sum(1), 1e-8))*np.exp(-qheight/3000)
        uv = (qxy-self.lo)/(self.hi-self.lo)*2-1
        static = np.c_[static, np.log1p(coast)/6, coast/500]
        return dict(history=history.astype('float16'), uv=uv.astype('float32'), static=static.astype('float32'),
                    y=y.astype('float32'), base=(base/100).astype('float32'),
                    epoch=int(self.hours[anchor]), layout=layout,
                    nearest=np.where(valid[None],distance,np.inf).min(1).astype('float32'))


def tensor_batch(events, indices, length, device='cuda'):
    selected = [events[int(i)] for i in indices]
    return {key:torch.as_tensor(np.stack([e[key][-length:] if key == 'history' else e[key]
                                        for e in selected]), dtype=torch.float32, device=device)
            for key in ('history','uv','static','y','base')}


@torch.no_grad()
def predict(model, events, length):
    model.eval()
    output = []
    for k in range(0, len(events), 4):
        b = tensor_batch(events, range(k, min(k+4,len(events))), length)
        output.append((model(b['history'],b['uv'],b['static'],background=b['base'])*100).cpu().numpy())
    return np.concatenate(output)


def fit(kind, length, seed, events, inner, steps, out, monitor=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalFoundation(kind,length).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best, chosen, checkpoint = float('inf'), steps, None
    trace, start = [], time.monotonic()
    for step in range(1,steps+1):
        model.train()
        b = tensor_batch(events,rng.integers(len(events),size=4),length)
        mask = torch.isfinite(b['y'])
        y = torch.nan_to_num(b['y'])/100
        pred = model(b['history'],b['uv'],b['static'],background=b['base'])
        loss = (pred-y)[mask].square().mean()
        imask = mask[:,1:]&mask[:,:-1]
        loss = loss + .1*(torch.diff(pred,dim=1)-torch.diff(y,dim=1))[imask].square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),2)
        opt.step()
        if monitor and (step==1 or step%100==0 or step==steps):
            p = predict(model, inner, length)
            metric = score(p,np.stack([e['y'] for e in inner]))['macro_rmse_mm']
            trace.append(dict(step=step,loss=float(loss.detach()),inner_macro_rmse_mm=metric))
            if metric < best:
                best,chosen = metric,step
                checkpoint = copy.deepcopy(model.state_dict())
            print(json.dumps(dict(stage='inner_selection',kind=kind,length=length,seed=seed,step=step,rmse=metric)),flush=True)
    if monitor:
        model.load_state_dict(checkpoint)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'model.pt')
    write_json(out/'fit.json',dict(kind=kind,length=length,seed=seed,steps=steps,chosen_step=chosen,
        elapsed_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),
        trace=trace,monitor=monitor))
    return model,chosen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--train-events',type=int,default=384)
    parser.add_argument('--histories',type=int,nargs='+',default=[12,48,72])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,17])
    parser.add_argument('--kinds',nargs='+',default=['none','mlp','gru','lstm','transformer'])
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'protocol.json').exists():
        raise FileExistsError('Use a fresh experiment output directory')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(False)
    config=vars(args).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    write_json(args.out/'protocol.json',dict(status='LOCAL_FORECAST_SCREEN_NOT_ANNUAL',config=config,
        code_sha256=sha(__file__),data_contract_sha256=sha(args.data/'contract.json'),
        task='6-hour forecast at unseen stations; no target history; dynamic inputs <= anchor',
        spatial='8x8 carrier, 16-channel spatial convolution/downsample/skip; exact point IDW persistence skip',
        coast='C1 query static only; C2 edge coast and terrain FiLM not yet activated',
        selection='2016-2018 inner fit, 2019 selection; scratch refit 2016-2019 for chosen updates; frozen 2020 dev',
        differences_from_final_plan='Local small-carrier screen; no diffusion or OOF residual claim; not p14 reproduction',
        device=torch.cuda.get_device_name(),torch_version=torch.__version__))
    ex=Events(args.data,max_history=max(args.histories))
    hour=lambda d:int(np.datetime64(d,'h').astype('int64'))
    fit_anchors=anchors_for_period(ex.hours,max(args.histories),6,hour('2016-01-01'),hour('2019-01-01'))
    fit_anchors=np.random.default_rng(778).choice(fit_anchors,args.train_events,replace=False)
    inner_anchors=calendar_events(ex.hours,2019)
    dev_anchors=calendar_events(ex.hours,2020)
    rows=[]
    for name, anchors, development in [('fit',fit_anchors,False),('inner',inner_anchors,False),('dev',dev_anchors,True)]:
        events=[]
        for i, anchor in enumerate(anchors):
            layouts=range(3) if name!='fit' else [i%3]
            for layout in layouts:
                events.append(ex.make(anchor,layout,development))
            if (i+1)%24==0:
                print(json.dumps(dict(stage='cache',split=name,done=i+1,total=len(anchors))),flush=True)
        rows.append(events)
    train,inner,dev=rows
    truth=np.stack([e['y'] for e in dev])
    base=np.stack([np.broadcast_to(e['base'][None]*100,(6,len(e['base']))) for e in dev])
    write_json(args.out/'event_manifest.json',dict(fit=[dict(epoch=e['epoch'],layout=e['layout']) for e in train],
        inner=[dict(epoch=e['epoch'],layout=e['layout']) for e in inner],
        development=[dict(epoch=e['epoch'],layout=e['layout']) for e in dev]))
    np.savez_compressed(args.out/'development_labels.npz',truth=truth,
        epochs=np.array([e['epoch'] for e in dev]),coast=ex.geo['dev_coast'],height=ex.geo['dev_height'],
        nearest=np.stack([e['nearest'] for e in dev]),xy=ex.geo['dev_xy'])
    summary=[dict(kind='spatial_persistence',length=0,seed=0,metrics=score(base,truth))]
    np.save(args.out/'persistence.npy',base)
    for kind in args.kinds:
        for length in ([12] if kind=='none' else args.histories):
            for seed in args.seeds:
                tag=f'{kind}_h{length}_s{seed}'
                model,chosen=fit(kind,length,seed,train,inner,args.steps,args.out/tag/'inner')
                del model;torch.cuda.empty_cache()
                # Fresh weights; 2019 now joins training only after its chosen step is fixed.
                model,_=fit(kind,length,seed,train+inner,[],chosen,args.out/tag/'refit',monitor=False)
                prediction=predict(model,dev,length)
                np.save(args.out/tag/'prediction.npy',prediction)
                item=dict(kind=kind,length=length,seed=seed,chosen_step=chosen,metrics=score(prediction,truth))
                write_json(args.out/tag/'metrics.json',item)
                summary.append(item);write_json(args.out/'summary.json',summary)
                print(json.dumps(dict(stage='dev_evaluated',**item)),flush=True)
                del model;torch.cuda.empty_cache()
    write_json(args.out/'status.json',dict(status='COMPLETE_LOCAL_FORECAST_SCREEN',fits=len(summary)-1))


if __name__=='__main__':main()
