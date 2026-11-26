"""Frozen-weight diagnostics. No training, sealed labels, or candidate selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from .data import Dataset,PROJECT,write_json,sha
from .model import DiffusionModel,v_target
from .train import history_push
from .metrics import evaluate

ROOT=PROJECT/'results/20261123_terrain_diffusion_v10'
OUT=PROJECT/'results/20261123_terrain_diffusion_v10_diagnosis'


def subset(d,q):
    e=dict(d)
    for key in ('query','qgeo','target','mask'):e[key]=d[key][:,q]
    e['history']=d['history'][:,:,q]
    return e


def predict(model,d,arm,noise):
    if arm=='deterministic':return model.baseline(d)[0]
    return model.sample(d,noise,20)


def score(pred,truth,data):
    return evaluate(pred*data.std[0]+data.mean[0],truth*data.std[0]+data.mean[0])


@torch.no_grad()
def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=False);data=Dataset();c=data.layouts()[0]
    q=np.arange(137);small=np.sort(np.random.default_rng(291).choice(137,32,False))
    results=[];curves=[];sources={}
    for run in sorted((ROOT/'screen').glob('*/history.csv')):
        h=pd.read_csv(run);config=json.loads((run.parent/'config.json').read_text(encoding='utf-8'))
        curves.append(dict(run=run.parent.name,first200_loss=float(h.loss.iloc[:200].mean()),last200_loss=float(h.loss.iloc[-200:].mean()),
                           first200_train_rmse=float(h.train_x0_rmse_mm.iloc[:200].mean()),last200_train_rmse=float(h.train_x0_rmse_mm.iloc[-200:].mean()),
                           validation=h.dropna(subset=['validation_macro_rmse_mm'])[['step','validation_macro_rmse_mm']].to_dict('records')))
    for name in ('b1_s42','d1_s42','b1_s17','d1_s17'):
        path=ROOT/'screen'/name/'best.pt';sources[name]=sha(path)
        pack=torch.load(path,map_location='cuda',weights_only=False);config=pack['config'];arm=config['arm']
        model=DiffusionModel(config['width'],config['blocks']).cuda().eval();model.load_state_dict(pack['model'])
        result=dict(run=name,selected_step=pack['step']);trajectories={mode:[] for mode in ('self','reset')};truths=[];qchanges=[]
        # Common noise and truth across modes. One layout, two months, four hours.
        for row in (0,6):
            hist=None;seq={k:[] for k in trajectories};yt=[]
            gen=torch.Generator(device='cuda').manual_seed(2301+row)
            for t in range(12,16):
                base=data.batch(row,t,c,q,'validation');yt.append(np.where(base['mask'].cpu()[0],base['target'].cpu()[0],np.nan))
                d={k:v.expand(32,*v.shape[1:]) for k,v in base.items()};noise=torch.randn(32,137,device='cuda',generator=gen)
                reset=predict(model,d,arm,noise);seq['reset'].append(reset.cpu().numpy())
                if hist is not None:d['history']=hist
                pred=predict(model,d,arm,noise);seq['self'].append(pred.cpu().numpy());hist=history_push(d,pred)
                if t==12:
                    # Start from the same missing history and same noise at retained coordinates.
                    e={k:v.expand(32,*v.shape[1:]) for k,v in base.items()}
                    sub=predict(model,subset(e,small),arm,noise[:,small])
                    qchanges.append((sub-reset[:,small]).cpu().numpy()*data.std[0])
            truths.append(np.stack(yt))
            for mode in trajectories:trajectories[mode].append(np.stack(seq[mode],1))
        truth=np.concatenate(truths);samples={k:np.concatenate(v,1) for k,v in trajectories.items()}
        result['validation']={mode:{str(k):score(s[:k],truth,data) for k in (4,8,16,32)} for mode,s in samples.items()}
        change=np.stack(qchanges);result['query32_vs137']=dict(member_rms_change_mm=float(np.sqrt(np.mean(change**2))),
            mean_rms_change_mm=float(np.sqrt(np.mean(change.mean(1)**2))),p95_abs_member_change_mm=float(np.quantile(abs(change),.95)))
        # Actual training windows only: teacher-history is a diagnostic, never validation input.
        eligible=np.flatnonzero(~np.isin(data.a['groups'],data.a['groups'][c]));tq=np.sort(np.random.default_rng(488).choice(eligible,32,False))
        trainpred={key:[] for key in ('teacher','missing')};ty=[]
        for row in (0,6,24,48):
            gen=torch.Generator(device='cuda').manual_seed(5500+row);noise=torch.randn(32,32,device='cuda',generator=gen)
            for mode in trainpred:
                d=data.batch(row,18,c,tq,teacher=mode=='teacher')
                if mode=='teacher':ty.append(np.where(d['mask'].cpu()[0],d['target'].cpu()[0],np.nan))
                d={k:v.expand(32,*v.shape[1:]) for k,v in d.items()}
                trainpred[mode].append(predict(model,d,arm,noise).cpu().numpy())
        result['training_history']={k:score(np.stack(v,1),np.stack(ty),data) for k,v in trainpred.items()}
        np.savez_compressed(OUT/f'{name}.npz',truth=truth,self_samples=samples['self'],reset_samples=samples['reset'],query_change=change)
        if arm=='diffusion':
            bands=[]
            for s in (1,10,25,50,75,90,100):
                errs=[];vl=[]
                for row in (0,6):
                    valid=np.flatnonzero(np.isfinite(data.a['val_y'][row,12]))[:32]
                    d=data.batch(row,12,c,valid,'validation');gen=torch.Generator(device='cuda').manual_seed(881+row)
                    noise=torch.randn(d['target'].shape,device='cuda',generator=gen);a=model.ab[s];x=a.sqrt()*d['target']+(1-a).sqrt()*noise
                    v=model(x,torch.tensor([s],device='cuda'),d);clean=a.sqrt()*x-(1-a).sqrt()*v
                    errs.append(float(((clean-d['target'])**2).mean())*float(data.std[0]**2));vl.append(float((v-v_target(d['target'],noise,a)).square().mean()))
                bands.append(dict(step=s,reconstruction_rmse_mm=float(np.sqrt(np.mean(errs))),v_mse=float(np.mean(vl))))
            result['noise_bands']=bands
        results.append(result);write_json(OUT/'diagnostics.json',results);print(json.dumps(dict(done=name)),flush=True)
    # Untuned, descriptive same-population spatial interpolation baselines.
    baselines=[]
    for li,c in enumerate(data.layouts()):
        dist=cdist(data.a['val_geo'][:,:2],data.a['geo'][c,:2]);rank=dist.argsort(1).argsort(1)
        w=(rank<16)/np.maximum(dist,1.)**2;preds={key:[] for key in ('surface_idw','height3000_idw')};ys=[]
        for row in (0,6):
            for t in range(12,16):
                z=data.a['val_context'][row,t,c,0];valid=np.isfinite(z);wv=w*valid[None];den=wv.sum(1)
                assert (den>0).all()
                preds['surface_idw'].append(wv@np.nan_to_num(z)/den)
                corrected=np.nan_to_num(z)*np.exp(data.a['geo'][c,2]/3)
                preds['height3000_idw'].append((wv@corrected/den)*np.exp(-data.a['val_geo'][:,2]/3))
                ys.append(data.a['val_y'][row,t])
        for name,p in preds.items():baselines.append(dict(layout=li,method=name,**evaluate(np.stack(p)[None],np.stack(ys))))
    write_json(OUT/'idw.json',baselines);write_json(OUT/'curve_statistics.json',curves)
    write_json(OUT/'receipt.json',dict(data_sha256=data.contract['data_sha256'],checkpoint_sha256=sources,code_sha256=sha(__file__),
        status='frozen_diagnostic_no_training',validation='layout0, Jan/Jul four hours, K32 nested ensemble',
        training_history='training410 only; four selected training windows; not a validation score',sealed_labels_read=False))


if __name__=='__main__':main()
