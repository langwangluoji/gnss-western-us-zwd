"""Throwaway research spike. No production imports are modified or deployed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from local_innovation_pilot import Pilot, ROOT as PRIORROOT
from local_innovation_components import query_nodes,station_macro_mse,epoch_batches
from gnss_field.artifacts import immutable_json,write_json,sha256
from gnss_field.runtime import configure

ROOT=Path(__file__).resolve().parent


class ResidualFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(69,32)
        self.residual=nn.Linear(32,32)
    def forward(self,x):
        h=torch.nn.functional.silu(self.input(x))
        return h+.1*torch.nn.functional.silu(self.residual(h))


class Correction(nn.Module):
    def __init__(self,features,coefficients):
        super().__init__();self.features=copy.deepcopy(features)
        self.output=nn.Linear(32,1)
        with torch.no_grad():
            self.output.weight.copy_(torch.tensor(coefficients[:-1])[None])
            self.output.bias.copy_(torch.tensor(coefficients[-1:]))
    def forward(self,x):return self.output(self.features(x)).squeeze(-1)


def append_one(x):return torch.cat([x,torch.ones_like(x[...,:1])],-1)


def main():
    configure(type('R',(),{'cpus':2})(),42)
    d=Pilot();scale=d.settings['grid_scale'];started=time.monotonic()
    immutable_json(ROOT/'run.lock.json',dict(source=sha256(__file__),data=sha256(PRIORROOT/'experiment.lock.json'),
        purpose='local research spike',arms=['zero','linear_ridge','nonlinear_ridge','nonlinear_zero_finetune','nonlinear_ridge_finetune'],
        lambdas=[1e-4,1e-2,1.],updates=588,EMA=2**(-1/42),lr=1e-4,min_lr=1e-4,warmup=0,seed=42,
        selection='same spatial x weather inner RMSE; raw primary; EMA diagnostic only; all outer results exploratory',server_jobs=False,
        confirm_gate='both folds inner >=0.5% improvement and outer direction consistent; then locked C2019 48 hours',D2020=False))
    x=torch.cat([d.t['features'],d.t['innovation']/scale],-1)
    all_results={};all_predictions={}
    for fold in (0,1):
        split=d.splits[fold];train=d.indices(split['train_hours']);inner=d.indices(split['inner_hours']);outer=d.indices(d.lock['outer_hours'])
        directory=ROOT/f'fold_{fold}';directory.mkdir(exist_ok=True)
        center=x[train].mean((0,1));std=x[train].std((0,1),correction=0);std[std<1e-4]=1.
        features=(x-center)/std
        torch.manual_seed(42);rf=ResidualFeatures().cuda()
        mask=d.a['valid'][train]&split['fit'][None];counts=mask.sum(0);use=counts>0
        weights=np.where(mask,1/np.maximum(counts,1)[None]/use.sum(),0.)
        target=np.where(mask,(d.a['truth'][train]-d.a['mean'][train])/scale,0.)
        def basis(linear):
            rows=[]
            with torch.no_grad():
                for ix in np.array_split(np.arange(len(x)),48):
                    z=append_one(features[ix] if linear else rf(features[ix]))
                    queried=(z[:,d.t['query_indices']]*d.t['query_weights'][None,:,:,None]).sum(2)
                    rows.append(queried.cpu().numpy().astype('f8'))
            return np.concatenate(rows)
        def metric(pred,indices,pop):return d.metric(pred,d.a['hours'][indices],pop)
        base_value=metric(d.a['mean'][inner],inner,split['inner'])['station_macro_rmse_mm']
        rows={'zero':dict(inner=metric(d.a['mean'][inner],inner,split['inner']),step=0)}
        predictions={'zero':d.a['mean'].copy()}
        chosen_coeff=None
        for name,linear in [('linear_ridge',True),('nonlinear_ridge',False)]:
            design=basis(linear);xt=design[train]
            xx=xt[mask];ww=weights[mask];yy=target[mask]
            gram=xx.T@(xx*ww[:,None]);rhs=xx.T@(ww*yy)
            candidates=[]
            for ridge in [1e-4,1e-2,1.]:
                beta=np.linalg.solve(gram+ridge*np.eye(gram.shape[0]),rhs)
                residual=(gram+ridge*np.eye(gram.shape[0]))@beta-rhs
                if np.max(abs(residual))>1e-10:raise ValueError('Ridge stationarity check failed')
                pred=d.a['mean']+scale*(design@beta)
                score=metric(pred[inner],inner,split['inner'])
                candidates.append(dict(ridge=ridge,inner=score,beta=beta,pred=pred))
            best=min(candidates,key=lambda z:z['inner']['station_macro_rmse_mm'])
            rows[name]=dict(inner=best['inner'],ridge=best['ridge'],step=0,
                ridge_candidates=[{k:v for k,v in c.items() if k not in ('pred','beta')} for c in candidates])
            predictions[name]=best['pred']
            np.savez_compressed(directory/f'{name}.npz',coefficients=best['beta'],center=center.cpu(),std=std.cpu())
            if not linear:chosen_coeff=best['beta'];chosen_lambda=best['ridge']
            print('RIDGE',fold,name,best['ridge'],best['inner']['station_macro_rmse_mm'],flush=True)
        # Same hidden initialization, data order, update budget, regularization
        # and evaluation checkpoints; only the readout initialization differs.
        for initialization in ['zero','ridge']:
            name=f'nonlinear_{initialization}_finetune'
            beta=np.zeros_like(chosen_coeff) if initialization=='zero' else chosen_coeff
            model=Correction(rf,beta).cuda();ema=copy.deepcopy(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            batches=epoch_batches(split['train_hours'],8,42,14)
            if len(batches)!=588:raise ValueError('Insufficient fixed paired update list')
            count_t=torch.tensor(counts,device='cuda');fit_t=torch.tensor(split['fit'],device='cuda')
            history=[];best=dict(value=base_value,step=-1,product='zero',state=None)
            @torch.no_grad()
            def predict(current,indices):
                out=[]
                for ix in np.array_split(indices,max(1,math.ceil(len(indices)/16))):
                    field=scale*current(features[ix]);change=query_nodes(field,d.t['query_indices'],d.t['query_weights'])
                    out.append((d.t['mean'][ix]+change).cpu().numpy())
                return np.concatenate(out)
            def evaluate(step):
                for kind,current in [('raw',model),('EMA',ema)]:
                    score=metric(predict(current,inner),inner,split['inner'])
                    history.append(dict(step=step,product=kind,inner=score))
                    if kind=='raw' and score['station_macro_rmse_mm']<best['value']:
                        best.update(value=score['station_macro_rmse_mm'],step=step,product=kind,state=copy.deepcopy(current.state_dict()))
            evaluate(0);losses=[]
            for step,hours in enumerate(batches,1):
                if (ROOT/'PAUSE').exists():raise RuntimeError('Research spike paused; no result is marked complete')
                ix=d.indices(hours);rate=1.
                for group in optimizer.param_groups:group['lr']=1e-4*rate
                optimizer.zero_grad(set_to_none=True)
                field=model(features[ix]);pred=d.t['mean'][ix]/scale+query_nodes(field,d.t['query_indices'],d.t['query_weights'])
                mse=station_macro_mse(pred,d.t['truth'][ix]/scale,d.t['valid'][ix]&fit_t[None],count_t,len(train))
                penalty=chosen_lambda*(model.output.weight.square().sum()+model.output.bias.square().sum())
                loss=mse+penalty
                if not torch.isfinite(loss):raise ValueError('Nonfinite research loss')
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(gn):raise ValueError('Nonfinite research gradient')
                optimizer.step()
                with torch.no_grad():
                    for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,1-2**(-1/42))
                if step%20==0:losses.append(dict(step=step,normalized_MSE=float(mse.detach()),penalty=float(penalty.detach()),lr=1e-4*rate))
                if step%42==0:evaluate(step)
            if best['state'] is None:pred=d.a['mean'].copy()
            else:
                model.load_state_dict(best['state']);pred=predict(model,np.arange(len(x)))
                torch.save(dict(state=best['state'],center=center.cpu(),std=std.cpu(),step=best['step'],product=best['product']),directory/f'{name}.pt')
            predictions[name]=pred
            rows[name]=dict(inner=metric(pred[inner],inner,split['inner']),step=best['step'],product=best['product'],ridge_penalty=chosen_lambda,history=history,training=losses)
            print('FINETUNE',fold,name,best['value'],best['step'],best['product'],flush=True)
        # No outer scores are available to the candidate/regularizer selection.
        order=list(rows)
        winner=min(order,key=lambda k:(rows[k]['inner']['station_macro_rmse_mm'],order.index(k)))
        immutable_json(directory/'selection.lock.json',dict(winner=winner,rows=rows,selection='new inner only',code=sha256(__file__)))
        for name in rows:
            rows[name]['outer']=metric(predictions[name][outer],outer,split['outer'])
            rows[name]['train']=metric(predictions[name][train],train,split['fit'])
        np.savez_compressed(directory/'predictions.npz',hours=d.a['hours'],**predictions)
        torch.save(rf.cpu().state_dict(),directory/'random_features.pt')
        all_results[str(fold)]=dict(selected=winner,rows=rows,base_inner_RMSE=base_value,
            relative_inner_gain=(base_value-rows[winner]['inner']['station_macro_rmse_mm'])/base_value)
        write_json(ROOT/'status.json',dict(fold_done=fold,seconds=time.monotonic()-started))
    passed=all(v['relative_inner_gain']>=.005 and v['rows'][v['selected']]['outer']['station_macro_rmse_mm']<v['rows']['zero']['outer']['station_macro_rmse_mm'] for v in all_results.values())
    write_json(ROOT/'results.json',dict(folds=all_results,local_gate_passed=passed,seconds=time.monotonic()-started,
        neural_scientific_updates=2352,server_jobs=False,C2019_confirmation_triggered=passed,D2020=False))
    print('SPIKE_COMPLETE',passed,time.monotonic()-started,flush=True)


if __name__=='__main__':main()
