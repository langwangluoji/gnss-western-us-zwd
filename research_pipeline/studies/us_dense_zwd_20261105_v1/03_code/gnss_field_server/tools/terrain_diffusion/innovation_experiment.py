"""Fit condition-innovation heads over continuous training blocks, then freeze."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from .data import PROJECT,sha,write_json
from .p14_fusion import FusionData,P14Fusion,chronological_blocks
from .p14_fusion_experiment import reconstruction_scores
from .station_query import hole_schedule
from .station_weighting import distance_weights,BIN_WEIGHTS,weighted_loss
from .innovation_refinement import innovation_fields,causal_features,normal_equations,solve_ridge

CASES={'I0_current':3,'I1_history':9,'I2_seasonal':27}


@torch.no_grad()
def feature_batch(data,model,anchors,context,development=False):
    begin=max(71,int(anchors[0])-3);c=torch.as_tensor(context,device='cuda')
    errors=[];valids=[]
    def visible_errors(indices,prediction):
        observed=data.values[torch.as_tensor(indices,device='cuda')[:,None],c[None],0]
        valid=torch.isfinite(observed)
        error=torch.where(valid,observed-(prediction[:,c]*data.std[0]+data.mean[0]),0)
        return error,valid
    if begin<int(anchors[0]):
        past=np.arange(begin,int(anchors[0]));b=data.batch(past,context)
        error,valid=visible_errors(past,model(b));errors.append(error);valids.append(valid)
    full=data.batch(anchors,context);full_prediction=model(full)
    error,valid=visible_errors(anchors,full_prediction);errors.append(error);valids.append(valid)
    b=data.batch(anchors,context,True) if development else full
    prediction=model(b) if development else full_prediction
    g=data.events.geo;prefix='dev_' if development else ''
    qxy=torch.as_tensor(g[prefix+'xy'],device='cuda',dtype=torch.float32)
    qh=torch.as_tensor(g[prefix+'height'],device='cuda',dtype=torch.float32)
    cxy=torch.as_tensor(g['xy'][context],device='cuda',dtype=torch.float32)
    ch=torch.as_tensor(g['height'][context],device='cuda',dtype=torch.float32)
    fields=innovation_fields(torch.cat(errors),torch.cat(valids),qxy,cxy,qh,ch)
    positions=torch.arange(int(anchors[0])-begin,int(anchors[-1])-begin+1,device='cuda')
    features=causal_features(fields,positions,data.hours[torch.as_tensor(anchors,device='cuda')])
    weights=distance_weights(qxy,cxy,valids[-1])
    if not torch.isfinite(features).all():raise FloatingPointError('nonfinite innovation features')
    return features,prediction,b,weights


@torch.no_grad()
def evaluate_heads(data,model,coefficients,out,seed,hours,months,source):
    predictions={name:[] for name in CASES};truth=[];references=[];e=data.events
    for li,c in enumerate(e.layouts):
        month_predictions={name:[] for name in CASES};month_references=[]
        for month in range(1,months+1):
            epoch=int(np.datetime64(f'2020-{month:02d}-08','h').astype('int64'))
            start=int(np.searchsorted(e.hours,epoch));rows={name:[] for name in CASES};base=[]
            for a in range(start,start+hours,12):
                anchors=np.arange(a,min(a+12,start+hours))
                x,p,b,w=feature_batch(data,model,anchors,c,True);physical=p*data.std[0]+data.mean[0]
                base.append(physical.cpu().numpy())
                for name,d in CASES.items():
                    correction=(x[...,:d].double()@coefficients[name].to('cuda')*10).clamp(-5,5)
                    rows[name].append((physical+correction).float().cpu().numpy())
            for name in CASES:month_predictions[name].append(np.concatenate(rows[name]))
            month_references.append(np.concatenate(base))
            if li==0:
                k=int(np.searchsorted(e.dev_hours,epoch));truth.append(np.array(e.dev_values[k:k+hours,:,0]))
        for name in CASES:predictions[name].append(np.stack(month_predictions[name]))
        references.append(np.stack(month_references))
        print(json.dumps(dict(stage='evaluation',seed=seed,layout=li)),flush=True)
    truth=np.stack(truth);reference=np.stack(references)
    original=np.load(source/f'gru_s{seed}/evaluation.npz')
    np.testing.assert_array_equal(truth,original['truth'][:months,:hours])
    discrepancy=float(np.max(np.abs(reference-original['prediction'][:,:months,:hours])))
    if discrepancy>1e-3:raise ValueError(f'original source predictions differ by {discrepancy}mm')
    write_json(out/f'source_reproduction_s{seed}.json',dict(max_difference_mm=discrepancy,truth_exact=True))
    rows=[]
    for name in CASES:
        folder=out/f'{name}_s{seed}';prediction=np.stack(predictions[name])
        np.savez_compressed(folder/'evaluation.npz',prediction=prediction,truth=truth)
        metrics=reconstruction_scores(prediction,truth);write_json(folder/'metrics.json',metrics)
        rows.append(dict(case=name,seed=seed,steps=None,fit_method='analytic_ridge_contiguous_blocks',metrics=metrics))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--source',type=Path,default=PROJECT/'results/20261125_p14_continuous_fusion')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seeds',type=int,nargs='+',default=[42,17])
    p.add_argument('--eval-hours',type=int,default=72);p.add_argument('--months',type=int,default=12)
    args=p.parse_args();torch.set_num_threads(3);torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.32);args.out.mkdir(parents=True,exist_ok=False)
    out=args.out
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_weights={str(s):sha(args.source/f'gru_s{s}/last.pt') for s in args.seeds},
        data_sha256=sha(args.data/'contract.json'),code_sha256={f:sha(Path(__file__).with_name(f)) for f in
            ('innovation_experiment.py','innovation_refinement.py','p14_fusion.py','temporal_encoders.py','station_query.py','station_weighting.py','unet_foundation.py','data.py')},
        fit_method=f'analytic normal equations from {args.steps} contiguous12h blocks, not SGD updates',
        innovation='observed minus frozen source prediction at same visible station; causal lags0/1/3h',
        feature_scale_mm=10,ridge_relative_diagonal=.01,inference_trust_region_mm=5,
        linear_surrogate_fitted_before_output_clipping=True,distance_weights=list(BIN_WEIGHTS),
        training='2016-2019 chronological blocks,75%random/25%hole100,station IDs only sampled',
        evaluation=f'2020 {args.months}monthly x{args.eval_hours}h x3layouts x137dev, exposed; no checkpoint selection',
        strict_oof=False,confirmation_C_D_read=False)
    write_json(out/'protocol.json',protocol)
    data=FusionData(args.data);e=data.events
    stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
    blocks=chronological_blocks(71,stop,12);summary=[]
    for seed in args.seeds:
        model=P14Fusion('gru').cuda().eval()
        model.load_state_dict(torch.load(args.source/f'gru_s{seed}/last.pt',weights_only=True))
        for param in model.parameters():param.requires_grad_(False)
        rng=np.random.default_rng(seed+28011);holes=hole_schedule(args.steps,seed)
        matrix=torch.zeros(27,27,dtype=torch.float64);rhs=torch.zeros(27,dtype=torch.float64);energy=0.
        trace=[];folder=out/f'trajectory_s{seed}';folder.mkdir();begin=time.monotonic()
        for step in range(1,args.steps+1):
            anchors=blocks[(step-1)%len(blocks)]
            if holes[step-1]:
                center=int(rng.integers(410));pool=np.flatnonzero(np.linalg.norm(e.geo['xy']-e.geo['xy'][center],axis=1)>100)
            else:pool=np.arange(410)
            c=np.sort(rng.choice(pool,82,False))
            x,pred,b,w=feature_batch(data,model,anchors,c)
            residual=(b['target']-pred)*data.std[0]/10
            m,v=normal_equations(x,residual,b['mask'],w);matrix+=m;rhs+=v
            energy+=float(weighted_loss(torch.zeros_like(residual,dtype=torch.float64),residual.double(),b['mask'],w.double()))
            if step==1 or step%100==0 or step==args.steps:
                objectives={}
                for name,d in CASES.items():
                    beta=solve_ridge(matrix[:d,:d],rhs[:d])
                    objectives[name]=max(0,float(energy-2*beta@rhs[:d]+beta@matrix[:d,:d]@beta))/step*100
                row=dict(step=step,seconds=time.monotonic()-begin,cumulative_surrogate_loss_mm2=objectives,
                    first_hour=int(e.hours[anchors[0]]),last_hour=int(e.hours[anchors[-1]]),
                    reference_batch_rmse_mm=float(((pred-b['target'])[b['mask']]).square().mean().sqrt()*data.std[0]))
                trace.append(row);write_json(folder/'progress.json',row);write_json(folder/'curve.json',trace)
                print(json.dumps(dict(seed=seed,**row)),flush=True)
            if step%500==0:
                torch.save(dict(matrix=matrix,rhs=rhs,target_energy=energy,step=step,seed=seed,numpy_rng=rng.bit_generator.state,
                    data_sha256=protocol['data_sha256'],source_sha256=protocol['source_weights'][str(seed)]),folder/f'statistics_{step}.pt')
        coefficients={name:solve_ridge(matrix[:d,:d],rhs[:d]) for name,d in CASES.items()}
        if not all(torch.isfinite(v).all() for v in coefficients.values()):raise FloatingPointError('nonfinite ridge solution')
        for name,d in CASES.items():
            dest=out/f'{name}_s{seed}';dest.mkdir();torch.save(coefficients[name],dest/'coefficients.pt')
            write_json(dest/'fit.json',dict(seed=seed,case=name,dimensions=d,blocks=args.steps,
                fit_method='analytic ridge, not SGD',trace=trace,coefficient_norm=float(coefficients[name].norm())))
        rows=evaluate_heads(data,model,coefficients,out,seed,args.eval_hours,args.months,args.source)
        for row in rows:row['steps']=args.steps
        summary.extend(rows);write_json(out/'summary.json',summary)
        print(json.dumps(rows),flush=True);del model;torch.cuda.empty_cache()
    write_json(out/'status.json',dict(status='COMPLETE_OBSERVED_INNOVATION_RIDGE',completed=len(summary)))


if __name__=='__main__':main()
