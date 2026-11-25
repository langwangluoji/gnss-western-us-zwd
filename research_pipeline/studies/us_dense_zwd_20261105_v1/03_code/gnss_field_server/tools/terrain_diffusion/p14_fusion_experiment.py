"""Continuous-time 3000-update current-field fusion with verified p14 weights."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .p14_fusion import FusionData,P14Fusion,chronological_blocks
from .data import write_json,sha


def reconstruction_scores(pred,truth):
    # [layout,month,consecutive hour,station], no increments across months.
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions in fixed scoring population')
    error=pred-truth[None]
    counts=np.isfinite(error).sum((1,2))
    station=np.sqrt(np.divide(np.nansum(error**2,axis=(1,2)),counts,
        out=np.full(counts.shape,np.nan),where=counts>0))
    inc=np.diff(pred,axis=2)-np.diff(truth,axis=1)[None]
    icount=np.isfinite(inc).sum((1,2))
    increment_station=np.sqrt(np.divide(np.nansum(inc**2,axis=(1,2)),icount,
        out=np.full(icount.shape,np.nan),where=icount>0))
    return dict(macro_rmse_mm=float(np.nanmean(station)),
        micro_rmse_mm=float(np.sqrt(np.nanmean(error**2))),
        mae_mm=float(np.nanmean(np.abs(error))),bias_mm=float(np.nanmean(error)),
        increment_macro_rmse_mm=float(np.nanmean(increment_station)),
        layout_macro_rmse_mm=np.nanmean(station,axis=1).tolist(),
        station_count=int(np.isfinite(station).any(0).sum()),valid_count=int(np.isfinite(error).sum()))


@torch.no_grad()
def evaluate(data,model,hours,months,out):
    model.eval();e=data.events;allpred=[];truth=[]
    for li,c in enumerate(e.layouts):
        monthpred=[]
        for month in range(1,months+1):
            epoch=int(np.datetime64(f'2020-{month:02d}-08','h').astype('int64'))
            start=int(np.searchsorted(e.hours,epoch));pred=[]
            for a in range(start,start+hours,12):
                anchors=np.arange(a,min(a+12,start+hours))
                b=data.batch(anchors,c,True,model.kind not in ('p14','features'),model.kind!='p14')
                pred.append((model(b)*data.std[0]+data.mean[0]).cpu().numpy())
            monthpred.append(np.concatenate(pred))
            if li==0:
                k=int(np.searchsorted(e.dev_hours,epoch));truth.append(np.array(e.dev_values[k:k+hours,:,0]))
        allpred.append(np.stack(monthpred))
    pred=np.stack(allpred);truth=np.stack(truth)
    np.savez_compressed(out/'evaluation.npz',prediction=pred,truth=truth)
    metrics=reconstruction_scores(pred,truth)
    write_json(out/'metrics.json',metrics)
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--length',type=int,default=72);p.add_argument('--batch-hours',type=int,default=12)
    p.add_argument('--eval-hours',type=int,default=72);p.add_argument('--months',type=int,default=12)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,17])
    p.add_argument('--kinds',nargs='+',default=['p14','features','mlp','gru','lstm','transformer'])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    resuming=(args.out/'protocol.json').exists()
    if resuming and not args.resume:raise FileExistsError('Use --resume or a fresh experiment output')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=False
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    protocol=dict(status='RUNNING_MATCHED_CURRENT_FIELD_FUSION',config=config,
        training='2016-2019 all consecutive target hours after history warmup; chronological blocks; no date sampling or shuffle',
        station_sampling='82/410 condition stations fixed across all history and target hours in each block; physical groups held out',
        task='Current-hour spatial reconstruction, not future forecast',
        evaluation='2020 validation137, days8-10 continuous each month x3 layouts; already development-exposed',
        steps='fixed 3000; final weights primary, no 2020 checkpoint selection',
        features='actual p14 full carrier/point113 preserved; coast+terrain edge residual and FiLM; causal temporal 8x8 residual carrier',
        historical_units='p14 support helper height convention preserved for checkpoint compatibility; new edges explicit scaled physical coordinates',
        time='source nominal product hour; no guessed UTC/GPST conversion',
        dataset_contract_sha256=sha(args.data/'contract.json'),
        code_hashes={name:sha(Path(__file__).with_name(name)) for name in ['p14_fusion.py','p14_fusion_experiment.py','fixed_support.py','temporal_encoders.py']},
        weights={str(s):sha(args.weights/f's{s}.pt') for s in args.seeds},confirmation_C_D_read=False)
    if resuming:
        old=json.loads((args.out/'protocol.json').read_text(encoding='utf-8'))
        for key in ('steps','length','batch_hours','eval_hours','months','seeds','kinds'):
            if old['config'][key]!=config[key]:raise ValueError(f'Resume configuration changed: {key}')
        write_json(args.out/'resume_receipt.json',dict(updated_protocol=protocol,
            reason='Full-axis causal observation age fix before temporal arms; completed non-temporal results unchanged'))
    else:write_json(args.out/'protocol.json',protocol)
    data=FusionData(args.data,args.length);e=data.events
    stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
    blocks=chronological_blocks(args.length-1,stop,args.batch_hours)
    write_json(args.out/'time_schedule.json',dict(first_target=int(e.hours[args.length-1]),last_target=int(e.hours[stop-1]),
        unique_target_hours=stop-args.length+1,blocks_per_epoch=len(blocks),batch_hours=args.batch_hours,
        chronological=True,time_random_sampling=False,history_length=args.length,
        note='After last block, next epoch restarts at the first; window state is recomputed, not carried across epoch boundary'))
    results=json.loads((args.out/'summary.json').read_text(encoding='utf-8')) if resuming else []
    for seed in args.seeds:
        source=torch.load(args.weights/f's{seed}.pt',weights_only=True)
        if not any(r['kind']=='frozen' and r['seed']==seed for r in results):
            frozen=P14Fusion('p14',args.length).cuda();frozen.base.load_state_dict(source)
            folder=args.out/f'frozen_s{seed}';folder.mkdir(exist_ok=True)
            metrics=evaluate(data,frozen,args.eval_hours,args.months,folder)
            results.append(dict(kind='frozen',seed=seed,steps=0,metrics=metrics))
            write_json(args.out/'summary.json',results);del frozen
        for kind in args.kinds:
            if any(r['kind']==kind and r['seed']==seed for r in results):continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed+28011)
            model=P14Fusion(kind,args.length).cuda();model.base.load_state_dict(source)
            baseparams=list(model.base.parameters());base_ids={id(v) for v in baseparams}
            extra=[v for v in model.parameters() if id(v) not in base_ids and v.requires_grad]
            groups=[dict(params=baseparams,lr=1e-5)]
            if extra:groups.append(dict(params=extra,lr=1e-4))
            opt=torch.optim.AdamW(groups,weight_decay=1e-4)
            folder=args.out/f'{kind}_s{seed}'
            if folder.exists():
                archive=args.out/f'{kind}_s{seed}_interrupted_{int(time.time())}'
                assert folder.resolve().is_relative_to(args.out.resolve()) and archive.resolve().is_relative_to(args.out.resolve())
                folder.rename(archive)
            folder.mkdir()
            start=time.monotonic();trace=[]
            for step in range(1,args.steps+1):
                anchors=blocks[(step-1)%len(blocks)]
                # Station identities only are sampled, never target times.
                c=np.sort(rng.choice(410,82,False))
                b=data.batch(anchors,c,False,kind not in ('p14','features'),kind!='p14')
                model.train();prediction=model(b);mask=b['mask']
                loss=(prediction-b['target'])[mask].square().mean()
                if len(anchors)>1:
                    imask=mask[1:]&mask[:-1]
                    loss=loss+.1*((prediction[1:]-prediction[:-1])-(b['target'][1:]-b['target'][:-1]))[imask].square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2)
                if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient')
                opt.step()
                if step==1 or step%100==0 or step==args.steps:
                    row=dict(step=step,epoch=(step-1)//len(blocks),
                        first_hour=int(e.hours[anchors[0]]),last_hour=int(e.hours[anchors[-1]]),
                        training_block_rmse_mm=float(((prediction-b['target'])[mask].square().mean().sqrt()*data.std[0]).detach()),
                        loss=float(loss.detach()),seconds=time.monotonic()-start)
                    trace.append(row);write_json(folder/'progress.json',row)
                    print(json.dumps(dict(stage='train',kind=kind,seed=seed,**row)),flush=True)
                if step%1000==0:torch.save(model.state_dict(),folder/f'step_{step}.pt')
            torch.save(model.state_dict(),folder/'last.pt')
            write_json(folder/'fit.json',dict(kind=kind,seed=seed,steps=args.steps,trace=trace,
                parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.monotonic()-start))
            metrics=evaluate(data,model,args.eval_hours,args.months,folder)
            results.append(dict(kind=kind,seed=seed,steps=args.steps,metrics=metrics))
            write_json(args.out/'summary.json',results)
            print(json.dumps(dict(stage='evaluated',kind=kind,seed=seed,metrics=metrics)),flush=True)
            del model,opt;torch.cuda.empty_cache()
    write_json(args.out/'status.json',dict(status='COMPLETE_CONTINUOUS_P14_FUSION',results=len(results)))


if __name__=='__main__':main()
