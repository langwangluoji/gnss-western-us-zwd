"""Exact source-background leave-out innovations; separate from trained OOF."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_paths import file_path


def spatial_groups(context,xy,physical_groups,count=8):
    context=np.asarray(context);xy=np.asarray(xy);physical_groups=np.asarray(physical_groups)
    labels=np.unique(physical_groups)
    labels=sorted(labels,key=lambda g:int(context[physical_groups==g].min()))
    representatives=np.array([xy[physical_groups==g].mean(0) for g in labels])
    count=min(count,len(labels));centers=[0]
    distance=np.linalg.norm(representatives[:,None]-representatives[None],axis=-1)
    while len(centers)<count:
        nearest=distance[:,centers].min(1);nearest[centers]=-1
        centers.append(int(np.argmax(nearest)))
    assignment=distance[:,centers].argmin(1)
    for fold,index in enumerate(centers):assignment[index]=fold
    return np.array([assignment[labels.index(g)] for g in physical_groups],dtype='int64')


def retained_context(context,withheld):
    result=np.sort(np.asarray(context)[~np.isin(context,withheld)])
    if not len(result):raise ValueError('leave-out context must remain nonempty')
    return result


def codes():
    names=['native_innovation_audit.py','p14_fusion.py','innovation_refinement.py','native_weather_cache.py',
        'native_weather_experiment.py','unet_foundation.py','fixed_support.py','temporal_encoders.py','temporal_experiment.py']
    return {name:sha(Path(__file__).with_name(name)) for name in names}


@torch.no_grad()
def main():
    from .p14_fusion import FusionData,P14Fusion
    from .native_weather_experiment import CachedData
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--embedding',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=int,default=0)
    p.add_argument('--origins',type=int,default=2);args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    if not 1<=args.origins<=12:raise ValueError('origins must be1..12 consecutive cached origins')
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.7)
    start=time.monotonic();cache=CachedData(args.cache)
    if cache.development:raise ValueError('initial audit uses training-pool geometry only')
    protocol=cache.protocol;seed=protocol['config']['seed'];manifest=json.loads((args.embedding/'manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_manifest_sha256']!=sha(args.cache/'manifest.json') or manifest['weights']!='last':raise ValueError('frozen mean cache or weights differ')
    if sha(args.embedding/'mean.npy')!=manifest['files']['mean.npy']:raise ValueError('mean array differs')
    args.out.mkdir(parents=True,exist_ok=False)
    data=FusionData(file_path(protocol['config']['data']));e=data.events
    source=file_path(protocol['config']['source']);source_weights=source/f'gru_s{seed}/last.pt'
    model=P14Fusion('gru').cuda().eval();model.load_state_dict(torch.load(source_weights,map_location='cpu',weights_only=True))
    hours=np.asarray(cache.a['origins'][args.index,:args.origins]);anchors=np.searchsorted(e.hours,hours)
    if not np.array_equal(e.hours[anchors],hours) or not np.array_equal(np.diff(anchors),np.ones(len(anchors)-1)):
        raise ValueError('audit origins must match consecutive original hours')
    positions=np.arange(int(anchors[0])-3,int(anchors[-1])+1);context=np.asarray(cache.a['context_ids'][args.index]).astype('int64')
    query=np.asarray(cache.a['query_ids'][args.index]).astype('int64')
    if np.isin(e.geo['groups'][query],e.geo['groups'][context]).any():raise ValueError('query physical groups overlap conditions')
    folds=spatial_groups(context,e.geo['xy'][context],e.geo['groups'][context])
    raw=data.values[torch.as_tensor(positions,device='cuda')[:,None],torch.as_tensor(context,device='cuda')[None],0].cpu().numpy()
    timings={};calls={}

    def background(c):
        batch=data.batch(positions,c)
        return (model(batch)*data.std[0]+data.mean[0]).cpu().numpy()

    t=time.monotonic();full=background(context);timings['self']=time.monotonic()-t;calls['self']=1
    self_error=np.where(np.isfinite(raw),raw-full[:,context],np.nan)
    cached=np.asarray(cache.a['history'][args.index,68:71+args.origins,:,9],dtype='float32')*10
    discrepancy=float(np.nanmax(np.abs(cached-self_error)))
    if discrepancy>.05:raise ValueError(f'raw self innovation differs from cache: {discrepancy}mm')
    errors={'self':self_error};calls['LOSO']=len(context);calls['LOGO']=len(np.unique(folds))
    for arm in ['LOSO','LOGO']:
        t=time.monotonic();result=np.full_like(raw,np.nan)
        groups=[np.array([j]) for j in range(len(context))] if arm=='LOSO' else [np.flatnonzero(folds==f) for f in np.unique(folds)]
        for number,indices in enumerate(groups):
            retained=retained_context(context,context[indices]);prediction=background(retained)
            result[:,indices]=raw[:,indices]-prediction[:,context[indices]]
            if number==0 or (number+1)%10==0 or number+1==len(groups):
                row=dict(arm=arm,completed=number+1,total=len(groups),seconds=time.monotonic()-t)
                write_json(args.out/'progress.json',row);print(json.dumps(row),flush=True)
        errors[arm]=result;timings[arm]=time.monotonic()-t
    # Verify actual information isolation, including all72h source-history inputs.
    j=int(context[0]);retained=retained_context(context,np.array([j]));before=background(retained)
    left=int(positions[0])-71;right=int(positions[-1])+1;original=data.values[left:right,j].clone()
    try:
        data.values[left:right,j]=data.values.new_tensor([1000.,350.,1500.])
        after=background(retained)
    finally:data.values[left:right,j]=original
    if not np.array_equal(before,after):raise ValueError('withheld station observation still affects leave-out background')
    mean=np.asarray(np.load(args.embedding/'mean.npy',mmap_mode='r')[args.index,:args.origins])
    truth=np.asarray(cache.a['truth'][args.index,:args.origins])
    np.savez_compressed(args.out/'audit_arrays.npz',context=context,query=query,folds=folds,hours=hours,
        self_error=errors['self'],LOSO_error=errors['LOSO'],LOGO_error=errors['LOGO'],
        query_target_error=truth-mean,query_truth=truth,query_mean=mean,query_static=e.geo['static'][query],
        context_xy=e.geo['xy'][context],query_xy=e.geo['xy'][query],context_height=e.geo['height'][context],query_height=e.geo['height'][query])
    rms={name:float(np.sqrt(np.nanmean(value.astype('float64')**2))) for name,value in errors.items()}
    result=dict(complete=True,kind='ENGINEERING_INFORMATION_AUDIT_ONLY',seed=seed,cache_index=args.index,origins=args.origins,
        source_hours=positions.size,history_span_hours=right-left,context_stations=len(context),spatial_groups=int(folds.max()+1),
        self_cache_max_difference_mm=discrepancy,withheld_input_perturbation_max_difference_mm=float(np.max(np.abs(before-after))),
        innovation_rms_mm=rms,rms_ratio_to_self={name:value/rms['self'] for name,value in rms.items()},
        calls=calls,seconds=timings,total_seconds=time.monotonic()-start,code_sha256=codes(),
        cache_manifest_sha256=sha(args.cache/'manifest.json'),cache_protocol_sha256=sha(args.cache/'protocol.json'),
        mean_manifest_sha256=sha(args.embedding/'manifest.json'),mean_array_sha256=sha(args.embedding/'mean.npy'),
        p14_source_weights_sha256=sha(source_weights),source_data_contract_sha256=sha(data.events.data/'contract.json'),
        arrays_sha256=sha(args.out/'audit_arrays.npz'),strict_oof=False,confirmation_C_D_read=False,
        note='station leave-out removes all its GNSS/T/P across source history; background weights still fitted2016-2019; raw RMS is not predictive gain')
    write_json(args.out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
