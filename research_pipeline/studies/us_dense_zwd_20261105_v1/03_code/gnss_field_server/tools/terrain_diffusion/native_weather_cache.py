"""Resumable chronological caches for native station encoders and R/F residuals."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from .data import PROJECT,sha,write_json
from .p14_fusion import FusionData,P14Fusion,chronological_blocks
from .innovation_refinement import innovation_fields,causal_features
from .station_query import hole_schedule
from .station_weighting import distance_weights
from .native_weather import weather_fields,history_features,validate_origin


def static(geo,prefix=''):
    coast=geo[prefix+'coast']
    return np.c_[geo[prefix+'static'],np.log1p(coast)/6,coast/500].astype('float32')


@torch.no_grad()
def encode_block(data,model,beta,anchors,context,development=False):
    e=data.events;g=e.geo;start=int(anchors[0]);end=int(anchors[-1])
    # 72 station states +3 innovation lags +72 source frames =146 raw hours.
    allidx=np.arange(start-74,end+1);c=torch.as_tensor(context,device='cuda')
    batch=data.batch(allidx,context);physical=model(batch)*data.std[0]+data.mean[0]
    raw=data.values[torch.as_tensor(allidx,device='cuda')[:,None],c[None]]
    valid=torch.isfinite(raw[:,:,0]);errors=torch.where(valid,raw[:,:,0]-physical[:,c],0)
    cxy=torch.as_tensor(g['xy'][context],device='cuda');ch=torch.as_tensor(g['height'][context],device='cuda')
    hours=data.hours[torch.as_tensor(allidx,device='cuda')]
    cfields=innovation_fields(errors,valid,cxy,cxy,ch,ch)
    position=torch.arange(len(allidx),device='cuda')
    ci=causal_features(cfields,position,hours)[...,:9]
    ccorrection=(ci.double()@beta*10).clamp(-5,5).float()
    normalized=torch.where(torch.isfinite(raw),(raw-raw.new_tensor([0,280,900]))/raw.new_tensor([100,20,100]),0)
    ages=data.all_ages[torch.as_tensor(allidx,device='cuda')[:,None],c[None]].log1p()/5
    phase=hours.double()
    clock=torch.stack([(phase%24*2*torch.pi/24).sin(),(phase%24*2*torch.pi/24).cos(),
        (phase*2*torch.pi/(365.25*24)).sin(),(phase*2*torch.pi/(365.25*24)).cos()],-1).float()
    state=torch.cat([normalized,torch.isfinite(raw).float(),ages,errors[:,:,None]/10,
        (errors-ccorrection)[:,:,None]/10,clock[:,None].expand(-1,len(c),-1)],-1)[3:]
    prefix='dev_' if development else ''
    qxy=torch.as_tensor(g[prefix+'xy'],device='cuda');qh=torch.as_tensor(g[prefix+'height'],device='cuda')
    if development:
        del batch
        batch=data.batch(allidx,context,True);physical=model(batch)*data.std[0]+data.mean[0]
    fields=innovation_fields(errors,valid,qxy,cxy,qh,ch)
    innovation=causal_features(fields,position,hours)
    mean=physical+(innovation[...,:9].double()@beta*10).clamp(-5,5).float()
    weather=weather_fields(raw,qxy,cxy,qh,ch)
    positions=torch.arange(74,len(allidx),device='cuda')
    distance=torch.cdist(qxy,cxy,compute_mode='donot_use_mm_for_euclid_dist')
    nearest=torch.where(valid[positions,None],distance[None],torch.inf).min(-1).values
    condition=history_features(innovation,mean,weather,hours,positions,
        torch.as_tensor(static(g,prefix),device='cuda'),nearest)
    weights=distance_weights(qxy,cxy,valid[positions])
    source=e.dev_values if development else e.values
    origin_idx=np.searchsorted(e.dev_hours,e.hours[anchors]) if development else anchors
    truth=np.stack([np.asarray(source[origin_idx+h,:,0]) for h in range(7)],axis=-1)
    return dict(condition=condition.cpu().numpy(),history=state.cpu().numpy(),
        mean=mean[positions].cpu().numpy(),truth=truth,weights=weights.cpu().numpy(),
        nearest=nearest.cpu().numpy())


def create_arrays(out,count,queries):
    shapes=dict(condition=(count,12,queries,111),history=(count,83,82,15),
        mean=(count,12,queries),truth=(count,12,queries,7),weights=(count,12,queries),
        nearest=(count,12,queries),context_ids=(count,82),query_ids=(count,queries),
        origins=(count,12))
    arrays={}
    for name,shape in shapes.items():
        dtype='int64' if name=='origins' else 'int16' if name.endswith('_ids') else 'float32' if name in ('mean','truth','nearest') else 'float16'
        p=out/f'{name}.npy'
        arrays[name]=np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
        if arrays[name].shape!=shape:raise ValueError(f'cache shape differs: {name}')
    return arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--source',type=Path,default=PROJECT/'results/20261125_p14_continuous_fusion')
    p.add_argument('--innovation',type=Path,default=PROJECT/'results/20261125_observed_innovation_precision')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--blocks',type=int,default=3000)
    p.add_argument('--calendar-day',type=int,default=0);p.add_argument('--months',type=int,default=12)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.65)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        data_sha256=sha(args.data/'contract.json'),source_sha256=sha(args.source/f'gru_s{args.seed}/last.pt'),
        I1_sha256=sha(args.innovation/f'I1_history_s{args.seed}/coefficients.pt'),
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ('native_weather_cache.py','native_weather.py','innovation_refinement.py','p14_fusion.py')},
        strict_oof=False,confirmation_C_D_read=False,history_state_hours=72,raw_span_hours=146,
        time='continuous12h blocks, station identities only sampled; future0..6h labels only',
        training_years=[2016,2017,2018,2019],development_year=2020)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text(encoding='utf-8'))!=protocol:raise ValueError('immutable cache protocol differs')
    else:write_json(out/'protocol.json',protocol)
    data=FusionData(args.data);e=data.events
    pos=torch.arange(len(data.values),device='cuda')[:,None,None]
    last=torch.cummax(torch.where(torch.isfinite(data.values),pos,-1),dim=0).values
    data.all_ages=(pos-last).float();del pos,last
    model=P14Fusion('gru').cuda().eval();model.load_state_dict(torch.load(args.source/f'gru_s{args.seed}/last.pt',weights_only=True))
    beta=torch.load(args.innovation/f'I1_history_s{args.seed}/coefficients.pt',weights_only=True).cuda()
    if args.calendar_day:
        blocks=[];contexts=[]
        for c in e.layouts:
            for month in range(1,args.months+1):
                hour=int(np.datetime64(f'2020-{month:02d}-{args.calendar_day:02d}','h').astype('int64'))
                anchor=int(np.searchsorted(e.hours,hour))
                for offset in range(0,72,12):blocks.append(np.arange(anchor+offset,anchor+offset+12));contexts.append(c)
        count=len(blocks);queries=137;stop=len(e.hours)
    else:
        stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
        blocks=[b for b in chronological_blocks(145,stop-6,12) if len(b)==12]
        count=args.blocks;queries=128;holes=hole_schedule(count,args.seed)
    arrays=create_arrays(out,count,queries)
    status=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else dict(completed=0)
    start_time=time.monotonic()
    for index in range(status['completed'],count):
        anchors=blocks[index if args.calendar_day else index%len(blocks)]
        validate_origin(anchors,anchors,0,stop,146,6)
        rng=np.random.default_rng(args.seed+261100+index)
        if args.calendar_day:c=contexts[index]
        else:
            if holes[index]:
                center=int(rng.integers(410));pool=np.flatnonzero(np.linalg.norm(e.geo['xy']-e.geo['xy'][center],axis=1)>100)
            else:pool=np.arange(410)
            c=np.sort(rng.choice(pool,82,False))
        row=encode_block(data,model,beta,anchors,c,bool(args.calendar_day))
        if args.calendar_day:q=np.arange(137)
        else:
            hidden=~np.isin(e.geo['groups'],e.geo['groups'][c])
            available=np.isfinite(row['truth']).all((0,2))&hidden
            if available.sum()<queries:raise ValueError(f'only{available.sum()} complete hidden stations')
            q=np.sort(rng.choice(np.flatnonzero(available),queries,False))
        for name in ('condition','mean','truth','weights','nearest'):arrays[name][index]=row[name][:,q]
        arrays['history'][index]=row['history'];arrays['context_ids'][index]=c
        arrays['query_ids'][index]=q;arrays['origins'][index]=e.hours[anchors]
        if not all(np.isfinite(arrays[n][index]).all() for n in ('condition','history','mean','weights')):raise FloatingPointError('nonfinite cache')
        for a in arrays.values():a.flush()
        write_json(out/'status.json',dict(completed=index+1,total=count,status='COMPLETE' if index+1==count else 'BUILDING',
            seconds=time.monotonic()-start_time,last_origin_hour=int(e.hours[anchors[-1]])))
        if index==0 or (index+1)%100==0 or index+1==count:
            print(json.dumps(dict(stage='cache',seed=args.seed,completed=index+1,total=count,seconds=time.monotonic()-start_time)),flush=True)
    write_json(out/'geometry.json',dict(train_static=static(e.geo).tolist(),dev_static=static(e.geo,'dev_').tolist()))
    write_json(out/'manifest.json',dict(files={p.name:sha(p) for p in out.glob('*.npy')},protocol_sha256=sha(out/'protocol.json'),
        source_exposure='base source fitted2016-2019; not strict OOF',complete_blocks=count))


if __name__=='__main__':main()
