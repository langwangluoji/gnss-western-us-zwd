"""Loss-only H continuation with registered within-forecast increment supervision."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .data import sha,write_json
from .native_physics_lm import NativePhysicsLM,causal_station_loss
from .native_mean_fusion_v2 import load_source_state
from .native_mean_fusion_v2 import NativeMeanFusionV2
from .native_increment_loss import increment_mean_loss,within_forecast_increment_loss
from .native_model_seal import seal_completed_model,validate_model_seal
from .native_paths import file_path
from .native_weather_experiment import CachedData,mean_loss,checkpoint,stop_requested
from .native_weather_evaluation import calendar_shape,point_metrics


def continuation_optimizer(model,source,saved,steps=3000,old_lr=1e-4,new_lr=3e-4):
    """Restore source AdamW moments by parameter name; start a new cosine cycle."""
    old_names=[name for name,_ in source.named_parameters()]
    target=dict(model.named_parameters());new_names=model.new_parameter_names()
    groups=saved['optimizer']['param_groups']
    if len(groups)!=1 or len(groups[0]['params'])!=len(old_names):
        raise ValueError('source optimizer must match the registered one-group model.parameters order')
    if set(target)!=set(old_names)|new_names:raise ValueError('unregistered continuation parameter')
    old_group={k:copy.deepcopy(v) for k,v in groups[0].items() if k not in ('params','lr','initial_lr')}
    parameter_groups=[dict(old_group,params=[target[n] for n in old_names],lr=old_lr,initial_lr=old_lr)]
    if new_names:
        parameter_groups.append(dict(old_group,params=[p for n,p in model.named_parameters() if n in new_names],
            lr=new_lr,initial_lr=new_lr))
    optimizer=torch.optim.AdamW(parameter_groups)
    restored=optimizer.state_dict();old_ids=groups[0]['params'];target_ids=restored['param_groups'][0]['params']
    restored['state']={new:copy.deepcopy(saved['optimizer']['state'][old])
        for old,new in zip(old_ids,target_ids) if old in saved['optimizer']['state']}
    optimizer.load_state_dict(restored)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda update:.1+.9*.5*(1+math.cos(math.pi*min(update,steps)/steps)))
    return optimizer,scheduler


def predict_absolute(model,inputs,base):
    correction,embedding=model(**inputs)
    return base[:,:,None]+correction,embedding


def continuation_indices(count,steps,same_cache,smoke=False):
    if count<1 or steps<1:raise ValueError('positive cache and update budgets required')
    if smoke:
        if steps>4:raise ValueError('independent smoke cache is limited to four updates')
        return [i%count for i in range(steps)]
    if not same_cache:raise ValueError('formal continuation requires exact source training cache')
    if steps>count:raise ValueError('consume each registered cached block at most once')
    return list(range(steps))


def code_hashes():
    names=('native_increment_loss.py','native_increment_experiment.py','native_model_seal.py',
        'native_mean_fusion_v2.py','native_mean_fusion_v2_experiment.py',
        'native_weather_models.py','native_physics_lm.py','native_weather_experiment.py',
        'native_weather_cache.py','native_weather.py')
    return {name:sha(Path(__file__).with_name(name)) for name in names}


def verify_codes(protocol):
    current=code_hashes()
    if protocol['code_sha256']!=current:raise ValueError('frozen continuation code changed')


def require_export_contract(training,cache):
    for name in ('data_sha256','source_sha256','I1_sha256'):
        if cache[name]!=training['source_protocol'][name]:raise ValueError('increment export source contract differs: '+name)
    if cache['config']['seed']!=training['config']['seed']:raise ValueError('increment export cache seed differs')


def require_final_ema(weights,checkpoint_state):
    expected=checkpoint_state.get('ema',{})
    if checkpoint_state.get('step')!=3000 or weights.keys()!=expected.keys() or not expected:
        raise ValueError('EMA does not belong to final3000 checkpoint')
    if any(not torch.equal(weights[name],value) for name,value in expected.items()):
        raise ValueError('EMA differs from sealed final3000 checkpoint')


def train(args):
    if not args.smoke_cache and args.steps!=3000:raise ValueError('formal increment arm requires exactly3000 updates')
    if not args.smoke_cache and args.increment_weight!=.5:raise ValueError('formal increment weight must be fixed0.5')
    data=CachedData(args.cache,device=args.device)
    if data.development:raise ValueError('development cache cannot train a mean')
    source_protocol=json.loads((args.source/'protocol.json').read_text(encoding='utf-8'))
    config=source_protocol['config']
    if (config['kind'],config.get('variant'),config['length'],config['seed'])!=('transformer','physics_aux',72,args.seed):
        raise ValueError('expected same-seed registered transformer72_physics_aux source')
    indices=continuation_indices(len(data),args.steps,
        source_protocol['cache_manifest_sha256']==sha(args.cache/'manifest.json'),args.smoke_cache)
    for name in ('data_sha256','source_sha256','I1_sha256'):
        if data.protocol[name]!=source_protocol['source_protocol'][name]:raise ValueError('cache source contract differs')
    if source_protocol['model_code_sha256']!=sha(Path(__file__).with_name('native_weather_models.py')):
        raise ValueError('source model implementation changed')
    for name,expected in source_protocol.get('extra_model_sources',{}).items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('source physics implementation changed')
    source_path=args.source/'checkpoint_3000.pt'
    # Keep Adam's non-capturable step counters and RNG byte states on CPU;
    # load_state_dict moves model weights/moments to each target parameter device.
    saved=torch.load(source_path,map_location='cpu',weights_only=True)
    if saved['step']!=3000:raise ValueError('source checkpoint must be exactly step3000')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    source=NativePhysicsLM('transformer',64,72).to(args.device).eval()
    source.load_state_dict(saved['model'])
    model=NativeMeanFusionV2('H').to(args.device);load_source_state(model,saved['model'])
    inputs,_,_,_=data.batch(0,72)
    model.eval()
    with torch.no_grad():
        expected=source(**inputs,return_aux=True);actual=model(**inputs,return_aux=True)
        difference=[float((a-b).abs().max()) for a,b in zip(actual,expected)]
        if max(difference)>2e-5:raise ValueError(f'nonzero-source warm start differs: {difference}')
        if not torch.count_nonzero(source.head.weight):raise ValueError('source head unexpectedly zero')
    optimizer,scheduler=continuation_optimizer(model,source,saved,args.steps,args.old_lr,args.new_lr)
    ema=copy.deepcopy(model) # All arms start EMA at the same raw source weights.
    torch.set_rng_state(saved['torch_rng'].cpu())
    if args.device.startswith('cuda'):torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_checkpoint_sha256=sha(source_path),source_protocol_sha256=sha(args.source/'protocol.json'),
        source_final_weights_sha256=sha(args.source/'last.pt'),source_step=3000,
        source=dict(source_checkpoint_sha256=sha(source_path),source_protocol_sha256=sha(args.source/'protocol.json'),
            source_weights_sha256=sha(args.source/'last.pt'),cache_manifest_sha256=sha(args.cache/'manifest.json')),
        cache_manifest_sha256=sha(args.cache/'manifest.json'),source_protocol=data.protocol,
        code_sha256=code_hashes(),new_parameter_names=sorted(model.new_parameter_names()),
        parameters=sum(p.numel() for p in model.parameters()),warm_start_max_difference=difference,
        warm_source_correction_rms_mm=float(expected[0].square().mean().sqrt()),
        optimizer=dict(old_state='source AdamW moments and steps restored by name; new parameters uninitialized',
            old_lr=args.old_lr,new_lr=args.new_lr,weight_decay=optimizer.param_groups[0]['weight_decay'],
            scheduler='explicit new cosine cycle; eta_min=0.1*group_initial_lr',cycle_updates=args.steps,
            initial_group_lrs=scheduler.get_last_lr()),
        cache_block_order=indices if args.smoke_cache else '0..steps-1, identical to registered C/H/S',
        experimental_mode='ENGINEERING_SMOKE_ONLY' if args.smoke_cache else 'MATCHED_CONTINUATION',
        aux_weight=.25,ema_decay=.999,
        ema_initialization='raw source, not historical EMA',strict_oof=False,confirmation_C_D_read=False)
    write_json(args.out/'protocol.json',protocol)
    del source,saved,expected,actual
    updates=0;start=time.monotonic();trace=[]
    for step in range(args.steps):
        index=indices[step];inputs,truth,base,weights=data.batch(index,72);model.train()
        correction,_,aux=model(**inputs,return_aux=True)
        auxiliary,cells=causal_station_loss(aux,inputs['history'])
        query=increment_mean_loss(correction,truth,base,weights,args.increment_weight);loss=query+.25*auxiliary
        increment=within_forecast_increment_loss(correction,truth,base,weights)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite mean continuation loss')
        if bool(torch.isfinite(truth).any()) or cells>0:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();scheduler.step();updates+=1
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.001)
        if step==0 or (step+1)%100==0 or step+1==args.steps:
            row=dict(step=step+1,supervised_updates=updates,loss=float(loss.detach()),
                query_loss=float(query.detach()),within_forecast_increment_loss=float(increment.detach()),
                causal_station_loss=float(auxiliary.detach()),
                auxiliary_label_cells=cells,finite_label_cells=int(torch.isfinite(truth).sum()),
                cache_block=index,last_origin_hour=int(data.a['origins'][index,-1]),group_lrs=scheduler.get_last_lr(),
                current_correction_rms_mm=float(correction[:,:,0].detach().square().mean().sqrt()),
                future6_correction_rms_mm=float(correction[:,:,6].detach().square().mean().sqrt()),
                seconds=time.monotonic()-start)
            row['current_readout_weight_norm']=float(model.current_residual.weight.detach().norm())
            row['current_readout_bias_norm']=float(model.current_residual.bias.detach().norm())
            trace.append(row);write_json(args.out/'curve.json',trace);write_json(args.out/'progress.json',row)
            print(json.dumps(dict(arm=args.arm,seed=args.seed,**row)),flush=True)
        if (step+1)%500==0 or step+1==args.steps or stop_requested():
            checkpoint(args.out,step+1,model,optimizer,scheduler,ema)
        if stop_requested():
            write_json(args.out/'status.json',dict(status='STOPPED_AT_BUDGET',blocks_consumed=step+1,supervised_updates=updates))
            raise SystemExit(75)
    torch.save(model.state_dict(),args.out/'last.pt');torch.save(ema.state_dict(),args.out/'ema.pt')
    write_json(args.out/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if args.smoke_cache else 'COMPLETE_MEAN_INCREMENT',
        steps=args.steps,blocks_consumed=args.steps,
        supervised_updates=updates,seconds=time.monotonic()-start,final_group_lrs=scheduler.get_last_lr(),
        ema_weights_sha256=sha(args.out/'ema.pt')))
    if not args.smoke_cache:seal_completed_model(args.out,'COMPLETE_MEAN_INCREMENT',code_hashes())


@torch.no_grad()
def export(args):
    data=CachedData(args.cache,device=args.device)
    protocol=json.loads((args.mean/'protocol.json').read_text(encoding='utf-8'));verify_codes(protocol)
    require_export_contract(protocol,data.protocol)
    validate_model_seal(args.mean,'COMPLETE_MEAN_INCREMENT',code_hashes())
    model=NativeMeanFusionV2('H').to(args.device).eval()
    state=torch.load(args.mean/f'{args.weights}.pt',map_location='cpu',weights_only=True)
    if args.weights=='ema':
        status=json.loads((args.mean/'status.json').read_text(encoding='utf-8'))
        if sha(args.mean/'ema.pt')!=status['ema_weights_sha256']:raise ValueError('sealed EMA file changed')
        final=torch.load(args.mean/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
        require_final_ema(state,final)
        del final
    model.load_state_dict(state)
    args.out.mkdir(parents=True,exist_ok=False);n=len(data);B,Q=data.a['mean'].shape[1:]
    embedding=np.lib.format.open_memmap(args.out/'embedding.npy',mode='w+',dtype='float16',shape=(n,B,Q,64))
    prediction=np.lib.format.open_memmap(args.out/'mean.npy',mode='w+',dtype='float32',shape=(n,B,Q,7))
    square=np.zeros(7);summed=np.zeros(7);count=np.zeros(7)
    for i in range(n):
        inputs,truth,base,_=data.batch(i,72);pred,h=predict_absolute(model,inputs,base)
        if not torch.isfinite(pred).all():raise FloatingPointError('nonfinite absolute mean')
        embedding[i]=h.cpu().numpy();prediction[i]=pred.cpu().numpy()
        error=(truth-pred).cpu().numpy().astype('float64')
        square+=np.nansum(error*error,(0,1));summed+=np.nansum(error,(0,1));count+=np.isfinite(error).sum((0,1))
        if i==0 or (i+1)%500==0:print(json.dumps(dict(stage='v2_export',completed=i+1,total=n)),flush=True)
    embedding.flush();prediction.flush()
    bias=np.divide(summed,count,out=np.full(7,np.nan),where=count>0)
    variance=np.divide(square,count,out=np.full(7,np.nan),where=count>0)-bias*bias
    write_json(args.out/'statistics.json',dict(bias_mm=bias.tolist(),scale_mm=np.sqrt(np.maximum(variance,1)).tolist(),
        count=count.tolist(),role='descriptive_development_not_fit' if data.development else 'training_only'))
    write_json(args.out/'manifest.json',dict(cache=str(args.cache.resolve()),cache_manifest_sha256=sha(args.cache/'manifest.json'),
        mean_protocol_sha256=sha(args.mean/'protocol.json'),mean_weights_sha256=sha(args.mean/f'{args.weights}.pt'),
        model_kind='transformer',mean_variant='increment_'+protocol['config']['arm'],weights=args.weights,
        files={name:sha(args.out/name) for name in ('embedding.npy','mean.npy')},complete=True,
        experimental_mode=protocol['experimental_mode'],
        absolute_mean_definition='cache.mean[:,:,None] + bounded correction_mm',code_sha256=code_hashes()))


@torch.no_grad()
def evaluate(args):
    data=CachedData(args.cache,device=args.device)
    if not data.development:raise ValueError('evaluation requires a registered development calendar')
    manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_manifest_sha256']!=sha(args.cache/'manifest.json'):raise ValueError('embedding calendar differs')
    for name,expected in manifest['files'].items():
        if sha(args.embeddings/name)!=expected:raise ValueError('embedding file differs')
    if manifest.get('mean_variant')!='increment_H_delta' or manifest.get('code_sha256')!=code_hashes():raise ValueError('increment embedding implementation differs')
    months=data.protocol['config']['months']
    truth=calendar_shape(np.asarray(data.a['truth']),months)
    prediction=calendar_shape(np.load(args.embeddings/'mean.npy'),months)
    reference=calendar_shape(np.repeat(data.a['mean'][...,None],7,-1),months)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/'evaluation.npz',prediction=prediction,reference=reference,truth=truth)
    payload=dict(reference=point_metrics(reference,truth),deterministic_mean=point_metrics(prediction,truth))
    write_json(args.out/'metrics.json',payload)
    write_json(args.out/'lineage.json',dict(cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),code_sha256=code_hashes(),
        prediction_sha256=sha(args.out/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['train','export','evaluate'],required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--mean',type=Path);parser.add_argument('--embeddings',type=Path)
    parser.add_argument('--increment-weight',type=float,default=.5)
    parser.add_argument('--arm',choices=['H_delta'],default='H_delta');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=3000);parser.add_argument('--old-lr',type=float,default=1e-4)
    parser.add_argument('--new-lr',type=float,default=3e-4);parser.add_argument('--weights',choices=['last','ema'],default='last')
    parser.add_argument('--smoke-cache',action='store_true',help='engineering only: independent same-source cache, at most4 updates')
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    required={'train':'source','export':'mean','evaluate':'embeddings'}[args.phase]
    if getattr(args,required) is None:parser.error(f'--{required} is required for {args.phase}')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.4)
    {'train':train,'export':export,'evaluate':evaluate}[args.phase](args)


if __name__=='__main__':main()
