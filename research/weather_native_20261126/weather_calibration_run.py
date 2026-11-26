"""Final matched weather-scale/tail calibration: frozen point mean and source field."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from campaign import RESULTS,read,STOP
from weather_calibration_cache import cache_codes
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_calibration import WeatherCalibrator,loss,transform_ensemble
from tools.terrain_diffusion.native_student_process import log_density
from tools.terrain_diffusion.native_weather_experiment import checkpoint
from tools.terrain_diffusion.native_model_seal import validate_model_seal,seal_completed_model,evaluation_marker
from tools.terrain_diffusion.native_inducing_flow_experiment import codes as source_codes
from tools.terrain_diffusion.native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps,safe_mean
from inducing_calibration_audit import rank_coverage
from weather_calibration_lock import LOCK,validate_lock,create_lock,require_lineage


ROOT=RESULTS/'20261126_weather_scale_tail'
ARMS=('G_weather','G_extra','T_weather')
VARIANTS=('fixed','constant','G_weather','G_extra','T_constant','T_weather')
GAMMA_SEED=261128


def codes():
    engine=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/tools/terrain_diffusion'
    return {**source_codes(),**{n:sha(engine/n) for n in ('native_student_process.py','native_weather_calibration.py')},
        'runner':sha(__file__),'lock_helper':sha(Path(__file__).with_name('weather_calibration_lock.py'))}


def cached(seed,day=0):
    folder=ROOT/f'cache_{"calendar8" if day else "train"}_s{seed}';manifest=read(folder/'manifest.json');protocol=read(folder/'protocol.json')
    if not manifest['complete'] or manifest['code_sha256']!=cache_codes() or protocol['seed']!=seed or protocol['calendar_day']!=day:
        raise ValueError('registered calibration cache differs')
    if manifest['protocol_sha256']!=sha(folder/'protocol.json'):raise ValueError('cache protocol differs')
    for n,v in manifest['files'].items():
        if sha(folder/n)!=v:raise ValueError('likelihood/weather statistic changed')
    arrays={n:np.load(folder/(n+'.npy'),mmap_mode='r') for n in ('q','logdet','dimension','weather','origins')}
    if arrays['q'].shape!=((216 if day else 3000),12):raise ValueError('full registered clock required')
    return folder,arrays,read(folder/'statistics.json')


def tensor(x,dtype=torch.float64):return torch.as_tensor(np.array(x),dtype=dtype)


def features(a,stats):
    x=tensor(a['weather'],torch.float32)
    return (x-torch.tensor(stats['weather_mean'],dtype=torch.float32))/torch.tensor(stats['weather_std'],dtype=torch.float32)


def model_load(seed,arm):
    folder=ROOT/f'{arm}_s{seed}';seal=validate_model_seal(folder,'COMPLETE_WEATHER_CALIBRATION',codes());p=read(folder/'protocol.json')
    if p['arm']!=arm or p['seed']!=seed:raise ValueError('calibration family/seed differs')
    if p['source']['training_cache_manifest_sha256']!=sha(ROOT/f'cache_train_s{seed}/manifest.json') or p['source']['normalization_sha256']!=sha(ROOT/f'cache_train_s{seed}/statistics.json'):
        raise ValueError('training covariance or normalization source changed')
    if arm!='G_weather' and p['source']['conditional_covariance_G_weather_weights_sha256']!=sha(ROOT/f'G_weather_s{seed}/last.pt'):
        raise ValueError('frozen weather covariance source changed')
    if p['source'].get('final_lock_sha256')!=sha(LOCK):raise ValueError('model does not belong to final locked experiment')
    model=WeatherCalibrator('tail' if arm=='T_weather' else 'scale').eval()
    model.load_state_dict(torch.load(folder/'last.pt',map_location='cpu',weights_only=True));model.requires_grad_(False)
    return model,seal


def train(seed,arm,smoke=False):
    cache,a,stats=cached(seed);steps=4 if smoke else 3000
    folder=ROOT/(f'smoke_{arm}_s{seed}' if smoke else f'{arm}_s{seed}')
    if folder.exists():
        if smoke:raise ValueError('preserve existing smoke evidence')
        model_load(seed,arm);return
    x=features(a,stats);q=tensor(a['q']);ld=tensor(a['logdet']);d=tensor(a['dimension'],torch.long)
    source=None;base=torch.ones_like(q)
    if arm!='G_weather':
        base_model,source=model_load(seed,'G_weather')
        with torch.no_grad():base=base_model(x.reshape(-1,90)).reshape_as(q).double()
    torch.manual_seed(seed if arm=='G_weather' else seed+5300)
    model=WeatherCalibrator('tail' if arm=='T_weather' else 'scale')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=3e-5)
    folder.mkdir(parents=True,exist_ok=False)
    source_receipt=dict(training_cache_manifest_sha256=sha(cache/'manifest.json'),normalization_sha256=sha(cache/'statistics.json'),
        conditional_covariance_G_weather_weights_sha256=source['weights_sha256'] if source else None,
        affine_source=read(cache/'protocol.json')['source_model_seal_sha256'],final_lock_sha256=None if smoke else sha(LOCK))
    write_json(folder/'protocol.json',dict(seed=seed,arm=arm,source=source_receipt,steps=steps,code_sha256=codes(),parameters=sum(p.numel() for p in model.parameters()),
        optimizer='AdamW lr3e-4 weight_decay0 clip1 cosine->3e-5',mean='frozen TF3000',input='training-normalized historical90 only',
        covariance='frozen G_weather covariance' if arm=='T_weather' else 'positive Gaussian variance scaling',
        strict_oof=False,confirmation_C_D_read=False))
    trace=[];updates=0;total=0;start=time.monotonic()
    for i in range(steps):
        if STOP.exists():
            checkpoint(folder,i,model,optimizer,scheduler);write_json(folder/'status.json',dict(status='STOPPED_AT_BUDGET',steps=i));raise SystemExit(75)
        output=model(x[i]);nu=output if arm=='T_weather' else None
        s2=base[i] if nu is not None else base[i]*output.double()
        value,count=loss(q[i],ld[i],d[i],s2,nu)
        if not torch.isfinite(value):raise FloatingPointError('nonfinite weather calibration loss')
        if count:
            optimizer.zero_grad();value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite calibration gradient')
            optimizer.step();scheduler.step();updates+=1;total+=int(d[i].sum())
        if i==0 or (i+1)%100==0 or i+1==steps:
            row=dict(step=i+1,supervised_updates=updates,nll_per_dimension=float(value.detach()),finite_target_cells=int(d[i].sum()),
                total_finite_target_cells=total,active_origins=count,seconds=time.monotonic()-start)
            trace.append(row);write_json(folder/'curve.json',trace);write_json(folder/'progress.json',row);print(dict(seed=seed,arm=arm,**row),flush=True)
    checkpoint(folder,steps,model,optimizer,scheduler);torch.save(model.state_dict(),folder/'last.pt')
    expected=int((np.asarray(a['dimension'])[:steps]>0).any(-1).sum())
    if updates!=expected:raise ValueError('supervised clock differs from actual label support')
    write_json(folder/'status.json',dict(status='ENGINEERING_SMOKE_COMPLETE' if smoke else 'COMPLETE_WEATHER_CALIBRATION',steps=steps,
        supervised_updates=updates,seconds=time.monotonic()-start,finite_target_cells=total))
    if not smoke:seal_completed_model(folder,'COMPLETE_WEATHER_CALIBRATION',codes())


def constants(seed):
    folder,a,stats=cached(seed);model,seal=model_load(seed,'G_weather');x=features(a,stats)
    with torch.no_grad():s2=model(x.reshape(-1,90)).reshape(a['q'].shape).double()
    q=tensor(a['q']);ld=tensor(a['logdet']);d=tensor(a['dimension'],torch.long);valid=d>0
    names=[None,3,4,6,8,12,20,40,80,160,1000]
    scores={str(v):float((-log_density(q[valid]/s2[valid],ld[valid]+d[valid].double()*s2[valid].log(),d[valid],v)/d[valid]).mean()) for v in names}
    chosen=min(names,key=lambda v:scores[str(v)])
    receipt=dict(seed=seed,nu=chosen,grid_nll_per_dimension=scores,training_cache_manifest_sha256=sha(folder/'manifest.json'),
        covariance_G_weather_weights_sha256=seal['weights_sha256'],constant_scale2=stats['constant_scale2'],role='training-only analytic fit, not3000 neural updates',final_lock_sha256=sha(LOCK))
    out=ROOT/f'constants_s{seed}.json'
    if out.exists():
        if read(out)!=receipt:raise ValueError('frozen constant probe changed')
    else:write_json(out,receipt)
    return receipt


@torch.no_grad()
def evaluate(seed):
    train_cache,_,stats=cached(seed);dev_cache,a,_=cached(seed,8);x=features(a,stats)
    g,gseal=model_load(seed,'G_weather');extra,eseal=model_load(seed,'G_extra');tail,tseal=model_load(seed,'T_weather')
    base=g(x.reshape(-1,90)).reshape(a['q'].shape).numpy();more=extra(x.reshape(-1,90)).reshape(a['q'].shape).numpy()
    nu=tail(x.reshape(-1,90)).reshape(a['q'].shape).numpy();const=constants(seed)
    old=RESULTS/f'20261126_inducing_latent_flow/calendar8/affine_time_s{seed}'
    parent=validate_model_seal(RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}','COMPLETE_INDUCING_FLOW',source_codes())
    evaluation_marker(old,parent);lineage=read(old/'lineage.json')
    original=read(dev_cache/'protocol.json')
    if lineage['cache_manifest_sha256']!=original['cache_manifest_sha256'] or lineage['embedding_manifest_sha256']!=original['embedding_manifest_sha256']:
        raise ValueError('parent forecast source differs from likelihood cache')
    if original['source_model_seal_sha256']!=sha(RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}/model_artifact_seal.json'):
        raise ValueError('likelihood covariance and Gaussian member source differ')
    for name in VARIANTS:
        if STOP.exists():raise SystemExit(75)
        out=ROOT/f'calendar8/{name}_s{seed}'
        primary={'G_weather':gseal,'G_extra':eseal,'T_weather':tseal}.get(name,gseal)
        expected_lineage=dict(name=name,seed=seed,members=32,code_sha256=codes(),final_lock_sha256=sha(LOCK),
            source_evaluation_sha256=lineage['prediction_sha256'],cache_manifest_sha256=lineage['cache_manifest_sha256'],
            embedding_manifest_sha256=lineage['embedding_manifest_sha256'],training_statistic_manifest_sha256=sha(train_cache/'manifest.json'),
            development_statistic_manifest_sha256=sha(dev_cache/'manifest.json'),constants_sha256=sha(ROOT/f'constants_s{seed}.json'),
            G_weather_weights_sha256=gseal['weights_sha256'],G_extra_weights_sha256=eseal['weights_sha256'],T_weather_weights_sha256=tseal['weights_sha256'],
            process_protocol_sha256=primary['protocol_sha256'],probability_weights_sha256=primary['weights_sha256'],
            independent_iid_member_noise=True,gamma_seed=GAMMA_SEED if name=='T_weather' or (name=='T_constant' and const['nu'] is not None) else None,
            gamma_shared='all137 queries and0..6h per member/origin; independent of source Gaussian',strict_oof=False,confirmation_C_D_read=False)
        if out.exists():
            evaluation_marker(out,primary);require_lineage(read(out/'lineage.json'),expected_lineage);continue
        out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
        with np.load(old/'evaluation.npz') as data:
            mu=data['prediction'];truth=data['truth'];source_samples=data['samples'];source_std=data['gaussian_distribution_std']
        s2=np.ones_like(base) if name=='fixed' else np.full_like(base,const['constant_scale2']) if name=='constant' else base*more if name=='G_extra' else base
        v=nu if name=='T_weather' else np.full_like(nu,const['nu']) if name=='T_constant' and const['nu'] is not None else None
        sc=calendar_shape(s2,12)
        if source_samples.shape!=(32,3,12,72,137,7):raise ValueError('full registered ensemble required')
        transformed=transform_ensemble(torch.from_numpy(source_samples.reshape(32,-1,137,7)),torch.from_numpy(mu.reshape(-1,137,7)),
            torch.from_numpy(sc.reshape(-1)),None if v is None else torch.from_numpy(calendar_shape(v,12).reshape(-1)),GAMMA_SEED)
        samples=transformed.numpy().reshape(source_samples.shape);del source_samples,transformed
        if not np.isfinite(samples).all():raise FloatingPointError('nonfinite calibrated field')
        crps=fair_crps(samples,truth);coverage=rank_coverage(samples,truth)
        q=tensor(a['q']);ld=tensor(a['logdet']);d=tensor(a['dimension'],torch.long);valid=d>0
        scale=tensor(s2);degrees=None if v is None else tensor(v)
        joint=np.full(q.shape,np.nan);joint[valid.numpy()]=(-log_density(q[valid]/scale[valid],ld[valid]+d[valid].double()*scale[valid].log(),d[valid],None if degrees is None else degrees[valid])/d[valid]).numpy()
        std=source_std*np.sqrt(sc)[...,None,None]
        payload=dict(deterministic_mean=point_metrics(mu,truth),sample_mean=point_metrics(samples.mean(0),truth),
            fair_crps_mm_by_horizon=safe_mean(crps,(0,1,2,3)).tolist(),
            forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1)))),
            randomized80_by_horizon=safe_mean(coverage['randomized80'],(0,1,2,3)).tolist(),
            negative_sample_fraction_by_horizon=(samples<0).mean((0,1,2,3,4)).tolist(),joint_nll_per_dimension=float(np.nanmean(joint)),
            seconds_transform_and_scoring=time.monotonic()-start,analytic_point_mean='frozen TF3000')
        artifacts=dict(prediction=mu,truth=truth,samples=samples,crps=crps.astype('float32'),nll_per_dimension=calendar_shape(joint,12),
            variance_scale2=sc,distribution_std=std)
        if v is None:
            artifacts.update(gaussian_distribution_mean=mu,gaussian_distribution_std=std)
            payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(gaussian_crps(mu,std,truth),(0,1,2,3)).tolist()
        else:artifacts['nu']=calendar_shape(v,12)
        np.savez_compressed(out/'evaluation.npz',**artifacts);write_json(out/'metrics.json',payload)
        write_json(out/'lineage.json',dict(**expected_lineage,prediction_sha256=sha(out/'evaluation.npz')))
        print(dict(seed=seed,name=name,F6=payload['fair_crps_mm_by_horizon'][6],NLL=payload['joint_nll_per_dimension']),flush=True)
        del samples,artifacts,crps,coverage,std,mu,truth
    for name in VARIANTS:
        out=ROOT/f'calendar8/{name}_s{seed}'
        if not (out/'joint_metrics.json').exists():
            if STOP.exists():raise SystemExit(75)
            original_cache=source_cache(seed)
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),'--evaluation',str(out),'--cache',str(original_cache),'--out',str(out)],check=True)


def source_cache(seed):
    from inducing_flow_series import source_paths
    return source_paths(seed,8)['cache']


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--phase',choices=['lock','smoke','train','evaluate'],required=True)
    args=p.parse_args();torch.set_num_threads(2)
    if args.phase=='lock':create_lock();return
    if args.phase!='smoke':validate_lock()
    if args.phase=='smoke':train(args.seed,'G_weather',True)
    elif args.phase=='train':
        for arm in ARMS:train(args.seed,arm)
        constants(args.seed)
    else:evaluate(args.seed)


if __name__=='__main__':main()
