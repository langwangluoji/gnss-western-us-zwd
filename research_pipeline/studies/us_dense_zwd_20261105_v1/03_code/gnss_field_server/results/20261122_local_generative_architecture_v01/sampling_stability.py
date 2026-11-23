"""Fixed-checkpoint NDP sampling audit; no training or checkpoint re-selection."""
import numpy as np
import torch
from pilot_data import ROOT,Data,monthly_subset
from pilot_models import SetDenoiser
from gnss_field.models.diffusion import keyed_noise,heun_sample
from gnss_field.artifacts import immutable_json,write_json,read_json,sha256
from grid_generation_metrics import score_ensemble
from gnss_field.evaluation import point_metrics


def audit_dependencies(root,names):
    records={};model_hash=sha256(root/'pilot_models.py');factor=sha256(root/'GP_correlation_factor.npy')
    for name in names:
        directory=root/name;result=read_json(directory/'result.json')
        record=dict(result=sha256(directory/'result.json'))
        if 'checkpoint' in result:
            checkpoint=sha256(directory/result['checkpoint'])
            if checkpoint!=result['checkpoint_sha256']:raise ValueError(f'Audit checkpoint mismatch: {name}')
            contract=read_json(directory/'lock.json')
            if contract['model']!=model_hash:raise ValueError(f'Audit model code mismatch: {name}')
            if 'factor' in contract and contract['factor']!=factor:raise ValueError(f'Audit GP factor mismatch: {name}')
            record.update(checkpoint=checkpoint,contract=sha256(directory/'lock.json'))
        records[name]=record
    return dict(model=model_hash,factor=factor,data=sha256(root/'data.lock.json'),artifacts=records)


def main():
    torch.set_num_threads(2);data=Data();records={}
    arms=('N-point','N-set','N-GP-point','N-GP-set','Gaussian-calibrated')
    dependencies=audit_dependencies(ROOT,[f'fold_{fold}/{arm}' for fold in (0,1) for arm in arms])
    L=torch.tensor(np.load(ROOT/'GP_correlation_factor.npy'),device=data.device)
    gaussian_gain=float(heun_sample(lambda x,s:x/(1+s*s),torch.ones(1,1,dtype=torch.float64),steps=18)[0,0])
    immutable_json(ROOT/'sampling_stability_v02.lock.json',dict(data=sha256(ROOT/'data.lock.json'),dependencies=dependencies,
        K=[32,128,512],seed=20261127,selection='already locked; no new selection',chunk_set=64,chunk_point=128,
        reason='NDP K32 mean MC standard error exceeds1mm; distinguish MC mean noise from point model error',
        control_variate='fixed analytic Gaussian-sampler gain times initial noise; zero expected value; no model change',
        runner=sha256(__file__)))
    for fold in (0,1):
        hours=monthly_subset(data.lock['outer_hours']);ix=data.indices(hours)
        population=data.a['valid'][ix]&np.array(data.lock['splits'][fold]['outer'])[None]
        for arm in arms:
            directory=ROOT/f'fold_{fold}'/arm;result=read_json(directory/'result.json')
            model=None
            if arm.startswith('N-'):
                model=SetDenoiser(attention=arm.endswith('set')).to(data.device)
                model.load_state_dict(torch.load(directory/result['checkpoint'],map_location=data.device,weights_only=True))
            rows=[];controls=[]
            with torch.no_grad():
                for h in hours:
                    hour=np.array([h]);ii,x,base,amp=data.batch(hour)
                    if model is not None:
                        c=data.conditions(x);noise=keyed_noise(hour,512,20261127,shape=(511,),device=data.device).flatten(0,1);samples=[]
                        if '-GP-' in arm:noise=noise@L.T
                        controls.append((10*amp[:,None]*gaussian_gain*noise).cpu().numpy())
                        chunk=64 if arm.endswith('set') else 128
                        for start in range(0,512,chunk):
                            k=min(chunk,512-start);mask=torch.ones(k,511,dtype=torch.bool,device=data.device)
                            z=heun_sample(lambda state,sigma:model.denoise(state,sigma,c.expand(k,-1,-1),data.coords,mask),noise[start:start+k],steps=18)
                            samples.append((base+10*amp[:,None]*z).cpu().numpy())
                        rows.append(np.concatenate(samples))
                    else:
                        latent=keyed_noise(hour,512,20261127,shape=(data.rank,),device=data.device)
                        remainder=keyed_noise(hour,512,20262127,shape=(data.rest,),device=data.device)
                        eta=keyed_noise(hour,512,20263127,shape=(905,),device=data.device)[...,data.a['station_ids']]
                        factor=latent@torch.tensor(data.a['projection'].T,device=data.device)+remainder@torch.tensor(data.a['remainder'].T,device=data.device)+data.tau*eta
                        rows.append((base[:,None]+amp[:,None,None]*data.scales[fold]*factor)[0].cpu().numpy())
            samples=np.array(rows);original=dict(np.load(directory/'outer_predictions.npz'))
            control=np.array(controls) if model is not None else None
            prefix=float(np.max(abs(samples[:,:32]-original['samples'])))
            if prefix>.001:raise ValueError(f'Sampling prefix mismatch {arm}: {prefix}')
            array_path=directory/'stability_K512.npz'
            np.savez_compressed(array_path,samples=samples,hours=hours,truth=data.a['truth'][ix],population=population)
            arm_results={}
            point_arrays={}
            for k in (32,128,512):
                z=samples[:,:k];mean=z.mean(1) if model is not None else data.a['mean'][ix]
                metrics=score_ensemble(data.a['truth'][ix],z,mean,population,data.a['xy'])
                metrics['mean_mc_se_mm']=float(np.mean((z.std(1,ddof=1)/np.sqrt(k))[population])) if model is not None else 0.
                if control is not None:
                    cv=z-control[:,:k];metrics['point_control_variate']=point_metrics(data.a['truth'][ix],cv.mean(1),population)
                    metrics['point_control_variate']['mean_mc_se_mm']=float(np.mean((cv.std(1,ddof=1)/np.sqrt(k))[population]))
                    point_arrays[f'control_variate_mean_K{k}']=cv.mean(1)
                arm_results[str(k)]=metrics
            records[f'{fold}/{arm}']=dict(prefix_max_mm=prefix,metrics=arm_results,gaussian_sampler_gain=gaussian_gain)
            if point_arrays:np.savez_compressed(directory/'control_variate_point_predictions.npz',hours=hours,station_ids=data.a['station_ids'],population=population,**point_arrays)
            write_json(ROOT/'sampling_stability.json',records);print('STABILITY',fold,arm,{k:r['station_macro_rmse_mm'] for k,r in arm_results.items()},flush=True)
            del model;torch.cuda.empty_cache()
    if audit_dependencies(ROOT,[f'fold_{fold}/{arm}' for fold in (0,1) for arm in arms])!=dependencies:raise ValueError('Audit dependencies changed during sampling')
    write_json(ROOT/'sampling_dependency_validation.json',dict(passed=True,lock=sha256(ROOT/'sampling_stability_v02.lock.json'),outputs=sha256(ROOT/'sampling_stability.json')))


if __name__=='__main__':main()
