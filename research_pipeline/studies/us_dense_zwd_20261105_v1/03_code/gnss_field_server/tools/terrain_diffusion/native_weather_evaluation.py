"""Float64 native R/F metrics and IID-member probability evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import erf
import torch
from .data import sha,write_json
from .native_weather_experiment import CachedData
from .native_weather_models import NativeDiffusion,NativeGaussian
from .native_copula import correlate_noise
from .native_frequency_tokens import probability_model
from .native_paths import file_path


def safe_mean(x,axis):
    n=np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x,axis=axis),n,out=np.full(n.shape,np.nan,dtype='float64'),where=n>0)


def fair_crps(samples,truth):
    x=np.asarray(samples,dtype='float64');K=len(x)
    if K<2:raise ValueError('fair CRPS requires at least two IID members')
    order=np.sort(x,axis=0);weight=(2*np.arange(K)-K+1).reshape((K,)+(1,)*(x.ndim-1))
    result=np.mean(np.abs(x-truth[None]),axis=0)-(order*weight).sum(0)/(K*(K-1))
    return np.where(np.isfinite(truth),result,np.nan)


def gaussian_crps(mean,std,truth):
    std=np.maximum(std,1e-12);z=(truth-mean)/std
    density=np.exp(-.5*z*z)/np.sqrt(2*np.pi);cdf=.5*(1+erf(z/np.sqrt(2)))
    return std*(z*(2*cdf-1)+2*density-1/np.sqrt(np.pi))


def point_metrics(prediction,truth):
    # [layout,month,continuous-origin-hour,station,horizon]; never join calendar gaps.
    error=prediction.astype('float64')-truth.astype('float64')
    macro=np.nanmean(np.sqrt(safe_mean(error*error,(1,2))),axis=(0,1))
    inc=np.diff(error[:,:,:,:,0],axis=2)
    return dict(horizon_macro_rmse_mm=macro.tolist(),
        horizon_micro_rmse_mm=np.sqrt(safe_mean(error*error,(0,1,2,3))).tolist(),
        horizon_mae_mm=safe_mean(np.abs(error),(0,1,2,3)).tolist(),
        horizon_bias_mm=safe_mean(error,(0,1,2,3)).tolist(),
        current_hourly_increment_macro_rmse_mm=float(np.nanmean(np.sqrt(safe_mean(inc*inc,(1,2))))),
        within_forecast_increment_micro_rmse_mm=float(np.sqrt(np.nanmean(np.diff(error,axis=-1)**2))),
        valid_count_by_horizon=np.isfinite(error).sum((0,1,2,3)).tolist())


def calendar_shape(x,months):
    return x.reshape(3,months,6,12,*x.shape[2:]).reshape(3,months,72,*x.shape[2:])


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--embeddings',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probability',type=Path);p.add_argument('--members',type=int,default=32)
    p.add_argument('--sample-steps',type=int,default=20);p.add_argument('--seed',type=int,default=261126)
    p.add_argument('--copula',type=Path)
    p.add_argument('--solver',choices=['ddim','heun'],default='ddim')
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    torch.set_num_threads(3);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.35)
    data=CachedData(args.cache)
    if not data.development:raise ValueError('calendar evaluation expects a registered development calendar')
    manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
    if manifest['cache_manifest_sha256']!=sha(args.cache/'manifest.json'):raise ValueError('embedding calendar differs')
    for name,expected in manifest['files'].items():
        if sha(args.embeddings/name)!=expected:raise ValueError('evaluation embedding differs')
    folder=args.out;folder.mkdir(parents=True,exist_ok=False)
    mean=np.load(args.embeddings/'mean.npy',mmap_mode='r');condition=np.load(args.embeddings/'embedding.npy',mmap_mode='r')
    months=data.protocol['config']['months'];truth=calendar_shape(np.asarray(data.a['truth']),months)
    deterministic=calendar_shape(np.asarray(mean),months)
    reference=calendar_shape(np.repeat(data.a['mean'][...,None],7,axis=-1),months)
    payload=dict(reference=point_metrics(reference,truth),deterministic_mean=point_metrics(deterministic,truth))
    artifacts=dict(prediction=deterministic,reference=reference,truth=truth)
    if args.probability:
        protocol=json.loads((args.probability/'protocol.json').read_text(encoding='utf-8'))
        if protocol['code_sha256']!=sha(Path(__file__).with_name('native_weather_models.py')):raise ValueError('frozen probability implementation changed')
        for name,expected in protocol.get('extra_model_sources',{}).items():
            if sha(Path(__file__).with_name(name))!=expected:raise ValueError('frozen spectral implementation changed')
        training_manifest=json.loads((Path(protocol['config']['embeddings'])/'manifest.json').read_text(encoding='utf-8'))
        if training_manifest['mean_weights_sha256']!=manifest['mean_weights_sha256']:raise ValueError('evaluation and probability mean weights differ')
        domain=protocol['config']['domain'];stats=protocol['normalization']
        bias=torch.tensor(stats['bias_mm'],device='cuda');scale=torch.tensor(stats['scale_mm'],device='cuda')
        model=probability_model(domain).cuda().eval()
        model.load_state_dict(torch.load(args.probability/'last.pt',weights_only=True))
        preconditioner=None
        if protocol.get('preconditioner_sha256'):
            path=Path(protocol['config']['preconditioner'])/'last.pt'
            if sha(path)!=protocol['preconditioner_sha256']:raise ValueError('Gaussian preconditioner changed')
            preconditioner=NativeGaussian().cuda().eval();preconditioner.load_state_dict(torch.load(path,weights_only=True))
        copula=json.loads(args.copula.read_text(encoding='utf-8')) if args.copula else None
        if copula and (domain!='gaussian' or copula['gaussian_weights_sha256']!=sha(args.probability/'last.pt')):raise ValueError('copula source differs')
        allsamples=[];analytic=[];gaussian_means=[];torch.manual_seed(args.seed)
        for i in range(len(data)):
            c=data.tensor(condition[i]);mu=data.tensor(mean[i]);eigen,U=data.graph(i)
            B,Q,_=c.shape;K=args.members
            noise=torch.randn(K*B,Q,7,device='cuda')
            if copula:
                st=data.query_static[data.tensor(data.a['query_ids'][i],torch.long)]
                noise=correlate_noise(noise,st,copula['spatial_rho'],torch.tensor(copula['time_correlation'],device='cuda',dtype=torch.float32))
            samples=model.sample(c.repeat(K,1,1),noise,U,eigen,args.sample_steps,args.solver)
            if preconditioner is not None:
                location,logscale=preconditioner(c)
                samples=samples*logscale.exp().repeat(K,1,1)+location.repeat(K,1,1)
            samples=(samples*scale+bias+mu.repeat(K,1,1)).reshape(K,B,Q,7)
            if not torch.isfinite(samples).all():raise FloatingPointError('nonfinite generated field')
            allsamples.append(samples.cpu().numpy())
            if domain=='gaussian':
                delta,logstd=model(c)
                gaussian_means.append((mu+bias+delta*scale).cpu().numpy())
                analytic.append(gaussian_crps((mu+bias+delta*scale).cpu().numpy(),(logstd.exp()*scale).cpu().numpy(),data.a['truth'][i]))
            if i==0 or (i+1)%50==0:print(json.dumps(dict(stage='sampling',domain=domain,completed=i+1,total=len(data))),flush=True)
        samples=np.stack(allsamples,axis=1)
        samples=np.stack([calendar_shape(x,months) for x in samples])
        crps=fair_crps(samples,truth)
        payload['sample_mean']=point_metrics(samples.mean(0),truth)
        payload['fair_crps_mm_by_horizon']=safe_mean(crps,(0,1,2,3)).tolist()
        quantile=np.quantile(samples,[.1,.9],axis=0)
        covered=(truth>=quantile[0])&(truth<=quantile[1]);valid=np.isfinite(truth)
        payload['coverage80_by_horizon']=(np.sum(covered&valid,axis=(0,1,2,3))/valid.sum((0,1,2,3))).tolist()
        payload['width80_mm_by_horizon']=safe_mean(np.where(valid,quantile[1]-quantile[0],np.nan),(0,1,2,3)).tolist()
        payload['forecast_increment_fair_crps_mm']=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1))))
        payload['negative_sample_fraction']=float((samples<0).mean())
        if analytic:
            payload['analytic_gaussian_crps_mm_by_horizon']=safe_mean(calendar_shape(np.stack(analytic),months),(0,1,2,3)).tolist()
            artifacts['gaussian_distribution_mean']=calendar_shape(np.stack(gaussian_means),months)
            payload['gaussian_distribution_mean']=point_metrics(artifacts['gaussian_distribution_mean'],truth)
        artifacts.update(samples=samples,crps=crps.astype('float32'))
    np.savez_compressed(folder/'evaluation.npz',**artifacts)
    write_json(folder/'metrics.json',payload)
    write_json(folder/'lineage.json',dict(cache_manifest_sha256=sha(args.cache/'manifest.json'),
        embedding_manifest_sha256=sha(args.embeddings/'manifest.json'),
        probability_weights_sha256=sha(args.probability/'last.pt') if args.probability else None,
        members=args.members if args.probability else None,sampling_steps=args.sample_steps if args.probability else None,
        independent_iid_member_noise=True,seed=args.seed,code_sha256=sha(Path(__file__)),
        copula_sha256=sha(args.copula) if args.copula else None,
        solver=args.solver,neural_evaluations_per_sample=(1 if domain=='gaussian' else args.sample_steps*(2 if args.solver=='heun' else 1)) if args.probability else None,
        prediction_sha256=sha(folder/'evaluation.npz'),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(payload))


if __name__=='__main__':main()
