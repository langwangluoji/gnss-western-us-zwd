"""Separate finite-ensemble interval coverage from analytic Gaussian calibration."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def rank_interval(samples):
    K=len(samples);tail=max(1,int(np.floor((K+1)*.1)))
    lower=tail;upper=K+1-tail
    ordered=np.partition(samples,[lower-1,upper-1],axis=0)
    return ordered[lower-1],ordered[upper-1],(upper-lower)/(K+1),[lower,upper]


def coverage(lower,upper,truth):
    valid=np.isfinite(truth);axes=tuple(range(truth.ndim-1))
    count=valid.sum(axes)
    hit=(((truth>=lower)&(truth<=upper))&valid).sum(axes)
    return dict(coverage_by_horizon=(hit/count).tolist(),
        mean_width_mm_by_horizon=(np.where(valid,upper-lower,0).sum(axes)/count).tolist(),valid_count_by_horizon=count.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--gaussian',type=Path);p.add_argument('--embeddings',type=Path)
    args=p.parse_args();sys.path.insert(0,str(ENGINE))
    from tools.terrain_diffusion.native_paths import file_path
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_weather_evaluation import calendar_shape,gaussian_crps
    lineage=json.loads((args.evaluation/'lineage.json').read_text(encoding='utf-8'))
    if sha(args.evaluation/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('changed evaluation')
    if sha(args.cache/'manifest.json')!=lineage['cache_manifest_sha256']:raise ValueError('different calendar')
    with np.load(args.evaluation/'evaluation.npz') as data:
        samples=data['samples'];truth=data['truth'];analytic_mean=data['gaussian_distribution_mean'] if args.gaussian else None
    lower,upper,reference,ranks=rank_interval(samples)
    result=dict(members=len(samples),rank_interval=coverage(lower,upper,truth),rank_order_one_based=ranks,
        exact_calibrated_continuous_iid_rank_reference=reference,
        interpretation='Rank interval is assessed against its exact finite-K reference; default value-interpolated 10/90% sample quantiles are not exact population 80% bounds.',
        source_evaluation_sha256=lineage['prediction_sha256'],strict_oof=False,confirmation_C_D_read=False)
    if args.gaussian:
        import torch
        from tools.terrain_diffusion.native_weather_models import NativeGaussian
        torch.set_num_threads(3);torch.set_grad_enabled(False)
        protocol=json.loads((args.gaussian/'protocol.json').read_text(encoding='utf-8'))
        if protocol['config']['domain']!='gaussian' or sha(args.gaussian/'last.pt')!=lineage['probability_weights_sha256']:raise ValueError('Gaussian source differs')
        if protocol['code_sha256']!=sha(ENGINE/'tools/terrain_diffusion/native_weather_models.py'):raise ValueError('Gaussian code differs')
        manifest=json.loads((args.embeddings/'manifest.json').read_text(encoding='utf-8'))
        if sha(args.embeddings/'manifest.json')!=lineage['embedding_manifest_sha256']:raise ValueError('Gaussian embeddings differ')
        embedding=np.load(args.embeddings/'embedding.npy',mmap_mode='r');sigmas=[]
        model=NativeGaussian().eval();model.load_state_dict(torch.load(args.gaussian/'last.pt',map_location='cpu',weights_only=True))
        scale=np.array(protocol['normalization']['scale_mm'])
        for c in embedding:
            _,logstd=model(torch.tensor(np.array(c),dtype=torch.float32));sigmas.append(logstd.exp().numpy()*scale)
        sigma=calendar_shape(np.stack(sigmas),truth.shape[1]);quantile=1.2815515655446004
        result['analytic_gaussian_80']=coverage(analytic_mean-quantile*sigma,analytic_mean+quantile*sigma,truth)
        expected=json.loads((args.evaluation/'metrics.json').read_text(encoding='utf-8'))['analytic_gaussian_crps_mm_by_horizon']
        actual=np.nanmean(gaussian_crps(analytic_mean,sigma,truth),axis=(0,1,2,3))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6)
        nearest=np.load(args.cache/'nearest.npy').reshape(3,12,6,12,137).reshape(3,12,72,137)
        strata=[]
        for lo,hi in [(0,50),(50,100),(100,150),(150,300),(300,np.inf)]:
            record=dict(distance_lower_km=lo,distance_upper_km=hi if np.isfinite(hi) else None)
            for h in (0,6):
                valid=(nearest>=lo)&(nearest<hi)&np.isfinite(truth[...,h]);error=truth[...,h]-analytic_mean[...,h]
                std=sigma[...,h];record[f'h{h}']=dict(count=int(valid.sum()),
                    standardized_error_rms=float(np.sqrt(np.mean((error[valid]/std[valid])**2))),
                    error_rmse_mm=float(np.sqrt(np.mean(error[valid]**2))),
                    forecast_std_rms_mm=float(np.sqrt(np.mean(std[valid]**2))),
                    analytic_coverage80=float(np.mean(np.abs(error[valid])<=quantile*std[valid])))
            strata.append(record)
        result['distance_calibration']=strata
        result['training_global_normalization']=protocol['normalization']
        result['gaussian_weights_sha256']=sha(args.gaussian/'last.pt')
    result['code_sha256']=sha(Path(__file__));args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'calibration_audit.json',result)
    print(json.dumps({k:result[k] for k in ('rank_order_one_based','exact_calibrated_continuous_iid_rank_reference','rank_interval','analytic_gaussian_80') if k in result}))


if __name__=='__main__':main()
