"""Fixed-history full-field radial/tail diagnosis; no2020 labels, no neural fitting."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import scipy
from scipy.stats import chi2
import torch
from campaign import RESULTS,read,STOP
from inducing_flow_series import source_paths
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_inducing_flow_experiment import codes,batch,affine_std
from tools.terrain_diffusion.native_inducing_flow import InducingLatentFlow
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_student_process import affine_covariance,masked_radius,log_density


GRID=[3,4,6,8,12,20,40,80,160,1000]
BLOCKS=np.linspace(0,2908,48,dtype=int).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();seed=args.seed
    if STOP.exists():raise SystemExit(75)
    torch.set_num_threads(2);paths=source_paths(seed)
    source=WeatherSources(SimpleNamespace(seed=seed,device=args.device,smoke_cache=False,**paths))
    if source.data.development:raise ValueError('training histories only')
    model_path=RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}'
    seal=validate_model_seal(model_path,'COMPLETE_INDUCING_FLOW',codes());protocol=read(model_path/'protocol.json')
    if protocol['source']!=source.receipt or protocol['config']['family']!='affine' or protocol['config']['domain']!='time':
        raise ValueError('registered affine-time source required')
    model=InducingLatentFlow('affine','time').to(args.device).eval()
    model.load_state_dict(torch.load(model_path/'last.pt',map_location=args.device,weights_only=True));model.requires_grad_(False)
    anchors=torch.tensor(protocol['anchors'],device=args.device,dtype=torch.float64)
    root=RESULTS/f'20261126_full_field_tail_probe/s{seed}';root.mkdir(parents=True,exist_ok=False)
    helper=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/tools/terrain_diffusion/native_student_process.py'
    write_json(root/'protocol.json',dict(seed=seed,device=args.device,source=source.receipt,
        model_seal_sha256=sha(model_path/'model_artifact_seal.json'),weights_sha256=seal['weights_sha256'],
        model_protocol_sha256=seal['protocol_sha256'],source_code_sha256=codes(),helper_sha256=sha(helper),driver_sha256=sha(__file__),
        cache_manifest_sha256=sha(paths['cache']/'manifest.json'),embedding_manifest_sha256=sha(paths['embeddings']/'manifest.json'),
        blocks=BLOCKS,nu_grid=GRID,source_mean='frozen TF3000',source_covariance='complete frozen affine_time process',
        fitting='analytic constant-nu and scalar-Gaussian probes on2016-2018; probe transfer2019; foundation already fitted2019',
        score='joint log density divided by observed dimension; block-resampled intervals',
        exclusion='last91 reused blocks excluded by fixed indices, no label-dependent block selection',
        torch_version=torch.__version__,scipy_version=scipy.__version__,strict_oof=False,confirmation_C_D_read=False))
    rows=[];start=time.monotonic();diagonal_error=0.
    with torch.no_grad():
        for j,i in enumerate(BLOCKS):
            if STOP.exists():raise SystemExit(75)
            b,truth,_,residual,_,_,_=batch(source,i,anchors,2,seed,True)
            C=affine_covariance(model,b,residual)
            expected=affine_std(model,b,residual).flatten(1).square()
            error=float((C.diagonal(dim1=-2,dim2=-1)-expected).abs().max());diagonal_error=max(diagonal_error,error)
            torch.testing.assert_close(C.diagonal(dim1=-2,dim2=-1),expected,rtol=1e-10,atol=1e-8)
            e=(truth.double()-b['mean'].double()).flatten(1);q,ld,d=masked_radius(e,C)
            origins=np.asarray(source.data.a['origins'][i])
            for n,hour in enumerate(origins):
                year=int(np.datetime64(int(hour),'h').astype('datetime64[Y]').astype(int)+1970)
                if year not in (2016,2017,2018,2019):raise ValueError('probe accessed a nontraining year')
                row=dict(block_index0=i,origin_index=n,origin_nominal_hour=int(hour),year=year,dimension=int(d[n]))
                if d[n]:
                    row.update(radius=float(q[n]),logdet=float(ld[n]),radius_per_dimension=float(q[n]/d[n]),
                        gaussian_radial_pit=float(chi2.cdf(float(q[n]),int(d[n]))))
                else:row.update(radius=None,logdet=None,radius_per_dimension=None,gaussian_radial_pit=None)
                rows.append(row)
            if j==0 or (j+1)%12==0:print(dict(seed=seed,completed_blocks=j+1,total_blocks=len(BLOCKS),seconds=time.monotonic()-start),flush=True)
    valid=[r for r in rows if r['dimension']>0]
    q=torch.tensor([r['radius'] for r in valid],dtype=torch.float64)
    ld=torch.tensor([r['logdet'] for r in valid],dtype=torch.float64);d=torch.tensor([r['dimension'] for r in valid])
    fit=np.array([r['year']<2019 for r in valid]);transfer=~fit
    if not fit.any() or not transfer.any():raise ValueError('both probe roles require observed labels')
    scale2=float(np.mean([r['radius_per_dimension'] for r in valid if r['year']<2019]))
    scores={'gaussian':(-log_density(q,ld,d)/d).numpy(),
        'scaled_gaussian':(-log_density(q/scale2,ld+d*math_log(scale2),d)/d).numpy()}
    scores.update({f't{nu}':(-log_density(q,ld,d,nu)/d).numpy() for nu in GRID})
    names=['gaussian',*[f't{nu}' for nu in GRID]];chosen=min(names,key=lambda n:float(scores[n][fit].mean()))
    for i,r in enumerate(valid):r['nll_per_dimension']={n:float(v[i]) for n,v in scores.items()}
    score_table={n:dict(fit2016_2018=float(v[fit].mean()),transfer2019=float(v[transfer].mean())) for n,v in scores.items()}
    summary=dict(complete=True,seed=seed,selected_tail_family=chosen,gaussian_scale2_fit=scale2,scores=score_table,
        fit_origins=int(fit.sum()),transfer_origins=int(transfer.sum()),valid_origins=len(valid),empty_origins=len(rows)-len(valid),
        fit_blocks=sorted({r['block_index0'] for r in valid if r['year']<2019}),transfer_blocks=sorted({r['block_index0'] for r in valid if r['year']==2019}),
        radial={role:radial_summary([r for r in valid if (r['year']<2019)==isfit]) for role,isfit in [('fit2016_2018',True),('transfer2019',False)]},
        covariance_diagonal_max_error_mm2=diagonal_error,body_seconds=time.monotonic()-start,protocol_sha256=sha(root/'protocol.json'),
        interpretation='analytic diagnostic only;2019 source models already trained; no forecast CRPS gain established',strict_oof=False,confirmation_C_D_read=False)
    write_json(root/'records.json',dict(rows=rows,source_protocol_sha256=sha(root/'protocol.json')))
    write_json(root/'summary.json',summary);print(summary,flush=True)


def math_log(x):return float(np.log(x))


def radial_summary(rows):
    r=np.array([v['radius_per_dimension'] for v in rows]);pit=np.array([v['gaussian_radial_pit'] for v in rows])
    return dict(radius_per_dimension_mean=float(r.mean()),radius_per_dimension_quantiles=np.quantile(r,[.5,.9,.95,.99]).tolist(),
        gaussian_radial_pit_above95=float((pit>.95).mean()),gaussian_radial_pit_below05=float((pit<.05).mean()))


if __name__=='__main__':main()
