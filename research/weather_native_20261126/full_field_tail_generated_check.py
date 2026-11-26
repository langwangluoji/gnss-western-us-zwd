"""Source-generated Gaussian radial self-check, separate from observed GNSS evidence."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import chi2
import torch
from campaign import RESULTS,read,STOP
from inducing_flow_series import source_paths
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_inducing_flow_experiment import codes,batch,coupled_draw
from tools.terrain_diffusion.native_inducing_flow import InducingLatentFlow
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_student_process import affine_covariance


@torch.no_grad()
def main():
    if STOP.exists():raise SystemExit(75)
    torch.set_num_threads(2);rows=[]
    for seed in (42,17):
        source=WeatherSources(SimpleNamespace(seed=seed,device='cuda',smoke_cache=False,**source_paths(seed)))
        model_path=RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}'
        seal=validate_model_seal(model_path,'COMPLETE_INDUCING_FLOW',codes());p=read(model_path/'protocol.json')
        if p['source']!=source.receipt:raise ValueError('source changed')
        model=InducingLatentFlow('affine','time').cuda().eval();model.load_state_dict(torch.load(model_path/'last.pt',map_location='cuda',weights_only=True))
        anchors=torch.tensor(p['anchors'],device='cuda',dtype=torch.float64)
        for block in (0,2000):
            b,truth,_,residual,_,_,_=batch(source,block,anchors,2,seed,True)
            single={k:(v[:,:1] if k in ('gp_path','latent_noise') else v[:1]) for k,v in b.items()}
            C=affine_covariance(model,single,residual[:1])[0]
            torch.manual_seed(seed+261127+block)
            path,eta=coupled_draw(residual[:1],single['basis'].double(),4096)
            single.update(gp_path=path,latent_noise=eta)
            samples=model(**single)[:,0];error=(samples.double()-single['mean'][0].double()).flatten(1)
            index=torch.nonzero(torch.isfinite(truth[0]).flatten()).flatten();L=torch.linalg.cholesky(C[index][:,index])
            radius=torch.linalg.solve_triangular(L,error[:,index].T,upper=False).square().sum(0).cpu().numpy()
            d=len(index);pit=chi2.cdf(radius,d)
            row=dict(seed=seed,training_block_index0=block,members=4096,observed_subvector_dimension=d,
                generated_radius_per_dimension_mean=float(radius.mean()/d),generated_pit_above95=float((pit>.95).mean()),
                generated_pit_below05=float((pit<.05).mean()),source_weights_sha256=seal['weights_sha256'])
            if abs(row['generated_radius_per_dimension_mean']-1)>.015 or abs(row['generated_pit_above95']-.05)>.02:
                raise ValueError('analytic covariance does not reproduce source-generated Gaussian radius')
            rows.append(row);print(row,flush=True)
    root=RESULTS/'20261126_full_field_tail_probe';write_json(root/'generated_check.json',dict(complete=True,rows=rows,
        code_sha256=sha(__file__),role='generated Gaussian diagnostic, not independent atmospheric forecast evidence',
        rounding='kernel float64 and decoder float32; covariance checked at source numeric precision',confirmation_C_D_read=False))


if __name__=='__main__':main()
