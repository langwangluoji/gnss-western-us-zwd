"""Real137-query mean invariance and common-GP-path shape subset probes."""
import argparse
import json
import sys
from pathlib import Path
import torch
from campaign import ROOT,RESULTS,read

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_weather_gp_diffusion import condition265
from tools.terrain_diffusion.native_centered_shape import CenteredShape
from tools.terrain_diffusion.native_centered_shape_experiment import codes,load_centered_model
from tools.terrain_diffusion.data import sha,write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();seed=args.seed;args.device='cuda';args.smoke_cache=args.smoke
    args.cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}'
    args.embeddings=RESULTS/'20261126_native_weather_embeddings'/f'transformer72_physics_aux_last_dev_s{seed}'
    args.gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}';args.copula=args.gaussian/'copula.json'
    args.gp=RESULTS/'20261126_weather_gp_gap_metric'/f'c265_s{seed}' if args.smoke else RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265'
    root=RESULTS/'20261126_centered_shape_gap' if args.smoke else RESULTS/'20261126_centered_shape'/f's{seed}'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.35);source=WeatherSources(args)
    models={}
    for mode in ('scale','shape','both'):
        folder=root/mode
        if args.smoke:
            protocol=read(folder/'protocol.json');status=read(folder/'status.json')
            if status['status']!='ENGINEERING_SMOKE_COMPLETE' or protocol['source']!=source.receipt or protocol['code_sha256']!=codes():raise ValueError('shape engineering source changed')
            if sha(folder/'last.pt')!=status['last_weights_sha256']:raise ValueError('engineering weights changed')
            model=CenteredShape(mode).cuda().eval();model.load_state_dict(torch.load(folder/'last.pt',map_location='cuda',weights_only=True))
        else:model,_=load_centered_model(folder,source.receipt,'cuda')
        models[mode]=model
    rows={mode:[] for mode in models};torch.manual_seed(261126)
    for block in (0,6,12):
        c,gm,gs,base,ids,st=source.batch(block);inputs,_,_,_=source.data.batch(block,72)
        history=inputs['history'][:1];c=c[:1];condition=condition265(c,history);sigma=(source.scale*gs)[:1];base=base[:1]
        R=source.weather_gp.correlation(c,history,st,ids)
        eta=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(R),torch.randn(32,1,959,device='cuda').double()).reshape(32,1,137,7).float()
        for mode,model in models.items():
            mean,std=model.moments(condition);a,b,d,_=model.coefficients(condition)
            center=base+sigma*mean;difference=float((center-base).abs().max())
            norm=float((a.square()+b.square()+d.square()-1).abs().max())
            if difference!=0 or norm>1e-6:raise ValueError('real-source exact-moment contract failed')
            full=base[None]+sigma[None]*model.transform(condition,eta)
            for n in (1,64,128,137):
                small=base[:,:n][None]+sigma[:,:n][None]*model.transform(condition[:,:n],eta[:,:,:n])
                delta=small-full[:,:,:n];maximum=float(delta.abs().max())
                if maximum>2e-4:raise ValueError('shape common-path query subset contract failed')
                rows[mode].append(dict(block=block,queries=n,analytic_mean_max_difference_mm=difference,coefficient_unit_variance_error=norm,
                    sample_path_max_difference_mm=maximum,path_rms_difference_mm=float(delta.square().mean().sqrt())))
    for mode in models:
        folder=root/mode
        write_json(folder/'query_audit.json',dict(rows=rows[mode],engineering_only=args.smoke,source=source.receipt,
            weights_sha256=sha(folder/'last.pt'),code_sha256=sha(__file__),strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(rows))


if __name__=='__main__':main()
