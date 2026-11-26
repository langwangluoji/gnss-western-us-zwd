"""Real137-query principal matrices and common-path local transport probes."""
import argparse
import json
import sys
from pathlib import Path
import torch
from campaign import RESULTS,ROOT,read

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources,codes
from tools.terrain_diffusion.native_weather_gp_diffusion import make_model,trajectory_factors,transform_gp_path
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import validate_model_seal


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--view',type=int,choices=[175,265],default=265);p.add_argument('--smoke',action='store_true')
    p.add_argument('--transport',type=Path);args=p.parse_args();seed=args.seed
    args.cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}'
    args.embeddings=RESULTS/'20261126_native_weather_embeddings'/f'transformer72_physics_aux_last_dev_s{seed}'
    args.gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}';args.copula=args.gaussian/'copula.json'
    args.gp=(RESULTS/'20261126_weather_gp_gap_metric'/f'c{args.view}_s{seed}' if args.smoke else
        RESULTS/'20261126_weather_projective_gp'/f's{seed}'/f'c{args.view}')
    args.smoke_cache=args.smoke;args.device='cuda';torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.4)
    source=WeatherSources(args);model=make_model('local').cuda().eval()
    if args.transport:
        args.transport=file_path(args.transport);protocol=read(args.transport/'protocol.json');status=read(args.transport/'status.json')
        expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_WEATHER_TRANSPORT'
        if status['status']!=expected or protocol['source']!=source.receipt or protocol['code_sha256']!=codes():raise ValueError('transport source changed')
        if args.smoke:
            if sha(args.transport/'last.pt')!=status['last_weights_sha256']:raise ValueError('engineering weights changed')
        else:validate_model_seal(args.transport,'COMPLETE_WEATHER_TRANSPORT',codes())
        model=make_model(protocol['config']['mode']).cuda().eval()
        model.load_state_dict(torch.load(args.transport/'last.pt',map_location='cuda',weights_only=True))
    rows=[];torch.manual_seed(261126)
    for block in (0,6,12):
        c,gm,gs,base,ids,st=source.batch(block);inputs,_,_,_=source.data.batch(block,72)
        # One actual origin per block keeps the probe independent of GPU budget.
        history=inputs['history'][:1];c=c[:1]
        condition=source.weather_batch(block)[0][:1]
        R=source.weather_gp.correlation(c,history,st,ids)
        root,_=trajectory_factors(source.weather_gp.query_time_blocks(c,history,st,ids))
        path=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(R),torch.randn(32,1,959,device='cuda').double()).reshape(32,137,7)
        full=transform_gp_path(model,condition.repeat(32,1,1),path,root.repeat(32,1,1,1))
        for n in (1,64,128,137):
            sub=torch.arange(n,device='cuda');flat=(sub[:,None]*7+torch.arange(7,device='cuda')).flatten()
            smallR=source.weather_gp.correlation(c[:,sub],history,st[sub],ids[sub])
            maximum=float((smallR-R[:,flat][:,:,flat]).abs().max())
            smallroot,_=trajectory_factors(source.weather_gp.query_time_blocks(c[:,sub],history,st[sub],ids[sub]))
            small=transform_gp_path(model,condition[:,sub].repeat(32,1,1),path[:,sub],smallroot.repeat(32,1,1,1))
            delta=(small-full[:,sub])*source.scale*gs[:1,sub]
            row=dict(block=block,queries=n,kernel_max_difference=maximum,path_max_difference_mm=float(delta.abs().max()),path_rms_difference_mm=float(delta.square().mean().sqrt()))
            if maximum>1e-5 or row['path_max_difference_mm']>2e-4:raise ValueError('real-source projectivity probe failed')
            rows.append(row)
    out=(args.transport if args.transport else args.gp)/'query_audit.json'
    write_json(out,dict(rows=rows,engineering_only=args.smoke,source=source.receipt,code_sha256=sha(__file__),
        transport_weights_sha256=sha(args.transport/'last.pt') if args.transport else None,
        meaning='Fixed visible histories and common superset GP initial path; no labels used; different Cholesky roots with the same seed are not a path-coupling test',strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps(dict(out=str(out),rows=rows)))


if __name__=='__main__':main()
