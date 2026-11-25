"""Frozen G265 plus matched local nonlinear and affine trajectory transport."""
import argparse
import subprocess
import sys
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read,run
from tools.terrain_diffusion.native_model_seal import seal_completed_model,validate_model_seal,evaluation_marker
from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();seed=args.seed
    gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    name='transformer72_physics_aux_last';emb=RESULTS/'20261126_native_weather_embeddings'
    cache=RESULTS/'20261126_native_weather_cache_v2'
    if args.smoke:
        if seed!=42:raise ValueError('engineering smoke has registered seed42 source')
        gp=RESULTS/'20261126_weather_gp_gap_metric/c265_s42';source=read(gp/'protocol.json')['config']
        traincache=source['cache'];trainemb=source['embeddings']
        root=RESULTS/'20261126_weather_transport_gap_sealed'
    else:
        gp=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265'
        traincache=cache/f'train_s{seed}';trainemb=emb/f'{name}_train_s{seed}'
        root=RESULTS/'20261126_weather_transport'/f's{seed}'
        seal_completed_model(gp,'COMPLETE_WEATHER_GP',gp_codes())
    common=['--gaussian',gaussian,'--copula',gaussian/'copula.json','--gp',gp,'--seed',seed]
    for mode in ('local','affine'):
        folder=root/mode;status=folder/'status.json'
        run('native_weather_gp_diffusion_experiment',['--phase','train','--mode',mode,
            '--cache',traincache,'--embeddings',trainemb,'--out',folder,'--steps',4 if args.smoke else 3000,
            *( ['--smoke-cache'] if args.smoke else []),*common],status)
        expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_WEATHER_TRANSPORT'
        if read(status)['status']!=expected:raise ValueError('transport trajectory incomplete')
        if args.smoke:continue
        seal=validate_model_seal(folder,'COMPLETE_WEATHER_TRANSPORT',codes())
        for solver,steps in (('heun0',20),('ddim',40)):
            evaluation=folder/f'eval_{solver}'
            run('native_weather_gp_diffusion_experiment',['--phase','evaluate',
                '--cache',cache/f'calendar8_s{seed}','--embeddings',emb/f'{name}_dev_s{seed}',
                '--process',folder,'--out',evaluation,'--members',32,'--sample-steps',steps,'--solver',solver,*common],evaluation_marker(evaluation,seal))
            evaluation_marker(evaluation,seal)
            if solver=='heun0' and not (evaluation/'joint_metrics.json').exists():
                if STOP.exists():raise SystemExit(75)
                subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                    '--cache',str(cache/f'calendar8_s{seed}'),'--evaluation',str(evaluation),'--out',str(evaluation)],cwd=ROOT,check=True)


if __name__=='__main__':main()
