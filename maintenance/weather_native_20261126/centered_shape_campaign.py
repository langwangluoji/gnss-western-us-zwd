"""Matched centered-scale/shape controls, after the predeclared diffusion line."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read,run
from tools.terrain_diffusion.native_model_seal import seal_completed_model,validate_model_seal,evaluation_marker
from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
from tools.terrain_diffusion.native_centered_shape_experiment import codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--smoke',action='store_true');p.add_argument('--wait-for-transport',action='store_true');args=p.parse_args();seed=args.seed
    if args.wait_for_transport:
        marker=RESULTS/'20261126_weather_transport'/f's{seed}/affine/query_audit.json'
        print('Waiting for completed predeclared transport and query audit',flush=True)
        while not marker.exists():
            if STOP.exists():raise SystemExit(75)
            time.sleep(15)
    gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    cache=RESULTS/'20261126_native_weather_cache_v2';emb=RESULTS/'20261126_native_weather_embeddings';name='transformer72_physics_aux_last'
    if args.smoke:
        if seed!=42:raise ValueError('engineering source is seed42')
        gp=RESULTS/'20261126_weather_gp_gap_metric/c265_s42';source=read(gp/'protocol.json')['config']
        traincache=source['cache'];trainemb=source['embeddings'];root=RESULTS/'20261126_centered_shape_gap'
    else:
        gp=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265'
        seal_completed_model(gp,'COMPLETE_WEATHER_GP',gp_codes())
        traincache=cache/f'train_s{seed}';trainemb=emb/f'{name}_train_s{seed}';root=RESULTS/'20261126_centered_shape'/f's{seed}'
    common=['--gaussian',gaussian,'--copula',gaussian/'copula.json','--gp',gp,'--seed',seed]
    targets=[]
    if not args.smoke:
        reference=root/'reference'
        run('native_centered_shape_experiment',['--phase','reference','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',emb/f'{name}_dev_s{seed}','--out',reference,*common],evaluation_marker(reference))
        evaluation_marker(reference);targets.append(reference)
    for mode in ('scale','shape','both'):
        folder=root/mode
        run('native_centered_shape_experiment',['--phase','train','--mode',mode,'--cache',traincache,'--embeddings',trainemb,
            '--out',folder,'--steps',4 if args.smoke else 3000,*( ['--smoke-cache'] if args.smoke else []),*common],folder/'status.json')
        expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_CENTERED_SHAPE'
        if read(folder/'status.json')['status']!=expected:raise ValueError('centered control training incomplete')
        if args.smoke:continue
        seal=validate_model_seal(folder,'COMPLETE_CENTERED_SHAPE',codes());evaluation=folder/'eval'
        run('native_centered_shape_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',emb/f'{name}_dev_s{seed}','--model',folder,'--out',evaluation,*common],evaluation_marker(evaluation,seal))
        evaluation_marker(evaluation,seal);targets.append(evaluation)
    for target in targets:
        if STOP.exists():raise SystemExit(75)
        if not (target/'joint_metrics.json').exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                '--cache',str(cache/f'calendar8_s{seed}'),'--evaluation',str(target),'--out',str(target)],cwd=ROOT,check=True)


if __name__=='__main__':main()
