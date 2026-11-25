"""Matched covariance-only175/265 controls on frozen station-causal marginals."""
import argparse
import subprocess
import sys
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed
    cache=RESULTS/'20261126_native_weather_cache_v2';embedding=RESULTS/'20261126_native_weather_embeddings'
    name='transformer72_physics_aux_last';gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    root=RESULTS/'20261126_weather_projective_gp'/f's{seed}'
    common=['--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed]
    reference=root/'reference'
    run('native_weather_gp_experiment',['--phase','reference','--cache',cache/f'calendar8_s{seed}',
        '--embeddings',embedding/f'{name}_dev_s{seed}','--out',reference,*common],reference/'metrics.json')
    targets=[reference]
    for view in (175,265):
        folder=root/f'c{view}'
        run('native_weather_gp_experiment',['--phase','train','--cache',cache/f'train_s{seed}',
            '--embeddings',embedding/f'{name}_train_s{seed}','--out',folder,'--view',view,
            '--steps',3000,'--lr',3e-4,'--metric-lr',1e-3,*common],folder/'status.json')
        if read(folder/'status.json')['status']!='COMPLETE_WEATHER_GP':raise ValueError('weather covariance training incomplete')
        evaluation=folder/'eval'
        run('native_weather_gp_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',embedding/f'{name}_dev_s{seed}','--model',folder,'--out',evaluation,*common],evaluation/'metrics.json')
        targets.append(evaluation)
    for target in targets:
        if not (target/'joint_metrics.json').exists():
            if STOP.exists():raise SystemExit(75)
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                '--cache',str(cache/f'calendar8_s{seed}'),'--evaluation',str(target),'--out',str(target)],cwd=ROOT,check=True)


if __name__=='__main__':main()
