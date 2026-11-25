"""Registered condition-information controls on one frozen causal mean."""
import argparse
from pathlib import Path
import subprocess
import sys
from campaign import ROOT,RESULTS,STOP,read,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed;name='transformer72_physics_aux'
    cache=RESULTS/'20261126_native_weather_cache_v2';train=cache/f'train_s{seed}';dev=cache/f'calendar8_s{seed}'
    embeddings=RESULTS/'20261126_native_weather_embeddings'
    train_embedding=embeddings/f'{name}_last_train_s{seed}';dev_embedding=embeddings/f'{name}_last_dev_s{seed}'
    mean=RESULTS/'20261126_native_weather_means'/f'{name}_s{seed}'
    if read(mean/'status.json')['status']!='COMPLETE_MEAN':raise ValueError('frozen mean is not complete')
    for data,target in ((train,train_embedding),(dev,dev_embedding)):
        run('native_weather_experiment',['--phase','embeddings','--cache',data,'--out',target,'--mean',mean,'--weights','last'],target/'manifest.json')
    for label,view,width in [('R111','raw',64),('S175','skip',64),('Lwide','latent',154)]:
        folder=RESULTS/'20261126_probability_context'/f'{label}_s{seed}'
        run('native_weather_experiment',['--phase','probability','--cache',train,'--embeddings',train_embedding,'--out',folder,
            '--domain','gaussian','--condition-view',view,'--width',width,'--steps',3000,'--seed',seed],folder/'status.json')
        status=read(folder/'status.json')
        if status['status']!='COMPLETE_PROBABILITY' or status['steps']!=3000:raise ValueError('control is incomplete')
        evaluation=folder/'eval_ddim'
        run('native_weather_evaluation',['--cache',dev,'--embeddings',dev_embedding,'--probability',folder,'--out',evaluation,
            '--members',32,'--sample-steps',40],evaluation/'metrics.json')
        if not (evaluation/'calibration_audit.json').exists():
            if STOP.exists():raise SystemExit(75)
            subprocess.run([sys.executable,str(Path(__file__).with_name('calibration_audit.py')),
                '--cache',str(dev),'--embeddings',str(dev_embedding),'--evaluation',str(evaluation),
                '--gaussian',str(folder),'--out',str(evaluation)],cwd=ROOT,check=True)


if __name__=='__main__':main()
