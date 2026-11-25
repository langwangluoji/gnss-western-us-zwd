"""Execute registered local phases, checking a root-controlled quota stop flag."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_paths import file_path
RESULTS=file_path(ENGINE/'results')
STOP=Path(__file__).with_name('STOP_NEW_JOBS')


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def probability_variants():
    return [(name,name,False,('ddim',)) for name in ('gaussian','spatial','graph','joint','frequency_tokens')]+[
        ('preconditioned_joint','joint',True,('ddim','heun')),
        ('preconditioned_frequency_tokens','frequency_tokens',True,('ddim','heun'))]


def run(module,arguments,done=None):
    if STOP.exists():raise SystemExit(75)
    if done is not None and done.exists():
        print(json.dumps(dict(reused_completed_artifact=str(done))),flush=True);return
    command=[sys.executable,'-m','tools.terrain_diffusion.'+module,*map(str,arguments)]
    print(json.dumps(dict(starting=command)),flush=True)
    env=os.environ.copy();env['GNSS_RESEARCH_STOP_FILE']=str(STOP)
    subprocess.run(command,cwd=ENGINE,env=env,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calendar','means','physics','probability'],required=True)
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--mean-name',default='transformer72')
    p.add_argument('--mean-weights',choices=['last','ema'],default='ema');args=p.parse_args();seed=args.seed
    cache=RESULTS/'20261126_native_weather_cache_v2';train=cache/f'train_s{seed}';dev=cache/f'calendar8_s{seed}'
    means=RESULTS/'20261126_native_weather_means';emb=RESULTS/'20261126_native_weather_embeddings'
    if args.phase=='calendar':
        command=['--out',f'results/20261126_native_weather_cache_v2/calendar8_s{seed}','--seed',seed,'--calendar-day',8]
        prefix=RESULTS/'20261126_native_weather_cache'/f'calendar8_s{seed}'
        if (prefix/'status.json').exists():command+=['--prefix-cache',str(prefix.relative_to(file_path(ENGINE)))]
        run('native_weather_cache',command,dev/'manifest.json');return
    for path in (train,dev):
        if not (path/'manifest.json').exists() or read(path/'status.json')['status']!='COMPLETE':raise ValueError(f'phase requires complete cache: {path}')
    if args.phase in ('means','physics'):
        if args.phase=='means':
            configs=[('history_mlp','weather',72,True,'baseline'),('weather_mlp','weather',72,False,'baseline'),
                ('gru72','gru',72,False,'baseline'),('lstm72','lstm',72,False,'baseline'),
                ('transformer72','transformer',72,False,'baseline'),('transformer24','transformer',24,False,'baseline')]
        else:
            configs=[(f'transformer72_{v}','transformer',72,False,v) for v in ('local','aux','physics','physics_aux')]
            configs += [(f'{kind}72_physics_aux',kind,72,False,'physics_aux') for kind in ('gru','lstm')]
        for name,kind,length,no_weather,variant in configs:
            folder=means/f'{name}_s{seed}'
            command=['--phase','mean','--cache',train,'--out',folder,'--kind',kind,'--length',length,'--seed',seed,'--steps',3000]
            if no_weather:command+=['--no-weather']
            if variant!='baseline':command+=['--variant',variant]
            run('native_weather_experiment',command,folder/'status.json')
            if read(folder/'status.json')['status']!='COMPLETE_MEAN' or read(folder/'status.json')['steps']!=3000:raise ValueError('mean trajectory is not complete')
            for weights in ('last','ema'):
                target=emb/f'{name}_{weights}_dev_s{seed}'
                run('native_weather_experiment',['--phase','embeddings','--cache',dev,'--out',target,'--mean',folder,'--weights',weights],target/'manifest.json')
                evaluation=folder/f'eval_{weights}'
                run('native_weather_evaluation',['--cache',dev,'--embeddings',target,'--out',evaluation],evaluation/'metrics.json')
        return
    name=args.mean_name;weights=args.mean_weights;folder=means/f'{name}_s{seed}'
    train_embedding=emb/f'{name}_{weights}_train_s{seed}';dev_embedding=emb/f'{name}_{weights}_dev_s{seed}'
    run('native_weather_experiment',['--phase','embeddings','--cache',train,'--out',train_embedding,'--mean',folder,'--weights',weights],train_embedding/'manifest.json')
    probability=RESULTS/'20261126_native_weather_probability'/f'{name}_{weights}_s{seed}'
    for domain,effective,preconditioned,solvers in probability_variants():
        path=probability/domain
        command=['--phase','probability','--cache',train,'--embeddings',train_embedding,'--out',path,'--domain',effective,'--seed',seed,'--steps',3000]
        if preconditioned:command+=['--preconditioner',probability/'gaussian']
        run('native_weather_experiment',command,path/'status.json')
        if read(path/'status.json')['status']!='COMPLETE_PROBABILITY':raise ValueError('probability trajectory incomplete')
        if domain=='gaussian':
            run('native_copula',['--cache',train,'--embeddings',train_embedding,'--gaussian',path],path/'copula.json')
        for solver in solvers:
            evaluation=path/f'eval_{solver}'
            run('native_weather_evaluation',['--cache',dev,'--embeddings',dev_embedding,'--probability',path,
                '--out',evaluation,'--members',32,'--sample-steps',20 if solver=='heun' else 40,'--solver',solver],evaluation/'metrics.json')
        if domain=='gaussian':
            evaluation=path/'eval_copula'
            run('native_weather_evaluation',['--cache',dev,'--embeddings',dev_embedding,'--probability',path,
                '--copula',path/'copula.json','--out',evaluation,'--members',32],evaluation/'metrics.json')


if __name__=='__main__':main()
