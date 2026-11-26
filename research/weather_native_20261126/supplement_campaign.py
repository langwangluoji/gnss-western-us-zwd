"""Frozen Full/Local/G-ref verification on additional dates; no training."""
import argparse
import subprocess
import sys
from pathlib import Path
from campaign import ROOT,RESULTS,STOP,read,run
from tools.terrain_diffusion.data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed
    lock=read(Path(__file__).with_name('supplement_lock.json'))
    for path,expected in lock['sources'][str(seed)].items():
        if sha(Path(path))!=expected:raise ValueError('additional-calendar frozen weights changed')
    cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar15_s{seed}'
    run('native_weather_cache',['--out',cache,'--seed',seed,'--calendar-day',15],cache/'manifest.json')
    if read(cache/'status.json')['status']!='COMPLETE':raise ValueError('additional cache incomplete')
    audit=Path(__file__).with_name(f'calendar15_audit_s{seed}.json')
    if not audit.exists():
        if STOP.exists():raise SystemExit(75)
        subprocess.run([sys.executable,str(Path(__file__).with_name('audit_cache.py')),'--cache',str(cache),'--out',str(audit)],cwd=ROOT,check=True)
    mean=RESULTS/'20261126_native_weather_means'/f'transformer72_physics_aux_s{seed}'
    embedding=RESULTS/'20261126_native_weather_embeddings'/f'transformer72_physics_aux_last_calendar15_s{seed}'
    run('native_weather_experiment',['--phase','embeddings','--cache',cache,'--mean',mean,'--weights','last','--out',embedding],embedding/'manifest.json')
    gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}';root=RESULTS/'20261126_gp_process_calendar15'/f's{seed}'
    common=['--cache',cache,'--embeddings',embedding,'--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed]
    for mode in lock['primary_candidates']:
        target=root/mode
        if mode=='reference':command=['--phase','reference']
        else:command=['--phase','evaluate','--process',RESULTS/'20261126_gp_process'/f's{seed}'/mode]
        run('native_process_experiment',[*command,*common,'--out',target,'--members',32,'--sample-steps',20,'--solver','heun0'],target/'metrics.json')
        if not (target/'joint_metrics.json').exists():
            if STOP.exists():raise SystemExit(75)
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                '--cache',str(cache),'--evaluation',str(target),'--out',str(target)],cwd=ROOT,check=True)


if __name__=='__main__':main()
