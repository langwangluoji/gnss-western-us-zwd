"""Execute GP reference and matched Full/Local/Affine controls, resume by receipts."""
import argparse
from campaign import RESULTS,read,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed
    cache=RESULTS/'20261126_native_weather_cache_v2'
    embedding=RESULTS/'20261126_native_weather_embeddings'
    train=cache/f'train_s{seed}';dev=cache/f'calendar8_s{seed}'
    name='transformer72_physics_aux_last'
    te=embedding/f'{name}_train_s{seed}';de=embedding/f'{name}_dev_s{seed}'
    gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    if not (gaussian/'copula.json').exists():
        run('native_copula',['--cache',train,'--embeddings',te,'--gaussian',gaussian],gaussian/'copula.json')
    root=RESULTS/'20261126_gp_process'/f's{seed}'
    common=['--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed]
    reference=root/'reference'
    run('native_process_experiment',['--phase','reference','--cache',dev,'--embeddings',de,
        '--out',reference,*common],reference/'metrics.json')
    for mode in ('full','local','affine'):
        folder=root/mode
        run('native_process_experiment',['--phase','train','--mode',mode,'--cache',train,
            '--embeddings',te,'--out',folder,*common],folder/'status.json')
        if read(folder/'status.json')['status']!='COMPLETE_PROCESS':raise ValueError('process not complete')
        for solver,steps in [('heun0',20),('ddim',40)]:
            target=folder/f'eval_{solver}'
            run('native_process_experiment',['--phase','evaluate','--cache',dev,'--embeddings',de,
                '--process',folder,'--solver',solver,'--sample-steps',steps,'--out',target,*common],target/'metrics.json')


if __name__=='__main__':main()
