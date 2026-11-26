"""Capacity-matched full-field Gaussian control of the nonlinear graph diffusion."""
import argparse
from campaign import RESULTS,run,read


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed
    cache=RESULTS/'20261126_native_weather_cache_v2';embedding=RESULTS/'20261126_native_weather_embeddings'
    name='transformer72_physics_aux_last';gaussian=RESULTS/'20261126_probability_context'/f'S175_s{seed}'
    root=RESULTS/'20261126_gp_process'/f's{seed}'/'graph_affine'
    common=['--gaussian',gaussian,'--copula',gaussian/'copula.json','--seed',seed,'--mode','full']
    run('native_graph_gaussian_experiment',['--phase','train','--cache',cache/f'train_s{seed}',
        '--embeddings',embedding/f'{name}_train_s{seed}','--out',root,*common],root/'status.json')
    if read(root/'status.json')['status']!='COMPLETE_PROCESS':raise ValueError('graph Gaussian not complete')
    for solver,steps in [('heun0',20),('ddim',40)]:
        target=root/f'eval_{solver}'
        run('native_graph_gaussian_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',embedding/f'{name}_dev_s{seed}','--process',root,'--solver',solver,
            '--sample-steps',steps,'--out',target,*common],target/'metrics.json')


if __name__=='__main__':main()
