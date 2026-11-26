"""Matched continuation of the frozen native Transformer, C/H/S separately."""
import argparse
from campaign import RESULTS, read, run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=parser.parse_args();seed=args.seed
    cache=RESULTS/'20261126_native_weather_cache_v2'
    train=cache/f'train_s{seed}';dev=cache/f'calendar8_s{seed}'
    source=RESULTS/'20261126_native_weather_means'/f'transformer72_physics_aux_s{seed}'
    root=RESULTS/'20261126_mean_fusion_v2'
    for arm in ('C','H','S'):
        folder=root/f'{arm}_s{seed}'
        run('native_mean_fusion_v2_experiment',['--phase','train','--cache',train,
            '--source',source,'--arm',arm,'--seed',seed,'--steps',3000,'--out',folder],folder/'status.json')
        status=read(folder/'status.json')
        if status['status']!='COMPLETE_MEAN' or status['steps']!=3000:
            raise ValueError('continuation must complete all3000 blocks')
        for weights in ('last','ema'):
            embedding=root/f'{arm}_{weights}_dev_s{seed}'
            run('native_mean_fusion_v2_experiment',['--phase','export','--cache',dev,
                '--mean',folder,'--weights',weights,'--out',embedding],embedding/'manifest.json')
            evaluation=folder/f'eval_{weights}'
            run('native_mean_fusion_v2_experiment',['--phase','evaluate','--cache',dev,
                '--embeddings',embedding,'--out',evaluation],evaluation/'metrics.json')


if __name__=='__main__':main()
