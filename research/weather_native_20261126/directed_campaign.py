"""Matched directed/factorized station-lag means and frozen evaluations."""
import argparse
from campaign import RESULTS,read,run
from tools.terrain_diffusion.native_directed_lag_experiment import code_hashes
from tools.terrain_diffusion.native_model_seal import validate_model_seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True)
    p.add_argument('--mode',choices=['factorized','joint'],required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--disabled',action='store_true');args=p.parse_args();seed=args.seed
    if args.disabled and not args.smoke:raise ValueError('disabled arm is engineering only')
    cache=RESULTS/'20261126_native_weather_cache_v2';source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
    root=RESULTS/('20261126_directed_lag_smoke' if args.smoke else '20261126_directed_lag')
    name='disabled' if args.disabled else args.mode;folder=root/f'{name}_s{seed}'
    run('native_directed_lag_experiment',['--phase','train','--cache',cache/f'train_s{seed}',
        '--source',source,'--out',folder,'--seed',seed,'--mode',args.mode,'--steps',4 if args.smoke else 3000,
        *(['--smoke-cache'] if args.smoke else []),*(['--disabled-transport'] if args.disabled else [])],folder/'status.json')
    status=read(folder/'status.json');protocol=read(folder/'protocol.json')
    expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_DIRECTED_LAG'
    if status['status']!=expected or status['steps']!=(4 if args.smoke else 3000) or protocol['code_sha256']!=code_hashes():
        raise ValueError('directed continuation incomplete or changed')
    if protocol['config']['mode']!=args.mode:raise ValueError('attention mode changed')
    if args.smoke:return
    validate_model_seal(folder,expected,code_hashes())
    for weights in ('last','ema'):
        embedding=root/f'{name}_{weights}_dev_s{seed}';evaluation=folder/f'eval_{weights}'
        run('native_directed_lag_experiment',['--phase','export','--cache',cache/f'calendar8_s{seed}',
            '--mean',folder,'--weights',weights,'--out',embedding,'--seed',seed],embedding/'manifest.json')
        manifest=read(embedding/'manifest.json')
        if manifest['mean_variant']!='directed_'+args.mode or manifest['code_sha256']!=code_hashes():raise ValueError('export recipe differs')
        run('native_directed_lag_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',embedding,'--out',evaluation,'--seed',seed],evaluation/'metrics.json')
    print('Complete directed',args.mode,seed,flush=True)


if __name__=='__main__':main()
