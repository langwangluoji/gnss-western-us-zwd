"""Frozen loss-only H continuation, then main-calendar raw/EMA evaluation."""
import argparse
from campaign import RESULTS,read,run,STOP
from tools.terrain_diffusion.native_increment_experiment import code_hashes
from tools.terrain_diffusion.native_model_seal import validate_model_seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--zero',action='store_true');args=p.parse_args()
    if args.zero and not args.smoke:raise ValueError('zero weight only for engineering replay')
    if STOP.exists():raise SystemExit(75)
    cache=RESULTS/'20261126_native_weather_cache_v2';seed=args.seed
    source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
    root=RESULTS/('20261126_increment_smoke' if args.smoke else '20261126_within_forecast_increment')
    name='H_zero' if args.zero else 'H_delta';folder=root/f'{name}_s{seed}'
    run('native_increment_experiment',['--phase','train','--cache',cache/f'train_s{seed}',
        '--source',source,'--out',folder,'--seed',seed,'--steps',4 if args.smoke else 3000,
        '--increment-weight',0 if args.zero else .5,*(['--smoke-cache'] if args.smoke else [])],folder/'status.json')
    status=read(folder/'status.json');protocol=read(folder/'protocol.json')
    expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_MEAN_INCREMENT'
    if status['status']!=expected or status['steps']!=(4 if args.smoke else 3000) or protocol['code_sha256']!=code_hashes():
        raise ValueError('increment training incomplete or changed')
    if args.smoke:return
    validate_model_seal(folder,expected,code_hashes())
    for weights in ('last','ema'):
        embedding=root/f'H_delta_{weights}_dev_s{seed}';evaluation=folder/f'eval_{weights}'
        run('native_increment_experiment',['--phase','export','--cache',cache/f'calendar8_s{seed}',
            '--mean',folder,'--weights',weights,'--out',embedding,'--seed',seed],embedding/'manifest.json')
        manifest=read(embedding/'manifest.json')
        if manifest['mean_variant']!='increment_H_delta' or manifest['code_sha256']!=code_hashes() or not manifest['complete']:
            raise ValueError('increment export changed')
        run('native_increment_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',embedding,'--out',evaluation,'--seed',seed],evaluation/'metrics.json')
    print('Complete H_delta final3000 and raw/EMA evaluation',flush=True)


if __name__=='__main__':main()
