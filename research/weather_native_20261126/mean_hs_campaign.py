"""Registered HS continuation and frozen raw/EMA main-calendar evaluation."""
import argparse
from campaign import RESULTS,read,run,STOP
from tools.terrain_diffusion.native_mean_hs_experiment import code_hashes
from tools.terrain_diffusion.native_model_seal import validate_model_seal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,17],required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();seed=args.seed
    if STOP.exists():raise SystemExit(75)
    cache=RESULTS/'20261126_native_weather_cache_v2';source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
    root=RESULTS/('20261126_mean_hs_gap_v2' if args.smoke else '20261126_mean_hs');folder=root/f'HS_s{seed}'
    run('native_mean_hs_experiment',['--phase','train','--cache',cache/f'train_s{seed}',
        '--source',source,'--out',folder,'--seed',seed,'--steps',4 if args.smoke else 3000,
        *(['--smoke-cache'] if args.smoke else [])],folder/'status.json')
    expected='ENGINEERING_SMOKE_COMPLETE' if args.smoke else 'COMPLETE_MEAN_HS'
    status=read(folder/'status.json');protocol=read(folder/'protocol.json')
    if status['status']!=expected or status['steps']!=(4 if args.smoke else 3000) or protocol['code_sha256']!=code_hashes():
        raise ValueError('HS training recipe incomplete or changed')
    if args.smoke:return
    validate_model_seal(folder,'COMPLETE_MEAN_HS',code_hashes())
    for weights in ('last','ema'):
        embedding=root/f'HS_{weights}_dev_s{seed}';evaluation=folder/f'eval_{weights}'
        run('native_mean_hs_experiment',['--phase','export','--cache',cache/f'calendar8_s{seed}',
            '--mean',folder,'--weights',weights,'--out',embedding,'--seed',seed],embedding/'manifest.json')
        manifest=read(embedding/'manifest.json')
        if manifest['mean_variant']!='fusion_v2_HS' or manifest['code_sha256']!=code_hashes() or not manifest['complete']:
            raise ValueError('HS export recipe changed')
        run('native_mean_hs_experiment',['--phase','evaluate','--cache',cache/f'calendar8_s{seed}',
            '--embeddings',embedding,'--out',evaluation,'--seed',seed],evaluation/'metrics.json')
    print('Complete HS final3000 and raw/EMA evaluation',flush=True)


if __name__=='__main__':main()
