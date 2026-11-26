"""Run a registered exact leave-out background diagnostic; no fitting."""
import argparse
from campaign import RESULTS,run,read
from tools.terrain_diffusion.native_innovation_audit import codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    p.add_argument('--index',type=int,default=0);p.add_argument('--origins',type=int,default=2);args=p.parse_args()
    cache=RESULTS/f'20261126_native_weather_cache_v2/train_s{args.seed}'
    embedding=RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_last_train_s{args.seed}'
    out=RESULTS/f'20261126_leaveout_innovation_audit/b{args.index:04d}_o{args.origins:02d}_s{args.seed}'
    run('native_innovation_audit',['--cache',cache,'--embedding',embedding,'--out',out,
        '--index',args.index,'--origins',args.origins],out/'summary.json')
    if read(out/'summary.json')['code_sha256']!=codes():raise ValueError('audit source changed')


if __name__=='__main__':main()
