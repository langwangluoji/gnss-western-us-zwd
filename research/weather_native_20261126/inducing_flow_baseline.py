"""Generate the locked Gaussian reference while the independent flow training runs."""
import argparse
from pathlib import Path
import subprocess
import sys
from campaign import RESULTS,read,run,STOP
from inducing_flow_series import LOCK,source_paths,source_receipt
from tools.terrain_diffusion.native_inducing_flow_experiment import codes
from tools.terrain_diffusion.native_model_seal import evaluation_marker
from tools.terrain_diffusion.data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);seed=p.parse_args().seed
    lock=read(LOCK)
    if lock['code_sha256']!=codes() or lock['sources'][str(seed)]!=source_receipt(seed):raise ValueError('locked source changed')
    for n,v in lock['evaluation_dependencies'].items():
        if sha(Path(__file__).with_name(n))!=v:raise ValueError('locked scorer changed')
    paths=source_paths(seed,8);calendar=lock['sources'][str(seed)]['calendars']['8']
    out=RESULTS/f'20261126_inducing_latent_flow/calendar8/gaussian_time_s{seed}'
    common=[v for n,path in paths.items() for v in ('--'+n,path)]
    run('native_inducing_flow_experiment',['--phase','evaluate',*common,'--seed',seed,'--family','gaussian','--domain','time',
        '--sample-seed',lock['sample_seed'],'--device',lock['evaluation_device'],'--out',out,
        '--evaluation-cache-sha256',calendar['cache_manifest_sha256'],'--evaluation-embedding-sha256',calendar['embedding_manifest_sha256']],
        evaluation_marker(out) if out.exists() else None)
    if STOP.exists():raise SystemExit(75)
    if not (out/'joint_metrics.json').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
            '--evaluation',str(out),'--cache',str(paths['cache']),'--out',str(out)],check=True)


if __name__=='__main__':main()
