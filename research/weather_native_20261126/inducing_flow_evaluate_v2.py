"""Same frozen evaluator; mask-aware completion and amended supplementary evidence gate."""
import argparse
from pathlib import Path
import subprocess
import sys
from campaign import RESULTS,read,run,STOP
from inducing_flow_series import ARMS,LOCK,source_paths,source_receipt
from inducing_flow_completion import AMENDMENT,validate_amendment,require_actual_completion
from tools.terrain_diffusion.native_inducing_flow_experiment import codes
from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
from tools.terrain_diffusion.data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);p.add_argument('--calendar-day',type=int,choices=[8,15],default=8)
    args=p.parse_args();seed=args.seed;day=args.calendar_day;amendment=validate_amendment();lock=read(LOCK)
    if lock['code_sha256']!=codes() or lock['sources'][str(seed)]!=source_receipt(seed):raise ValueError('locked source changed')
    for n,v in lock['evaluation_dependencies'].items():
        if sha(Path(__file__).with_name(n))!=v:raise ValueError('original scorer dependency changed')
    root=RESULTS/'20261126_inducing_latent_flow'
    if day==15:
        from inducing_flow_summary_v2 import require_supplement
        summary=read(root/'paired_calendar8_v2/summary.json')
        if summary.get('evaluation_amendment_sha256')!=sha(AMENDMENT):raise ValueError('supplement summary amendment differs')
        require_supplement(summary,sha(LOCK),sha(Path(__file__).with_name('inducing_flow_summary_v2.py')))
    paths=source_paths(seed,day);calendar=lock['sources'][str(seed)]['calendars'][str(day)]
    common=[v for n,path in paths.items() for v in ('--'+n,path)]+['--seed',seed]
    for family,domain in [('gaussian','time'),*ARMS]:
        if STOP.exists():raise SystemExit(75)
        model=root/f'{family}_{domain}_s{seed}';out=root/f'calendar{day}/{family}_{domain}_s{seed}'
        seal=None if family=='gaussian' else validate_model_seal(model,'COMPLETE_INDUCING_FLOW',codes())
        if seal:require_actual_completion(model,amendment['label_support'][str(seed)])
        run('native_inducing_flow_experiment',['--phase','evaluate',*common,'--family',family,'--domain',domain,
            '--process',model,'--out',out,'--sample-seed',lock['sample_seed'],'--device',lock['evaluation_device'],
            '--evaluation-cache-sha256',calendar['cache_manifest_sha256'],'--evaluation-embedding-sha256',calendar['embedding_manifest_sha256']],
            evaluation_marker(out,seal) if out.exists() else None)
        if not (out/'joint_metrics.json').exists():
            if STOP.exists():raise SystemExit(75)
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                '--evaluation',str(out),'--cache',str(paths['cache']),'--out',str(out)],check=True)
    print('Complete amended frozen evaluation',seed,day,flush=True)


if __name__=='__main__':main()
