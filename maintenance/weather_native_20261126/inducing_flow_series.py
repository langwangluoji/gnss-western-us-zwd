"""Frozen four-family inducing flow queue; same coupled GP sampling for all arms."""
import argparse
from pathlib import Path
import subprocess
import sys
from campaign import RESULTS,read,run,STOP
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.native_inducing_flow_experiment import codes
from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes


LOCK=Path(__file__).with_name('inducing_flow_lock.json')
ARMS=[(family,domain) for family in ('affine','odd') for domain in ('time','frequency')]


def source_paths(seed,calendar=0):
    gp=RESULTS/f'20261126_weather_projective_gp/s{seed}/c265';cfg=read(gp/'protocol.json')['config']
    paths={n:file_path(cfg[n]) for n in ('cache','embeddings','gaussian','copula')};paths['gp']=gp
    if calendar:
        paths['cache']=RESULTS/f'20261126_native_weather_cache_v2/calendar{calendar}_s{seed}'
        suffix='dev' if calendar==8 else 'calendar15'
        paths['embeddings']=RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_last_{suffix}_s{seed}'
    return paths


def source_receipt(seed):
    paths=source_paths(seed);seal=validate_model_seal(paths['gp'],'COMPLETE_WEATHER_GP',gp_codes())
    calendars={str(day):dict(cache_manifest_sha256=sha(source_paths(seed,day)['cache']/'manifest.json'),
        embedding_manifest_sha256=sha(source_paths(seed,day)['embeddings']/'manifest.json')) for day in (8,15)}
    return dict(calendars=calendars,gp_seal_sha256=sha(paths['gp']/'model_artifact_seal.json'),gp_weights_sha256=seal['weights_sha256'],
        gp_protocol_sha256=seal['protocol_sha256'],cache_manifest_sha256=sha(paths['cache']/'manifest.json'),
        embedding_manifest_sha256=sha(paths['embeddings']/'manifest.json'),
        gaussian_weights_sha256=sha(paths['gaussian']/'last.pt'),copula_sha256=sha(paths['copula']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['lock','smoke','train','evaluate'],required=True)
    p.add_argument('--seed',type=int,choices=[42,17],default=42);p.add_argument('--calendar-day',type=int,choices=[8,15],default=8)
    args=p.parse_args();seed=args.seed
    if args.phase=='lock':
        if LOCK.exists():raise ValueError('lock already exists; never rewrite a scientific lock')
        write_json(LOCK,dict(code_sha256=codes(),driver_sha256=sha(__file__),sources={str(s):source_receipt(s) for s in (42,17)},
            evaluation_dependencies={n:sha(Path(__file__).with_name(n)) for n in
                ('probability_joint_audit.py','inducing_flow_summary.py','weather_controls_summary.py','lead_mean_audit.py','event_audit.py','mean_hs_summary.py')},
            event_lock_sha256=sha(Path(__file__).with_name('event_audit_lock.json')),
            arms=ARMS,steps=3000,members_training=8,queries_training=32,members_evaluation=32,main_calendar=8,
            sample_seed=261126,evaluation_device='cuda',
            point_mean='frozen TF3000',scale='frozen S175 conditional std',same_noise=True,
            primary='odd_frequency vs affine_frequency F6 fairCRPS; compare affine_frequency vs frozen Gaussian separately',
            secondary=['internal increment CRPS','geometry-stratified pair CRPS','energy','coverage','cost'],
            supplement='all frozen arms calendar15 if raw odd_frequency improves main F6 fairCRPS over affine_frequency',
            time_frequency_comparison='same family, conditions, all7 coefficients, parameters and members',
            strict_oof=False,confirmation_C_D_read=False))
        print('Locked inducing-flow protocol',flush=True);return
    if args.phase!='smoke':
        lock=read(LOCK)
        if lock['code_sha256']!=codes() or lock['driver_sha256']!=sha(__file__) or lock['sources'][str(seed)]!=source_receipt(seed):
            raise ValueError('locked implementation or source differs')
        for n,expected in lock['evaluation_dependencies'].items():
            if sha(Path(__file__).with_name(n))!=expected:raise ValueError('locked evaluator changed')
    paths=source_paths(seed,args.calendar_day if args.phase=='evaluate' else 0)
    common=[v for n,path in paths.items() for v in ('--'+n,path)]+['--seed',seed]
    root=RESULTS/'20261126_inducing_latent_flow'
    if args.phase=='smoke':
        out=root/f'smoke_s{seed}'
        run('native_inducing_flow_experiment',['--phase','train',*common,'--family','odd','--domain','frequency',
            '--steps',4,'--smoke-cache','--out',out],out/'status.json')
        status=read(out/'status.json')
        if status['status']!='ENGINEERING_SMOKE_COMPLETE' or status['supervised_updates']!=4:raise ValueError('smoke incomplete')
        return
    if args.phase=='train':
        for family,domain in ARMS:
            out=root/f'{family}_{domain}_s{seed}'
            run('native_inducing_flow_experiment',['--phase','train',*common,'--family',family,'--domain',domain,'--out',out],out/'status.json')
            validate_model_seal(out,'COMPLETE_INDUCING_FLOW',codes())
        print('Complete inducing-flow training',seed,flush=True);return
    if args.calendar_day==15:
        from inducing_flow_summary import require_supplement
        summary=read(root/'paired_calendar8/summary.json')
        require_supplement(summary,sha(LOCK),sha(Path(__file__).with_name('inducing_flow_summary.py')))
    for family,domain in [('gaussian','time'),*ARMS]:
        if STOP.exists():raise SystemExit(75)
        model=root/f'{family}_{domain}_s{seed}';out=root/f'calendar{args.calendar_day}/{family}_{domain}_s{seed}'
        seal=None if family=='gaussian' else validate_model_seal(model,'COMPLETE_INDUCING_FLOW',codes())
        marker=evaluation_marker(out,seal) if out.exists() else None
        calendar=lock['sources'][str(seed)]['calendars'][str(args.calendar_day)]
        run('native_inducing_flow_experiment',['--phase','evaluate',*common,'--family',family,'--domain',domain,
            '--process',model,'--out',out,'--sample-seed',lock['sample_seed'],'--device',lock['evaluation_device'],
            '--evaluation-cache-sha256',calendar['cache_manifest_sha256'],
            '--evaluation-embedding-sha256',calendar['embedding_manifest_sha256']],marker)
        if not (out/'joint_metrics.json').exists():
            if STOP.exists():raise SystemExit(75)
            subprocess.run([sys.executable,str(Path(__file__).with_name('probability_joint_audit.py')),
                '--evaluation',str(out),'--cache',str(paths['cache']),'--out',str(out)],check=True)
    print('Complete inducing-flow evaluation',seed,args.calendar_day,flush=True)


if __name__=='__main__':main()
