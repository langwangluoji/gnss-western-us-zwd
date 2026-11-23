"""One bounded A800 preflight, then a CPU-only handoff to the serial campaign."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import argparse
import time

from gnss_field.artifacts import immutable_json,read_json,write_json
from gnss_field.generative.campaign import (Campaign,load_config,build_data,estimate_budget,
                                           diagnostic_preflight_config)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('command',choices=('preflight','handoff'))
    args=parser.parse_args(argv);config=load_config(args.config);campaign=Campaign(config)
    status_path=campaign.root/'preflight_status.json'
    def status(state,**extra):
        value=dict(state=state,updated_unix=time.time(),**extra)
        write_json(status_path,value);print('PREFLIGHT',value,flush=True)
    if args.command=='handoff':
        previous=read_json(status_path) if status_path.exists() else {}
        if previous.get('state')!='COMPLETE':
            print('NO_TRAINING_SUBMITTED',previous,flush=True);return
        # Slurm accounting may become visible a few seconds after afterany fires.
        for attempt in range(6):
            try:
                result=campaign.submit(args.config)
                write_json(campaign.root/'first_training_submission.json',result)
                print('FIRST_TRAINING_SUBMITTED',result,flush=True);return
            except ValueError as exc:
                if 'accounting' not in str(exc).lower() and 'sacct' not in str(exc).lower():raise
                if attempt==5:raise
                time.sleep(10)
        return
    import torch
    from gnss_field.generative.benchmark import run_benchmark
    from gnss_field.generative.diagnostics import run_representation_diagnostics
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    job=os.environ.get('SLURM_JOB_ID')
    if not job or not job.isdigit():raise RuntimeError('Preflight requires a registered Slurm allocation')
    if not any(row['job_id']==job for row in campaign.manifest['jobs']):
        campaign.manifest['jobs'].append(dict(job_id=job,stage='benchmark',allocated_seconds=3600))
        campaign.save()
    clock=campaign.root/'preflight_clock.lock.json'
    if clock.exists():raise RuntimeError('Preflight is not automatically rerun; previous allocation remains charged')
    immutable_json(clock,dict(started_unix=time.time(),cap_seconds=3600,slurm_job_id=job))
    try:
        import copy
        benchmark_config=copy.deepcopy(config)
        benchmark_config['data']['cache_root']=str(campaign.root/'benchmark_cache')
        status('BENCHMARK',job_id=job)
        data=build_data(benchmark_config,allow_background_inference=True)
        report=run_benchmark(config,data,campaign.root/'benchmark.json')
        del data;torch.cuda.empty_cache()
        status('BUDGET_CHECK',job_id=job)
        audit=read_json(campaign.root/'audit_B.json')
        # This happens BEFORE full cache preparation and BEFORE any pilot.
        prior=sum(float(row.get('elapsed_seconds',0)) for row in campaign.manifest['jobs'] if row['job_id']!=job)
        estimate=estimate_budget(audit['qualified_hours'],report,accounted_seconds=prior)
        write_json(campaign.root/'budget_estimate.json',estimate)
        if config['runtime'].get('staged_preparation',False) or config['runtime'].get('preflight_mode')=='benchmark_only':
            campaign.lock_budget(report,accounted_seconds=prior)
            status('COMPLETE',job_id=job,mode='benchmark_only',measured_total_card_hours=estimate['total_seconds']/3600)
            return
        diagnostic_preflight_config(campaign)
        status('FULL_CACHE_PREPARATION',job_id=job)
        data=build_data(config,allow_background_inference=True)
        development=data._other_view('development')
        data.prepare(role='B');development.prepare(development.audit('C')['hour_ids'],role='C')
        limited=diagnostic_preflight_config(campaign)
        status('TRAINING_INTERNAL_DIAGNOSTICS',job_id=job)
        result=run_representation_diagnostics(limited,data,campaign.root)
        immutable_json(campaign.root/'diagnostics.json',{**result,'passed':result.get('status')=='complete'})
        campaign.lock_budget()
        status('COMPLETE',job_id=job,latent_dim=result['latent_dim'],sigma_eff_mm=result['sigma_eff_mm'])
    except (ValueError,TimeoutError) as exc:
        message=str(exc)
        if any(word in message.lower() for word in ('budget','cap','card-hour','envelope')):
            status('STOP_BUDGET',job_id=job,reason=message);return
        status('ERROR',job_id=job,reason=message);raise
    except Exception as exc:
        status('ERROR',job_id=job,reason=str(exc));raise


if __name__=='__main__':main()
