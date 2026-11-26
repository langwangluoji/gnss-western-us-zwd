"""Fail-closed, sequential single-A800 orchestration for the registered full run.

This module never invokes SSH. Submission is an explicit CLI operation; planning,
CPU audits and importing the module cannot submit a job or initialize CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import uuid

from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json


FAMILIES = ('vae', 'gan', 'diffusion')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL'}
RUNTIME_FIELDS = {'run_root', 'device', 'budget_seconds', 'max_updates', 'output_dir',
                  'checkpoint_every_updates', 'chunk_seconds', 'python', 'project_root'}


def stage_catalogue():
    rows = [dict(id='control', stage='deterministic_control', model_family='vae', epochs=10, batch_size=8),
            dict(id='vae_representation', stage='vae_representation', model_family='vae', epochs=30, batch_size=8),
            dict(id='posterior_cache', stage='posterior_cache', model_family='vae', epochs=1, batch_size=8),
            dict(id='gan_pretrain', stage='gan_pretrain', model_family='gan', epochs=20, batch_size=128),
            dict(id='diffusion_pretrain', stage='diffusion_pretrain', model_family='diffusion', epochs=20, batch_size=128)]
    for prefix, stage in (('shared', 'prior_es'), ('complete', 'complete_model_es')):
        rows.extend(dict(id=f'{prefix}_{family}', stage=stage, model_family=family,
                         epochs=3, batch_size=8) for family in FAMILIES)
    for row in rows:
        row['requires_representation_gate'] = row['id'] not in ('control', 'vae_representation')
    return rows + [dict(id='final_evaluation', stage='final_evaluation', requires_representation_gate=True)]


def stage_work(qualified_hours):
    if not isinstance(qualified_hours, int) or isinstance(qualified_hours, bool) or qualified_hours < 1:
        raise ValueError('A positive audited qualified-hour count is required')
    return {row['id']: dict(epochs=row['epochs'], batch_size=row['batch_size'],
            batches_per_epoch=math.ceil(qualified_hours/row['batch_size']),
            updates=math.ceil(qualified_hours/row['batch_size'])*row['epochs'],
            hours_exposed=qualified_hours*row['epochs'],
            kl_warmup_updates=(math.ceil(math.ceil(qualified_hours/row['batch_size'])*row['epochs']*.1) if row['id']=='vae_representation' else 0))
            for row in stage_catalogue() if 'epochs' in row}


def registered_development_hours(config):
    """The fixed576-hour C panel is chosen from the complete2019 calendar."""
    import numpy as np
    from .evaluation import fixed_development_hours
    calendar=np.arange(np.datetime64('2019-01-01','h').astype('i8'),
                       np.datetime64('2020-01-01','h').astype('i8'))
    hours=fixed_development_hours(calendar).tolist()
    for section in ('scientific','training','evaluation'):
        supplied=config.get(section,{}).get('development_hours')
        if supplied is not None and list(supplied)!=hours:
            raise ValueError('Registered development hours differ from the original-calendar576-hour panel')
    return hours


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'Invalid measured benchmark cost: {name}')
    return value


def estimate_budget(qualified_hours, benchmark, cap_card_hours=24, *, accounted_seconds=0):
    """Price every epoch, dual raw/EMA development call, write and final product.

    Measurements are A800 wall-clock seconds, not extrapolated published speeds.
    Allocation chunks are deliberately separate from immutable scientific caps.
    """
    cap = _positive(cap_card_hours, 'cap')*3600
    if cap > 24*3600:
        raise ValueError('Total card-hour cap exceeds authorized 24 hours')
    required = ('stage_seconds_per_update', 'evaluation_seconds_per_call',
                'checkpoint_seconds_per_write', 'posterior_cache_seconds_per_batch',
                'preparation_seconds', 'diagnostics_seconds', 'final_evaluation_seconds',
                'benchmark_elapsed_seconds', 'priced_latent_dim')
    missing = [name for name in required if name not in benchmark]
    if missing:
        raise ValueError('Missing benchmark: '+', '.join(missing))
    if benchmark.get('measured') is not True or benchmark.get('device') != 'A800' or benchmark.get('full_working_set') is not True:
        raise ValueError('Missing benchmark: measured full-working-set A800 evidence')
    priced_latent_dim=benchmark['priced_latent_dim']
    if priced_latent_dim not in (64,128): raise ValueError('Invalid benchmark priced latent dimension')
    elapsed = _positive(accounted_seconds, 'accounted_seconds', True)
    work = stage_work(qualified_hours)
    caps = {}
    for row in stage_catalogue():
        name = row['id']
        if 'epochs' not in row:
            continue
        spec = work[name]
        if name == 'posterior_cache':
            cost = (_positive(benchmark['posterior_cache_seconds_per_batch'], name)*spec['updates']+
                    _positive(benchmark.get('posterior_cache_startup_seconds',0),'posterior_cache_startup_seconds',True))
            evaluations = 0
        else:
            if name not in benchmark['stage_seconds_per_update']:
                raise ValueError('Missing benchmark: '+name)
            cost = (_positive(benchmark['stage_seconds_per_update'][name], name)*spec['updates']+
                    _positive(benchmark.get('stage_startup_seconds',{}).get(name,0),name+' startup',True))
            # Both raw and EMA are evaluated at step0 and after every full epoch.
            evaluations = 2*(row['epochs']+1+len(benchmark.get('evaluation_check_updates',[200])))
            try:
                cost += evaluations*_positive(benchmark['evaluation_seconds_per_call'][row['model_family']], 'evaluation')
            except KeyError as exc:
                raise ValueError('Missing benchmark: evaluation '+row['model_family']) from exc
        writes = math.ceil(spec['updates']/100)+row['epochs']+1+2*evaluations
        cost += writes*_positive(benchmark['checkpoint_seconds_per_write'], 'checkpoint')
        if name=='vae_representation' and 'representation_check_seconds' in benchmark:
            cost += _positive(benchmark['representation_check_seconds'],'representation_check_seconds')
        caps[name] = 1.5*cost
    for name, key in (('prepare','preparation_seconds'), ('diagnostics','diagnostics_seconds'),
                      ('final_evaluation','final_evaluation_seconds')):
        cost=_positive(benchmark[key], key)
        if name=='prepare': cost+=_positive(benchmark.get('preparation_startup_seconds',0),'preparation_startup_seconds',True)
        caps[name] = 1.5*cost
    measured_elapsed = _positive(benchmark['benchmark_elapsed_seconds'], 'benchmark_elapsed_seconds')
    # If real Slurm accounting already includes benchmark, never subtract it back.
    prior = elapsed+measured_elapsed
    total = prior+sum(caps.values())
    if total > cap:
        raise ValueError(f'Unsafe card-hour cap: measured plan {total/3600:.3f} > {cap/3600:.3f}')
    if prior > 3600:
        raise ValueError('Preflight short benchmark and rechecks exceed one card-hour cap')
    return dict(schema=1, qualified_hours=qualified_hours, benchmark_sha256=fingerprint(benchmark),
                safety_margin=1.5, priced_latent_dim=priced_latent_dim, stage_caps_seconds=caps, accounted_seconds=elapsed,
                prior_seconds=prior, total_seconds=total, cap_seconds=cap,
                work=work, preflight_cap_seconds=3600)


def representation_gate(values):
    values = dict(values)
    if 'grid_query_max_error_mm' in values:
        values.setdefault('query_error_mm', values['grid_query_max_error_mm'])
    for source, target in (('field_latent_variance_mm2','latent_field_rms_mm'), ('station_latent_variance_mm2','latent_station_rms_mm')):
        if source in values:
            value = float(values[source])
            values.setdefault(target, math.sqrt(value) if value >= 0 else float('nan'))
    failures = []
    for name in ('finite', 'label_isolation'):
        if values.get(name) is not True:
            failures.append(name)
    tests = {'query_error_mm': lambda x: 0 <= x <= .001,
             'latent_field_rms_mm': lambda x: x > 1e-4,
             'latent_station_rms_mm': lambda x: x > 1e-4,
             'prior_rmse_mm': lambda x: x >= 0,
             'background_rmse_mm': lambda x: x > 0}
    for name, rule in tests.items():
        try:
            value = float(values[name])
            if not math.isfinite(value) or not rule(value): failures.append(name)
        except (KeyError, TypeError, ValueError): failures.append(name)
    if not failures and values['prior_rmse_mm'] > 1.5*values['background_rmse_mm']:
        failures.append('catastrophic_prior_degradation')
    return dict(passed=not failures, failures=failures, evidence=values)


def _selected(results, stage, product='joint'):
    result = results.get(stage, {})
    if result.get('status') != 'complete':
        raise ValueError('Incomplete dependency: '+stage)
    try:
        selection = result['best'][product]
        path, digest = selection['checkpoint'], selection['sha256']
    except KeyError as exc:
        raise ValueError(f'Missing selected {product} candidate for {stage}') from exc
    if not Path(path).is_file() or sha256(path) != digest:
        raise ValueError('Selected candidate SHA256 mismatch: '+str(path))
    return str(path)


def resolve_dependencies(stage, results):
    if stage == 'control': return {}
    if stage == 'vae_representation':
        return {'representation': _selected(results, 'control', 'point')}
    dependencies = {'representation': _selected(results, 'vae_representation')}
    if stage == 'posterior_cache': return dependencies
    if stage.startswith(('gan_pretrain', 'diffusion_pretrain', 'shared_')):
        cache = results.get('posterior_cache', {})
        path = cache.get('cache')
        if cache.get('status') != 'complete' or not path or not cache.get('cache_sha256') or sha256(path) != cache['cache_sha256']:
            raise ValueError('Posterior cache missing or SHA256 changed')
        dependencies['posterior_cache'] = path
    if stage.startswith('shared_') and stage != 'shared_vae':
        dependencies['prior'] = _selected(results, stage[7:]+'_pretrain')
    if stage.startswith('complete_'):
        dependencies['prior'] = _selected(results, 'shared_'+stage[9:])
    return dependencies


def load_config(path):
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        config = read_json(path)
    if not isinstance(config, dict) or not all(isinstance(config.get(k), dict) for k in ('scientific', 'data', 'runtime')):
        raise ValueError('Config needs distinct scientific/data/runtime mappings')
    if any(RUNTIME_FIELDS.intersection(config.get(section,{})) for section in ('scientific','training')):
        raise ValueError('Scientific configuration contains runtime controls')
    if set(config['runtime']).intersection({'epochs', 'min_epochs', 'latent_dim', 'sigma_eff_mm', 'train_years', 'seed'}):
        raise ValueError('Runtime configuration contains scientific controls')
    if config['scientific'].get('max_train_hours') is not None or config['data'].get('max_train_hours') is not None:
        raise ValueError('Full campaign must use every qualified hour')
    return config


def background_code_binding(runtime, current_source):
    """Reuse a frozen code asset's identity only after verifying equal bytes."""
    current=Path(current_source)
    anchor=Path(runtime.get('background_identity_source',current))
    digest=sha256(current)
    if not anchor.is_file() or sha256(anchor)!=digest:
        raise ValueError('Frozen background source bytes changed across runtime releases')
    return {str(anchor):digest}


def build_data(config, *, purpose='train', allow_background_inference=False):
    """Keep frozen model construction lazy, and bill it only during explicit prep."""
    from .data import FullFieldData, VerifiedBackgroundProvider
    from .background import FusionBackgroundAssets, FrozenFusionBackground
    from gnss_field.data import PackedStore
    data_config = dict(config['data'])
    source = dict(data_config.pop('background_source'))
    assets = FusionBackgroundAssets(**source)
    assets.verify()
    store = PackedStore(data_config['packed_root'], purpose=purpose)
    lazy = []
    def callback(hours, frames):
        if not allow_background_inference:
            raise ValueError('Formal cache miss: explicit prepare required before training/evaluation')
        if not lazy:
            lazy.append(FrozenFusionBackground(store, assets, device=config['runtime'].get('device','cuda'),
                        patch_chunk=config['runtime'].get('background_patch_chunk',8)))
        return lazy[0](hours, frames)
    sources = {str(Path(getattr(assets, name))): source['expected_sha256'][str(Path(getattr(assets, name)))]
               for name in ('network_checkpoint','geometry_coefficients','fusion_settings')}
    sources.update(background_code_binding(config['runtime'],Path(__file__).with_name('background.py')))
    provider = VerifiedBackgroundProvider(callback, sources, identity={**assets.verify(), 'geometry_readout_index': assets.geometry_readout_index})
    data_config['background_assets'] = sources
    return FullFieldData(data_config, purpose=purpose, store=store, background_provider=provider)


def verified_job_accounting(job_ids, *, run=subprocess.run):
    ids = [str(i) for i in job_ids]
    if not ids: return dict(card_seconds=0, jobs={})
    if any(not i.isdigit() for i in ids): raise ValueError('Invalid tracked Slurm job id')
    query = ','.join(ids)
    active = run(['squeue','-h','-j',query,'-o','%i|%T'], check=True, capture_output=True, text=True)
    if active.stdout.strip(): raise ValueError('Tracked GPU job is still active; budget cannot be reused')
    rows = run(['sacct','-X','-n','-P','-j',query,'--format=JobIDRaw,State,Elapsed,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode'],
               check=True, capture_output=True, text=True)
    jobs = {}
    for line in rows.stdout.splitlines():
        columns = line.strip().split('|')
        if not line.strip(): continue
        if len(columns) < 7: raise ValueError('Incomplete sacct accounting')
        job, state, _, elapsed, _, tres, exit_code = columns[:7]
        if job not in ids: continue
        state = state.split()[0].rstrip('+')
        if state not in TERMINAL: raise ValueError('Accounting not terminal: '+job)
        # Require one and only one GPU allocation; never infer elapsed from CPU time.
        gpu = [x for x in tres.split(',') if x.startswith(('gres/gpu=', 'gres/gpu:a800=', 'gpu:'))]
        if not gpu or any(not (x.endswith('=1') or x == 'gpu:a800:1') for x in gpu):
            raise ValueError('Unexpected GPU allocation in accounting')
        seconds = _positive(elapsed, 'ElapsedRaw', True)
        jobs[job] = dict(state=state, elapsed_seconds=seconds, exit_code=exit_code)
    if set(jobs) != set(ids): raise ValueError('Missing terminal sacct records; budget cannot be reused')
    return dict(card_seconds=sum(v['elapsed_seconds'] for v in jobs.values()), jobs=jobs)


def allocation_cpus(stage, runtime, slurm):
    workers=runtime.get('preparation_workers',3)
    cpus=1+2*workers if stage=='prepare' else slurm.get('training_cpus',2)
    if not isinstance(cpus,int) or cpus<1 or cpus>slurm.get('max_cpus_per_gpu',7):
        raise ValueError('Requested processes exceed the site CPU limit per GPU')
    return cpus


def retry_allowed(status, scheduler_state, *, attempts, remaining_seconds):
    return (attempts < 3 and remaining_seconds > 0 and status != 'error' and
            ((status == 'paused' and scheduler_state == 'COMPLETED') or scheduler_state in ('NODE_FAIL', 'BOOT_FAIL')))


@contextmanager
def submit_lock(root):
    # The dispatcher runs on Linux; Windows planning remains importable.
    import fcntl
    path = Path(root)/'submit.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Campaign:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['runtime']['run_root']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'campaign.json'
        immutable_json(self.root/'scientific_config.json', {key:config.get(key,{}) for key in ('scientific','data','training','diagnostics','evaluation','benchmark')})
        self.manifest = read_json(self.path) if self.path.exists() else dict(schema=1, stages={}, jobs=[], intents=[])
        changed = False
        for job in config['runtime'].get('preflight_jobs',[]):
            if not str(job['job_id']).isdigit() or job['stage'] not in ('benchmark','prepare','diagnostics','preflight_prepare'):
                raise ValueError('Invalid explicit preflight job registration')
            if not any(row['job_id']==str(job['job_id']) for row in self.manifest['jobs']):
                self.manifest['jobs'].append({**job,'job_id':str(job['job_id'])})
                changed = True
        if changed: self.save()

    def save(self): write_json(self.path, self.manifest)

    def status(self): return self.manifest

    def require_training_gates(self, stage):
        budget_path = self.root/'budget.lock.json'
        if not budget_path.is_file(): raise ValueError('Measured budget lock is required before training')
        lock = read_json(budget_path)
        if lock['cap_seconds'] > 24*3600: raise ValueError('Invalid budget cap')
        benchmark = read_json(self.root/'benchmark.json')
        if fingerprint(benchmark) != lock['benchmark_sha256']: raise ValueError('Measured budget benchmark changed')
        protocol=self.config.get('evaluation',{}).get('final_selection_protocol','top3_union')
        if protocol not in ('top3_union','subset_locked_products'): raise ValueError('Unknown final selection protocol')
        final_bound=10 if protocol=='subset_locked_products' else 39
        if benchmark.get('max_final_candidates',40)<final_bound:
            raise ValueError('Measured final candidate budget does not cover the registered selection protocol')
        if stage in ('prepare','diagnostics'): return lock
        diagnostics = read_json(self.root/'diagnostics.json')
        if diagnostics.get('passed') is not True: raise ValueError('Training-only diagnostics did not pass')
        selection_path=self.root/'training_only_selection.lock.json'
        selection=read_json(selection_path) if selection_path.is_file() else diagnostics
        selected_latent=selection.get('latent_dim',self.config['scientific'].get('latent_dim',64))
        if selected_latent not in (64,128) or selected_latent>lock.get('priced_latent_dim',0):
            raise ValueError('Selected latent dimension exceeds measured budget pricing')
        if stage not in ('control', 'vae_representation'):
            gate = read_json(self.root/'representation_gate.json')
            if gate.get('passed') is not True: raise ValueError('Useful representation gate did not pass')
        return lock

    def audit(self, data=None):
        data = data or build_data(self.config)
        report = data.audit('B')
        immutable_json(self.root/'audit_B.json', report)
        development = data._other_view('development').audit('C')
        immutable_json(self.root/'audit_C.json', development)
        return report

    def lock_budget(self, benchmark=None, accounted_seconds=0):
        benchmark = benchmark or read_json(self.root/'benchmark.json')
        audit = read_json(self.root/'audit_B.json')
        lock = estimate_budget(audit['qualified_hours'], benchmark, cap_card_hours=24, accounted_seconds=accounted_seconds)
        immutable_json(self.root/'benchmark.json', benchmark)
        immutable_json(self.root/'budget.lock.json', lock)
        return lock

    def train(self, stage, data=None, *, chunk_seconds=None):
        from .training import FullTrainingConfig, run_stage
        lock = self.require_training_gates(stage)
        row = next((r for r in stage_catalogue() if r['id'] == stage and 'epochs' in r), None)
        if row is None: raise ValueError('Unknown training stage: '+stage)
        deps = resolve_dependencies(stage, self.manifest['stages'])
        output = self.root/'stages'/stage
        previous = self.manifest['stages'].get(stage, {})
        if previous.get('status') == 'complete': raise ValueError('Stage already complete')
        if previous.get('checkpoint'): deps['resume'] = previous['checkpoint']
        elapsed = self._stage_elapsed(stage)
        remaining = lock['stage_caps_seconds'][stage]-elapsed
        if remaining <= 0: raise ValueError('Stage measured budget exhausted')
        budget = min(remaining, chunk_seconds or self.config['runtime'].get('chunk_seconds',1800))
        science = {**self.config['scientific'], **self.config.get('training',{})}
        # Diagnostics options and evaluation panels are registered separately.
        for key in ('diagnostics', 'benchmark', 'evaluation'): science.pop(key, None)
        values = dict(science, stage=row['stage'], model_family=row['model_family'], epochs=row['epochs'],
             min_epochs=row['epochs'], batch_size=row['batch_size'], patience=0,
             data_config=self.config['data'], dependency_paths=deps, output_dir=str(output),
             device=self.config['runtime'].get('device','cuda'), budget_seconds=budget,
             reference_count=read_json(self.root/'audit_B.json')['reference_count'],
             kl_warmup_updates=lock['work'][stage]['kl_warmup_updates'])
        values.update(self.config['scientific'].get('evaluation', {}))
        values.update({k:v for k,v in self.config.get('evaluation', {}).items() if k!='final_selection_protocol'})
        values.pop('final_selection_protocol',None)
        values['development_hours']=registered_development_hours(self.config)
        selection_path = self.root/'training_only_selection.lock.json'
        if selection_path.is_file():
            selection = read_json(selection_path)
            values.update(sigma_eff_mm=selection['sigma_eff_mm'],latent_dim=selection['latent_dim'])
        training = FullTrainingConfig(**values)
        data = data or build_data(self.config)
        try:
            result = run_stage(training, data=data)
        except Exception:
            self.manifest['stages'][stage] = {**previous, 'status':'error', 'checkpoint':str(output/'last.pt')}
            self.save()
            raise
        if result.get('cache'): result['cache_sha256'] = sha256(result['cache'])
        self.manifest['stages'][stage] = result;self.save()
        if stage == 'vae_representation' and result['status'] == 'complete':
            from .diagnostics import check_representation
            evidence = check_representation(self.config, data, _selected(self.manifest['stages'], stage), self.root)
            gate = representation_gate(evidence)
            immutable_json(self.root/'representation_gate.json', gate)
            if not gate['passed']: raise ValueError('Useful representation gate failed')
        return result

    def run_auxiliary(self, stage, *, chunk_seconds=None):
        import copy
        import signal
        import time
        if stage not in ('prepare','diagnostics'): raise ValueError('Unknown auxiliary stage')
        lock=self.require_training_gates(stage)
        remaining=min(lock['stage_caps_seconds'][stage]-self._stage_elapsed(stage),
                      lock['cap_seconds']-self.manifest.get('accounted_card_seconds',0))
        duration=min(remaining,float(chunk_seconds) if chunk_seconds is not None else remaining)
        if not math.isfinite(duration) or duration<=0: raise ValueError('Auxiliary stage budget exhausted')
        deadline=time.monotonic()+duration
        def stopped(signum,frame): raise TimeoutError('Controlled auxiliary allocation pause')
        previous={}
        for name in ('SIGUSR1','SIGTERM'):
            sig=getattr(signal,name,None)
            if sig is not None:
                previous[sig]=signal.getsignal(sig);signal.signal(sig,stopped)
        try:
            if stage=='prepare':
                from .preparation import prepare_parallel
                audit_B=read_json(self.root/'audit_B.json');audit_C=read_json(self.root/'audit_C.json')
                result=prepare_parallel(self.config,train_hours=audit_B['eligible_hours'],
                    development_hours=audit_C['hour_ids'],workers=self.config['runtime'].get('preparation_workers',4),
                    timeout_seconds=max(.001,deadline-time.monotonic()))
                result={**result,'status':'complete'}
                write_json(self.root/'preparation.json',result)
            else:
                from .diagnostics import run_representation_diagnostics
                config=copy.deepcopy(self.config)
                config['runtime'].update(diagnostics_budget_seconds=lock['stage_caps_seconds'][stage],
                    diagnostics_remaining_seconds=max(.001,deadline-time.monotonic()),
                    diagnostics_deadline_unix=time.time()+max(0.,deadline-time.monotonic()))
                config['runtime'].setdefault('evaluation_hours_batch',config.get('evaluation',{}).get('evaluation_hours_batch',8))
                config['runtime'].setdefault('evaluation_chunk',config.get('evaluation',{}).get('evaluation_chunk',16))
                data=build_data(config)
                result=run_representation_diagnostics(config,data,self.root)
                result={**result,'passed':result.get('status')=='complete'}
                immutable_json(self.root/'diagnostics.json',result)
        except TimeoutError as exc:
            result=dict(status='paused',pause_reason=str(exc))
        except Exception:
            self.manifest['stages'][stage]={'status':'error'};self.save();raise
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
        self.manifest['stages'][stage]=result;self.save()
        return result

    def _stage_elapsed(self, stage):
        return sum(j.get('elapsed_seconds',0) for j in self.manifest['jobs'] if j['stage'] == stage)

    def refresh_accounting(self, *, run=subprocess.run):
        report = verified_job_accounting([j['job_id'] for j in self.manifest['jobs']],run=run)
        for row in self.manifest['jobs']:
            row.update(report['jobs'][row['job_id']])
        self.manifest['accounted_card_seconds'] = report['card_seconds'];self.save()
        preflight = sum(j.get('elapsed_seconds',0) for j in self.manifest['jobs'] if j['stage'] in ('benchmark','benchmark_recheck'))
        if preflight > 3600: raise ValueError('Preflight one card-hour cap exceeded')
        return report

    def _next_stage(self):
        prefix=[{'id':'prepare'},{'id':'diagnostics'}] if self.config['runtime'].get('staged_preparation',False) else []
        for row in prefix+stage_catalogue():
            if self.manifest['stages'].get(row['id'],{}).get('status') != 'complete': return row['id']
        return None

    def submit(self, config_path, *, run=subprocess.run):
        """Just in time one-GPU submission, followed by a two-CPU afterany checker.

        An ambiguous sbatch result leaves its durable intent unresolved. Operators
        must reconcile the unique token with Slurm rather than submit duplicates.
        """
        with submit_lock(self.root):
            # Reload after obtaining the interprocess lock, never trust stale state.
            if self.path.exists(): self.manifest = read_json(self.path)
            if (self.root/'pause.request').exists():
                self.manifest['control_status']='paused_by_operator';self.save()
                return dict(status='paused_by_operator',next_stage=self._next_stage())
            accounting = self.refresh_accounting(run=run)
            if any(not i.get('job_id') for i in self.manifest['intents']):
                raise ValueError('Unresolved submission intent; reconcile token with Slurm first')
            stage = self._next_stage()
            if stage is None:
                if self.manifest.get('results_export_job_id'):
                    return dict(status='complete',results_export_job_id=self.manifest['results_export_job_id'])
                if self.manifest.get('results_export_intent'):
                    raise ValueError('Unresolved report submission intent; reconcile before resubmitting')
                runtime=self.config['runtime'];project=Path(runtime.get('project_root',Path.cwd())).resolve()
                token=uuid.uuid4().hex
                self.manifest['results_export_intent']=token;self.save()
                response=run(['sbatch','--parsable','--partition',self.config.get('slurm',{}).get('cpu_partition','intel'),
                    '--job-name','fg-results-'+token[:8],'--cpus-per-task=2','--mem=16G','--time=60','--no-requeue',
                    '--chdir',str(project),'--output',str(self.root/'results-%j.out'),'--error',str(self.root/'results-%j.err'),
                    '--wrap','exec '+shlex.join([runtime.get('python',sys.executable),
                         str(project/'tools/export_full_generative_results.py'),'--config',str(Path(config_path).resolve())])],
                    check=True,capture_output=True,text=True)
                job=response.stdout.strip().split(';')[0]
                if not job.isdigit():raise ValueError('Ambiguous report sbatch response; preserve intent')
                self.manifest['results_export_job_id']=job;self.save()
                return dict(status='complete',results_export_job_id=job)
            lock = self.require_training_gates(stage)
            if stage not in ('prepare','diagnostics','final_evaluation'): resolve_dependencies(stage, self.manifest['stages'])
            remaining_total = lock['cap_seconds']-accounting['card_seconds']
            remaining_stage = lock['stage_caps_seconds'][stage]-self._stage_elapsed(stage)
            attempts = [j for j in self.manifest['jobs'] if j['stage'] == stage]
            if attempts:
                previous = self.manifest['stages'].get(stage,{}).get('status','running')
                if not retry_allowed(previous, attempts[-1]['state'], attempts=1+sum(j['state'] in ('NODE_FAIL','BOOT_FAIL') for j in attempts), remaining_seconds=min(remaining_stage,remaining_total)):
                    raise ValueError('Automatic retry is not authorized for this status or exhausted budget')
            requested=remaining_stage if stage in ('prepare','diagnostics') else float(self.config['runtime'].get('chunk_seconds',1800))
            chunk = 60*math.floor(min(requested, remaining_total, remaining_stage)/60)
            if chunk < 60: raise ValueError('Insufficient remaining stage/total budget for a safe chunk')
            grace=600 if stage=='prepare' else min(180,max(20,math.ceil(.1*chunk)))
            if chunk-grace<60 and stage=='prepare': raise ValueError('Preparation budget cannot retain its600-second atomic month completion grace')
            work_seconds=chunk-grace
            token = uuid.uuid4().hex
            intent = dict(token=token, stage=stage, allocated_seconds=math.ceil(chunk), job_id=None)
            self.manifest['intents'].append(intent);self.save()
            immutable_json(self.root/'intents'/f'{token}.json', intent)
            runtime = self.config['runtime']
            project = Path(runtime.get('project_root',Path.cwd())).resolve()
            python = runtime.get('python',sys.executable)
            command = [python,str(project/'tools/run_full_generative_campaign.py'), '--config', str(Path(config_path).resolve())]
            if stage in ('prepare','diagnostics'): execution=command+[stage,'--chunk-seconds',str(work_seconds)]
            elif stage=='final_evaluation': execution=command+['evaluate','--chunk-seconds',str(work_seconds)]
            else: execution=command+['train','--stage',stage,'--chunk-seconds',str(work_seconds)]
            # GPU allocation includes all evaluation/checkpoint work until process exit.
            args = ['sbatch','--parsable','--job-name',f'fg-{stage}-{token[:8]}','--partition',self.config.get('slurm',{}).get('gpu_partition','GPUA800'),
                    '--gres=gpu:a800:1',f"--cpus-per-task={allocation_cpus(stage,runtime,self.config.get('slurm',{}))}",'--mem=48G','--no-requeue',f'--signal=B:USR1@{grace}',
                    '--time',str(math.ceil(chunk/60)), '--chdir',str(project),
                    '--output',str(self.root/f'{stage}-%j.out'), '--error',str(self.root/f'{stage}-%j.err'),
                    '--export=ALL,OMP_NUM_THREADS=2,MKL_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2',
                    '--wrap','exec '+shlex.join(execution)]
            response = run(args, check=True, capture_output=True, text=True)
            job = response.stdout.strip().split(';')[0]
            if not job.isdigit(): raise ValueError('Ambiguous sbatch response; preserve submission intent')
            intent['job_id'] = job
            self.manifest['jobs'].append(dict(job_id=job,stage=stage,token=token,allocated_seconds=math.ceil(chunk)))
            self.save()
            dispatcher = run(['sbatch','--parsable','--job-name',f'fg-dispatch-{token[:8]}',
                 '--partition',self.config.get('slurm',{}).get('cpu_partition','intel'),
                 '--dependency',f'afterany:{job}','--cpus-per-task=2','--mem=4G','--no-requeue','--time=10',
                 '--chdir',str(project),'--output',str(self.root/'dispatch-%j.out'),
                 '--wrap',shlex.join(command+['dispatch'])], check=True,capture_output=True,text=True)
            self.manifest['jobs'][-1]['dispatcher_job_id'] = dispatcher.stdout.strip().split(';')[0]
            self.save()
            return dict(job_id=job, stage=stage, token=token)


def final_candidates(results, stages=None, *, protocol='top3_union'):
    if protocol not in ('top3_union','subset_locked_products'): raise ValueError('Unknown final selection protocol')
    if protocol=='subset_locked_products':
        stages=stages or ['control']+[f'{prefix}_{family}' for prefix in ('shared','complete') for family in FAMILIES]
        registry={}
        for stage in stages:
            result=results.get(stage,{})
            if result.get('selection_status')=='baseline_only_no_formal_candidates':
                raise ValueError('Formal subset-selected products are required: '+stage)
            products=('point',) if stage=='control' else ('joint',) if stage.startswith('shared_') else ('point','joint')
            rows=[]
            for product in products:
                _selected(results,stage,product)
                selected=result['best'][product]
                duplicate=next((r for r in rows if r['sha256']==selected['sha256']),None)
                if duplicate is None:
                    rows.append({**selected,'frozen_products':[product],'selection_basis':'fixed_C2019_576_subset'})
                else: duplicate['frozen_products'].append(product)
            registry[stage]=rows
        return registry
    stages = stages or ['control','vae_representation']+[f'{prefix}_{family}' for prefix in ('shared','complete') for family in FAMILIES]
    selected = {}
    for stage in stages:
        result = results.get(stage,{})
        if result.get('status') != 'complete': raise ValueError('Incomplete final candidate stage: '+stage)
        products = ('point',) if stage=='control' else ('point','joint')
        rows=[]
        for product in products:
            if stage in ('control','vae_representation'):
                entries = [result.get('best',{}).get(product)]
            else:
                entries = result.get('candidates',{}).get(product)
                if not entries: raise ValueError('Missing registered top-three candidates: '+stage)
                if len(entries)>3: raise ValueError('Top-three candidate registry exceeded')
            for row in entries:
                if not row or not Path(row['checkpoint']).is_file() or sha256(row['checkpoint'])!=row['sha256']:
                    raise ValueError('Final candidate SHA256 mismatch or missing: '+stage)
                if not any(v['sha256']==row['sha256'] for v in rows): rows.append(row)
        selected[stage]=rows
    return selected


def diagnostic_preflight_config(campaign):
    """Reject unsafe work before constructing a pilot or any diagnostic CUDA model."""
    import copy
    import time
    benchmark=read_json(campaign.root/'benchmark.json')
    audit=read_json(campaign.root/'audit_B.json')
    planned=estimate_budget(audit['qualified_hours'],benchmark,cap_card_hours=24,
        accounted_seconds=campaign.manifest.get('accounted_card_seconds',0))
    if planned['stage_caps_seconds']['diagnostics']+planned['stage_caps_seconds']['prepare']+benchmark['benchmark_elapsed_seconds']>3600:
        raise ValueError('Preflight legacy diagnostics require a staged full campaign')
    clock=read_json(campaign.root/'preflight_clock.lock.json')
    if clock.get('cap_seconds')!=3600: raise ValueError('Preflight global clock cap differs')
    elapsed=_positive(time.time()-float(clock['started_unix']),'Preflight elapsed',True)
    remaining=3600-max(elapsed,float(benchmark['benchmark_elapsed_seconds']),
                       campaign.manifest.get('accounted_card_seconds',0))
    if planned['stage_caps_seconds']['diagnostics']>remaining:
        raise ValueError('Preflight remaining global budget cannot cover registered diagnostics')
    config=copy.deepcopy(campaign.config)
    config['runtime'].update(diagnostics_deadline_unix=float(clock['started_unix'])+3600,
        diagnostics_remaining_seconds=min(remaining,planned['stage_caps_seconds']['diagnostics']))
    return config


def verify_development_release(campaign, data):
    """Check label values as well as masks before any final-product cache reuse."""
    locked=read_json(campaign.root/'audit_C.json')
    current=data.audit('C')
    for key in ('labels_identity','condition_identity','hour_ids'):
        if key in locked and current.get(key)!=locked[key]:
            raise ValueError('C development release changed: '+key)
    return locked


def evaluation_candidate_config(state, options, device, hours, *, deadline):
    """Bind every candidate to the remaining time in its current GPU allocation."""
    import time
    from .training import FullTrainingConfig
    from .evaluation import EvaluationPaused
    remaining=deadline-time.monotonic()
    if remaining<=0: raise EvaluationPaused('Current final-evaluation allocation budget exhausted')
    options={k:v for k,v in options.items() if k!='final_selection_protocol'}
    return FullTrainingConfig(**{**state['config'],**options,'development_hours':hours,
        'eval_members':128,'point_members':max(128,int(options.get('point_members',128))),
        'device':device,'budget_seconds':remaining})


def final_product_reports(rows, *, protocol='top3_union'):
    metrics=(('point','station_macro_rmse_mm'),('joint','group_ES_mm'))
    products={}
    if protocol=='subset_locked_products':
        metric_names=dict(metrics)
        for row in rows:
            for product in row['registered_selection']['frozen_products']:
                if product in products: raise ValueError('Duplicate frozen final product: '+product)
                metric=metric_names[product]
                products[product]=dict(checkpoint=row['checkpoint'],sha256=row['sha256'],
                    selection_basis='fixed_C2019_576_subset',full_year_selection=False,
                    subset_selection_value=row['registered_selection']['value'],
                    value_K128=row['metrics_K128'][metric],value_K64=row['metrics_K64'][metric],
                    member_count_sensitivity=row['metrics_K128'][metric]-row['metrics_K64'][metric])
        return products
    for product,metric in metrics:
        winner64=min(rows,key=lambda r:r['metrics_K64'][metric])
        winner128=min(rows,key=lambda r:r['metrics_K128'][metric])
        products[product]=dict(checkpoint=winner128['checkpoint'],sha256=winner128['sha256'],
            value_K128=winner128['metrics_K128'][metric],winner_K64=winner64['checkpoint'],
            stable_K64_K128=winner64['sha256']==winner128['sha256'])
    return products


def evaluate_campaign(campaign, data=None, *, chunk_seconds=None):
    """Score registered products on the complete C2019 calendar.

    The subset-locked protocol freezes ten products before annual evaluation.
    K64/K128 differences measure sampling sensitivity at each same checkpoint.
    The legacy top-three protocol remains available to its original run configs.
    """
    from .evaluation import evaluate_model, EvaluationPaused
    from .training import FullTrainingConfig, load_stage_model
    from grid_generation_metrics import score_ensemble
    import numpy as np
    import time
    started=time.monotonic()
    lock=campaign.require_training_gates('final_evaluation')
    stage_remaining=lock['stage_caps_seconds']['final_evaluation']-campaign._stage_elapsed('final_evaluation')
    duration=min(stage_remaining,float(chunk_seconds if chunk_seconds is not None else campaign.config['runtime'].get('chunk_seconds',1800)))
    if not math.isfinite(duration) or duration<=0: raise ValueError('Final evaluation has no remaining chunk budget')
    deadline=started+duration
    data = data or build_data(campaign.config, purpose='development')
    audit = verify_development_release(campaign,data)
    hours = audit['hour_ids']
    if len(hours)!=8760: raise ValueError('Full C2019 evaluation requires all8760 calendar hours')
    reports = {}
    protocol=campaign.config.get('evaluation',{}).get('final_selection_protocol','top3_union')
    candidates = final_candidates(campaign.manifest['stages'],protocol=protocol)
    immutable_json(campaign.root/'final_candidate_union.lock.json',candidates)
    options = {**campaign.config['scientific'].get('evaluation',{}), **campaign.config.get('evaluation',{})}
    if not options.get('evaluation_population_manifest'): raise ValueError('Locked C2019 masks required for final reranking')
    try:
        for stage, selections in candidates.items():
            rows=[]
            for selection in selections:
                path=selection['checkpoint'];output=campaign.root/'evaluation'/stage/selection['sha256'][:20]
                complete=output/'candidate_complete.json'
                if complete.is_file():
                    row=read_json(complete)
                    if row['sha256']!=sha256(path): raise ValueError('Completed candidate hash changed')
                    if row.get('C_release_sha256')!=fingerprint(audit): raise ValueError('Completed candidate C release changed')
                    rows.append(row);continue
                model,state=load_stage_model(path,campaign.config['runtime'].get('device','cuda'))
                cfg=evaluation_candidate_config(state,options,campaign.config['runtime'].get('device','cuda'),hours,deadline=deadline)
                metrics=evaluate_model(model,data,hours,cfg,output=output)
                with np.load(output/'station_predictions.npz',allow_pickle=False) as archive:
                    valid=archive['population'].any(1)
                    prefix=score_ensemble(archive['truth'][valid],archive['samples'][valid,:64],
                                         archive['mean'][valid],archive['population'][valid],archive['xy'])
                row=dict(checkpoint=path,sha256=sha256(path),C_release_sha256=fingerprint(audit),registered_selection=selection,
                         metrics_K128=metrics,metrics_K64=prefix,
                         iid_prefix=True,point_pool_members=cfg.point_members)
                write_json(complete,row);rows.append(row)
            winners=final_product_reports(rows,protocol=protocol)
            reports[stage]=dict(candidates=rows,products=winners)
    except EvaluationPaused:
        result=dict(status='paused',reports=reports)
        campaign.manifest['stages']['final_evaluation']=result;campaign.save()
        return result
    result=dict(status='complete',role='C',year=2019,population_sha256=fingerprint(audit),
        max_candidates=10 if protocol=='subset_locked_products' else 39,
        final_selection_protocol=protocol,full_year_selection=protocol!='subset_locked_products',
        full_calendar_hours=8760,probability_members=128,prefix_members=64,
        frozen_product_count=(sum(len(row['frozen_products']) for entries in candidates.values() for row in entries) if protocol=='subset_locked_products' else None),
        reports=reports)
    write_json(campaign.root/'evaluation.json',result)
    campaign.manifest['stages']['final_evaluation']=result;campaign.save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    commands = parser.add_subparsers(dest='command',required=True)
    plan = commands.add_parser('plan');plan.add_argument('--qualified-hours',type=int)
    for command in ('audit','benchmark','status','submit','dispatch'):
        commands.add_parser(command)
    prepare=commands.add_parser('prepare');prepare.add_argument('--chunk-seconds',type=float)
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--chunk-seconds',type=float)
    diagnostics = commands.add_parser('diagnostics');diagnostics.add_argument('--checkpoint');diagnostics.add_argument('--chunk-seconds',type=float)
    train = commands.add_parser('train');train.add_argument('--stage',required=True)
    train.add_argument('--chunk-seconds',type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    campaign = Campaign(config)
    if args.command == 'plan':
        hours = args.qualified_hours
        if hours is None:
            audit_path = campaign.root/'audit_B.json'
            if not audit_path.exists(): raise ValueError('CPU audit required before measured full-hour plan')
            hours = read_json(audit_path)['qualified_hours']
        result = dict(stages=stage_catalogue(),work=stage_work(hours),
                      preflight_card_hour_cap=1,total_card_hour_cap=24,
                      needs_measured_benchmark=True)
    elif args.command == 'status': result = campaign.status()
    elif args.command == 'audit': result = campaign.audit()
    elif args.command == 'prepare' and config['runtime'].get('staged_preparation',False):
        result=campaign.run_auxiliary('prepare',chunk_seconds=args.chunk_seconds)
    elif args.command == 'prepare':
        data = build_data(config,allow_background_inference=True)
        development=data._other_view('development')
        result = dict(B=data.prepare(role='B'),C=development.prepare(development.audit('C')['hour_ids'],role='C'))
        write_json(campaign.root/'preparation.json',result)
    elif args.command == 'benchmark':
        import time
        clock_path=campaign.root/'preflight_clock.lock.json'
        if not clock_path.exists():
            immutable_json(clock_path,dict(started_unix=time.time(),cap_seconds=3600,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        from .benchmark import run_benchmark
        import copy
        benchmark_config = copy.deepcopy(config)
        benchmark_config['data']['cache_root'] = str(campaign.root/'benchmark_cache')
        data = build_data(benchmark_config,allow_background_inference=True)
        result = run_benchmark(config,data,campaign.root/'benchmark.json')
        immutable_json(campaign.root/'benchmark.json',result)
    elif args.command == 'diagnostics' and config['runtime'].get('staged_preparation',False) and not args.checkpoint:
        result=campaign.run_auxiliary('diagnostics',chunk_seconds=args.chunk_seconds)
    elif args.command == 'diagnostics':
        if not args.checkpoint: config=diagnostic_preflight_config(campaign)
        from .diagnostics import run_representation_diagnostics, check_representation
        data = build_data(config)
        if args.checkpoint:
            result = representation_gate(check_representation(config,data,args.checkpoint,campaign.root))
            immutable_json(campaign.root/'representation_gate.json',result)
        else:
            result = run_representation_diagnostics(config,data,campaign.root)
            result = {**result,'passed':result.get('status')=='complete'}
            immutable_json(campaign.root/'diagnostics.json',result)
            if result.get('passed') is True:
                campaign.lock_budget(accounted_seconds=campaign.manifest.get('accounted_card_seconds',0))
    elif args.command == 'train': result = campaign.train(args.stage,chunk_seconds=args.chunk_seconds)
    elif args.command == 'evaluate': result = evaluate_campaign(campaign,chunk_seconds=args.chunk_seconds)
    else: result = campaign.submit(args.config)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x)))
    return 0
