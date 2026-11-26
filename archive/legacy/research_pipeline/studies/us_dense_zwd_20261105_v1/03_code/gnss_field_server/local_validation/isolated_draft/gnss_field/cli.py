"""Explicit local/Slurm commands; read-only commands never start training."""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys

from gnss_field.artifacts import (fingerprint, immutable_json, read_json, release_digest,
                                  sha256, write_json)
from gnss_field.contracts import ScienceConfig, RuntimeConfig, load_config


def protocol_path():
    return Path(__file__).parent/'resources/evaluation_protocol_v2.json'


def _reservations(path):
    if path is None:
        return {}
    values = read_json(path)
    if not isinstance(values, dict) or not values:
        raise ValueError('Reservations must be a nonempty task/stage to card-hours mapping')
    try:
        if any(not isinstance(k, str) or not math.isfinite(float(v)) or float(v) <= 0
               for k, v in values.items()):
            raise ValueError('Each GPU reservation must be positive finite card-hours')
        return {k: float(v) for k, v in values.items()}
    except (TypeError, ValueError) as e:
        raise ValueError('Each GPU reservation must be positive finite card-hours') from e


def settings(args):
    if getattr(args, 'config', None):
        path = Path(args.config).resolve()
        science, runtime, campaign = load_config(path)
        project = Path(campaign.get('project_root', path.parent.parent)).resolve()
    else:
        science, runtime, campaign = ScienceConfig(), RuntimeConfig(), {}
        project = Path.cwd().resolve()
    values = asdict(runtime)
    for key in ('data_root', 'run_root', 'cache_root'):
        value = getattr(args, key, None) or values[key]
        p = Path(os.path.expandvars(value)).expanduser()
        values[key] = str(p.resolve() if p.is_absolute() else (project/p).resolve())
    if getattr(args, 'device', None):
        values['device'] = args.device
    campaign['project_root'] = str(project)
    return science, RuntimeConfig(**values), campaign


def _campaign(args, science, runtime, extra):
    from gnss_field.data import PackedStore
    from gnss_field.orchestration import Campaign, Task, Coordinator, build_default_campaign, write_campaign
    if args.action == 'plan':
        store = PackedStore(runtime.data_root)
        hashes = dict(plan=fingerprint(science), data=store.manifest['fingerprint'],
                      code=release_digest(extra['project_root']), protocol=sha256(protocol_path()))
        name = args.name or extra.get('name', 'tongji_field_v2')
        costs = _reservations(args.reservations)
        campaign = build_default_campaign(name, hashes, card_hours=costs)
        if args.scope == 'bootstrap':
            tasks = tuple(t for t in campaign.tasks if t.phase <= 3)
            campaign = Campaign(name, tasks, hashes)
        elif args.scope == 'research':
            if not costs:
                raise ValueError('Research planning requires explicit benchmark-informed --reservations')
            if not args.benchmark:
                raise ValueError('Research planning needs a completed Tongji benchmark report')
            measured = read_json(args.benchmark)
            if not measured.get('verified_for_tongji_budget') or measured.get('hashes') != hashes:
                raise ValueError('Local/unmatched benchmark cannot unlock a research campaign')
            kept = {t.task_id for t in campaign.tasks if t.phase >= 5}
            tasks = tuple(replace(t, phase=t.phase-5,
                                  dependencies=tuple(d for d in t.dependencies if d in kept))
                          for t in campaign.tasks if t.task_id in kept)
            campaign = Campaign(name, tasks, hashes)
            if any(t.gpu and t.task_id not in costs and t.stage not in costs for t in tasks):
                raise ValueError('Reservations must cover every GPU task/stage in the research campaign')
        root = Path(args.directory or Path(runtime.run_root)/name)
        resolved = dict(science=asdict(science), runtime=asdict(runtime), campaign=extra)
        immutable_json(root/'resolved.json', resolved)
        write_campaign(root, campaign, runtime, environment=extra.get('environment', 'gnss-field'),
                       working_directory=extra['project_root'])
        if args.benchmark:
            immutable_json(root/'benchmark_parent.json', read_json(args.benchmark))
        if costs:
            immutable_json(root/'reservations.json', costs)
        summary = dict(directory=str(root.resolve()), scope=args.scope, tasks=len(campaign.tasks),
                       dry_run=True, hashes=hashes,
                       worst_case_card_hours=sum(t.card_hours*t.max_attempts for t in campaign.tasks))
        write_json(root/'plan_summary.json', summary)
        return summary
    if not args.directory:
        raise ValueError('--directory is required for an existing campaign')
    coordinator = Coordinator.from_directory(args.directory)
    if args.action == 'status':
        return coordinator.status(query_slurm=args.refresh, write=False)
    if args.action == 'pause':
        return coordinator.pause() if args.execute else {'dry_run': True, 'action': 'pause'}
    if args.action == 'lock-budget':
        if not args.benchmark or args.total_card_hours is None:
            raise ValueError('Budget lock requires a Tongji benchmark and an explicit total card-hour limit')
        benchmark = read_json(args.benchmark)
        if not benchmark.get('verified_for_tongji_budget') or benchmark.get('hashes') != dict(coordinator.campaign.hashes):
            raise ValueError('A real matching Tongji benchmark is required')
        stages = {}
        for task in coordinator.campaign.tasks:
            stages[task.stage] = stages.get(task.stage, 0.)+task.card_hours*task.max_attempts
        if sum(stages.values()) > args.total_card_hours:
            raise ValueError('Declared worst-case attempts exceed requested total; replan stage reservations first')
        lock = dict(locked=True, hashes=dict(coordinator.campaign.hashes), total_card_hours=args.total_card_hours,
                    stage_card_hours=stages, benchmark_sha256=sha256(args.benchmark), gpu_unit_CNY=2,
                    cpu_charge='unconfirmed_separately_reported', storage_charge='account_quota_preflight')
        immutable_json(Path(args.directory)/'budget.lock.json', lock)
        return lock
    unlock = dict(execute=args.execute,
                  preflight=read_json(args.preflight) if args.preflight else None,
                  budget_lock=read_json(args.budget_lock) if args.budget_lock else None,
                  benchmark_cap=args.benchmark_cap)
    if args.action == 'submit':
        if args.phase is None:
            raise ValueError('Select an explicit --phase')
        return coordinator.submit_phase(args.phase, **unlock)
    if args.action == 'resume':
        if args.phase is None:
            raise ValueError('Resume requires the explicit interrupted --phase')
        return coordinator.resume(args.phase, **unlock)
    raise ValueError('Unknown campaign action')


def _worker(args):
    from gnss_field.orchestration import Task, set_task_state, commit_task_result, verified_receipt
    from gnss_field.training.engine import PauseRequested
    task_file = Path(args.task_file).resolve()
    root = task_file.parent.parent
    spec = read_json(task_file)
    task = Task(**spec['tasks'][args.index])
    config_args = argparse.Namespace(config=str(root/'resolved.json'), data_root=None, run_root=None,
                                     cache_root=None, device=None)
    science, runtime, extra = settings(config_args)
    from gnss_field.data import PackedStore
    packed = PackedStore(runtime.data_root)
    if not packed.manifest['registered']:
        raise ValueError('Slurm observational campaign cannot use a synthetic engineering pack')
    actual = dict(plan=fingerprint(science), data=packed.manifest['fingerprint'],
                  code=release_digest(extra['project_root']), protocol=sha256(protocol_path()))
    if actual != spec['hashes']:
        raise ValueError('Worker data/code/protocol/scientific configuration differs from planned release')
    if (root/'PAUSE').exists():
        set_task_state(root, task, 'PAUSED')
        return 75
    parent_tasks = read_json(root/'campaign.json')['tasks']
    lookup = {t['task_id']: Task(**t) for t in parent_tasks}
    for dep in task.dependencies:
        verified_receipt(root, lookup[dep])
    if not task.gpu:
        runtime = replace(runtime, device='cpu')
    set_task_state(root, task, 'RUNNING')
    try:
        alias = task.argv[0]
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--arm')
        parser.add_argument('--fold', type=int)
        parser.add_argument('--variant')
        parser.add_argument('--seed', type=int, default=42)
        kw = vars(parser.parse_args(task.argv[1:]))
        if alias in ('pack', 'audit'):
            result = _data_command(alias, runtime, extra, root/'task_reports'/alias)
            outputs = {'report': result['report']}
        elif alias == 'doctor':
            from gnss_field.runtime import doctor
            path = root/'task_reports/doctor.json'
            doctor(runtime, live_tongji=True, hashes=spec['hashes'], output=path)
            outputs = {'report': path}
        elif alias == 'benchmark':
            from gnss_field.runtime import benchmark
            report = benchmark(science, runtime, root/'benchmark', hashes=spec['hashes'],
                               static_preflight=read_json(root/'preflight.json'), max_card_hours=task.card_hours)
            write_json(root/'preflight.json', read_json(root/'benchmark/verified_preflight.json'))
            outputs = {'benchmark': root/'benchmark/benchmark.json'}
        elif alias == 'budget_gate':
            lock = root/'budget.lock.json'
            if not lock.exists() or not read_json(lock).get('locked'):
                raise ValueError('Formal training budget is not locked')
            outputs = {'budget': lock}
        elif alias == 'report':
            from gnss_field.reporting import build_report
            outputs = {'report': build_report(root)}
        else:
            from gnss_field.training.pipeline import TrainingPipeline
            selected_seed = kw.pop('seed')
            pipeline = TrainingPipeline(science, runtime, root, seed=42 if alias == 'replicate' else selected_seed,
                                        legacy_source=extra.get('legacy_source'))
            if alias == 'replicate':
                kw['seed'] = selected_seed
            outputs = pipeline.run_task(alias, **{k: v for k, v in kw.items() if v is not None})
        commit_task_result(root, task, outputs)
        return 0
    except PauseRequested:
        set_task_state(root, task, 'PAUSED')
        return 75
    except Exception as exc:
        state = 'STOPPED_NO_GAIN' if 'STOPPED_NO_GAIN' in str(exc) else 'FAILED'
        set_task_state(root, task, state, detail=str(exc))
        raise


def _data_command(command, runtime, extra, output):
    from gnss_field.data import PackedStore, pack_legacy
    if command == 'pack' and not (Path(runtime.data_root)/'manifest.json').exists():
        if not extra.get('legacy_source'):
            raise ValueError('No packed data and no explicit legacy source provided')
        source = Path(os.path.expandvars(extra['legacy_source'])).expanduser()
        if not source.is_absolute():
            source = Path(extra['project_root'])/source
        pack_legacy(source, runtime.data_root)
    store = PackedStore(runtime.data_root)
    report = store.verify_all()
    report.update(counts=store.manifest['counts'], units=store.manifest['label_units'],
                  training_started=False, remote_jobs_submitted=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'data_audit.json'
    write_json(path, report)
    return {'report': str(path.resolve()), **report}


def parser():
    p = argparse.ArgumentParser(prog='gnss-field', description='Native PyTorch / Tongji Slurm GNSS field research')
    for name in ('config', 'data-root', 'run-root', 'cache-root', 'device'):
        p.add_argument('--'+name)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('pack', 'audit', 'doctor', 'smoke', 'benchmark', 'report', 'monitor'):
        q = sub.add_parser(name)
        q.add_argument('--output')
        if name == 'doctor':
            q.add_argument('--site', choices=['local', 'Tongji'], default='local')
            q.add_argument('--campaign-dir')
        if name in ('smoke', 'benchmark'):
            q.add_argument('--resume-updates', type=int, default=0)
            q.add_argument('--campaign-dir')
        if name == 'monitor':
            q.add_argument('--port', type=int, default=8767)
            q.add_argument('--ssh-host')
            q.add_argument('--remote-root')
    for name in ('train', 'cache', 'evaluate', 'export'):
        q = sub.add_parser(name)
        q.add_argument('--stage')
        q.add_argument('--directory', required=True)
        q.add_argument('--arm')
        q.add_argument('--fold', type=int)
        q.add_argument('--variant')
        q.add_argument('--seed', type=int, default=42)
    q = sub.add_parser('campaign')
    q.add_argument('action', choices=['plan', 'status', 'submit', 'pause', 'resume', 'lock-budget'])
    q.add_argument('--directory')
    q.add_argument('--name')
    q.add_argument('--scope', choices=['blueprint', 'bootstrap', 'research'], default='blueprint')
    q.add_argument('--phase', type=int)
    q.add_argument('--execute', action='store_true')
    q.add_argument('--refresh', action='store_true')
    q.add_argument('--preflight')
    q.add_argument('--budget-lock')
    q.add_argument('--benchmark')
    q.add_argument('--reservations')
    q.add_argument('--benchmark-cap', type=float)
    q.add_argument('--total-card-hours', type=float)
    q = sub.add_parser('worker')
    q.add_argument('--task-file', required=True)
    q.add_argument('--index', required=True, type=int)
    return p


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    try:
        args = parser().parse_args(argv)
        if args.command == 'worker':
            return _worker(args)
        science, runtime, extra = settings(args)
        if args.command in ('pack', 'audit'):
            result = _data_command(args.command, runtime, extra, args.output or 'local_validation/data')
        elif args.command == 'doctor':
            from gnss_field.runtime import doctor
            hashes = read_json(Path(args.campaign_dir)/'campaign.json')['hashes'] if args.campaign_dir else None
            result = doctor(runtime, live_tongji=args.site == 'Tongji', hashes=hashes,
                            output=args.output or 'local_validation/doctor.json')
        elif args.command == 'smoke':
            from gnss_field.runtime import engineering_smoke
            result = engineering_smoke(science, runtime, args.output or 'local_validation/smoke', resume_updates=args.resume_updates)
        elif args.command == 'benchmark':
            from gnss_field.runtime import benchmark
            if not args.campaign_dir:
                raise ValueError('Use a hash-bound bootstrap campaign for the Tongji benchmark')
            root = Path(args.campaign_dir)
            result = benchmark(science, runtime, args.output or root/'benchmark',
                               hashes=read_json(root/'campaign.json')['hashes'], static_preflight=read_json(root/'preflight.json'))
        elif args.command == 'campaign':
            result = _campaign(args, science, runtime, extra)
        elif args.command == 'report':
            from gnss_field.reporting import build_report
            result = {'report': str(build_report(args.output or runtime.run_root))}
        elif args.command == 'monitor':
            from gnss_field.monitor import serve
            serve(args.output or runtime.run_root, port=args.port,
                  ssh_host=args.ssh_host, remote_root=args.remote_root)
            return 0
        else:
            from gnss_field.training.pipeline import TrainingPipeline
            pipeline = TrainingPipeline(science, runtime, args.directory, args.seed, legacy_source=extra.get('legacy_source'))
            aliases = {'cache': 'mean_cache', 'evaluate': 'full2019_rerank', 'export': 'export'}
            stage = args.stage or aliases.get(args.command)
            if stage is None:
                raise ValueError('Train requires an explicit --stage')
            result = pipeline.run_task(stage, **{k: getattr(args, k) for k in ('arm', 'fold', 'variant') if getattr(args, k) is not None})
        print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 75
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
