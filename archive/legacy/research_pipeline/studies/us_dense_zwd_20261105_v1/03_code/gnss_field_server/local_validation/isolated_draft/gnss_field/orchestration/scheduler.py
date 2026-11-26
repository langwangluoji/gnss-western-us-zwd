"""Explicit Slurm submission with single-owner manifests and conservative reservations."""
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
from collections import defaultdict
import math
import os
import re
import shlex
import socket
import subprocess

from ..artifacts import atomic_bytes, fingerprint, immutable_json, read_json, write_json
from ..contracts import RuntimeConfig
from .budget import validate_budget, validate_preflight
from .campaign import Campaign, Task, safe_name, write_task_file
from .state import set_task_state, single_owner, verified_receipt

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def _state(value):
    return str(value).strip().split()[0].split('+')[0].upper()


class SlurmClient:
    """The command adapter is inert until called by an explicitly unlocked coordinator."""
    def submit(self, script, *, dependency=None):
        command = ['sbatch', '--parsable']
        if dependency:
            command.append(f'--dependency={dependency}')
        command.append(str(script))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';')[0]
        if not re.fullmatch(r'\d+', job_id):
            raise ValueError('Slurm returned an invalid job ID')
        return job_id

    def status(self, job_id):
        result = subprocess.run(['sacct', '-n', '-X', '-P', '-j', str(job_id),
                                 '--format=JobIDRaw,State'], check=True, capture_output=True, text=True)
        rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
        states = [_state(row[1]) for row in rows if len(row) > 1 and not '.' in row[0]]
        if not states:
            return 'UNKNOWN'
        if all(state == 'COMPLETED' for state in states):
            return 'COMPLETED'
        return next((state for state in states if state in ACTIVE), states[0])

    def task_status(self, job_id, index):
        return self.status(f'{job_id}_{index}')

    def cancel(self, job_id):
        subprocess.run(['scancel', str(job_id)], check=True, capture_output=True, text=True)


def _validate_runtime(runtime):
    if runtime.max_parallel != 2 or runtime.wall_hours > 24:
        raise ValueError('Campaign requires concurrency two and at most 24-hour segments')
    if runtime.gres.lower() != 'gpu:l40:1' or not 1 <= runtime.cpus <= 7 or runtime.memory_gb != 64:
        raise ValueError('Campaign requires one L40 GPU, at most seven CPUs and 64 GB')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', runtime.partition):
        raise ValueError('Unsafe partition')


def _execution_settings(root, environment=None, working_directory=None):
    """Read persisted settings, including campaigns generated before execution.json existed."""
    path = root / 'execution.json'
    saved = read_json(path) if path.is_file() else {}
    if not saved:
        for script in sorted((root / 'slurm').glob('phase_*.sh')):
            for line in script.read_text(encoding='utf-8').splitlines():
                tokens = shlex.split(line)
                if tokens[:2] == ['conda', 'activate'] and len(tokens) == 3:
                    saved['environment'] = tokens[2]
                if tokens[:1] == ['cd'] and len(tokens) == 2:
                    saved['working_directory'] = tokens[1]
            if {'environment', 'working_directory'} <= saved.keys():
                break
    name = environment or saved.get('environment', 'gnss-field')
    safe_name(name)
    work = working_directory if working_directory is not None else saved.get('working_directory', Path.cwd())
    return {'environment': name, 'working_directory': str(Path(work).resolve())}


def render_script(directory, tasks, runtime, *, task_file, environment='gnss-field',
                  working_directory=None, summary=False) -> str:
    """Render a Slurm script using literal task JSON and shell-quoted paths."""
    _validate_runtime(runtime)
    if len({task.gpu for task in tasks}) != 1:
        raise ValueError('An array must use one homogeneous CPU/GPU resource class')
    root = Path(directory).resolve()
    work = Path(working_directory or Path.cwd()).resolve()
    safe_name(environment)
    phase = tasks[0].phase
    # One array has one time limit. The shortest GPU reservation keeps every
    # member within its own declared worst case, including the four-hour benchmark.
    gpu_limits = [task.card_hours for task in tasks if task.gpu]
    wall_hours = min([runtime.wall_hours, *gpu_limits]) if gpu_limits and not summary else runtime.wall_hours
    seconds = int(wall_hours * 3600)
    if seconds < 1:
        raise ValueError('A GPU reservation must cover at least one second')
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    suffix = 'summary' if summary else 'array'
    lines = ['#!/bin/bash', f'#SBATCH --job-name=gnss-{phase:03d}-{suffix}',
             f'#SBATCH --partition={runtime.partition}',
             f'#SBATCH --cpus-per-task={1 if summary else runtime.cpus}',
             f'#SBATCH --mem={4 if summary else runtime.memory_gb}G',
             f'#SBATCH --time={hours:02d}:{minutes:02d}:{seconds:02d}',
             '#SBATCH --signal=USR1@180',
             f'#SBATCH --output={shlex.quote(str(root / "logs" / f"phase_{phase:03d}_{suffix}_%A_%a.out"))}',
             f'#SBATCH --error={shlex.quote(str(root / "logs" / f"phase_{phase:03d}_{suffix}_%A_%a.err"))}']
    if not summary:
        lines.append(f'#SBATCH --array=0-{len(tasks) - 1}%2')
        if any(task.gpu for task in tasks):
            lines.append(f'#SBATCH --gres={runtime.gres}')
    lines += ['set -euo pipefail', 'source /share/apps/miniconda3/etc/profile.d/conda.sh',
              f'conda activate {shlex.quote(environment)}', f'cd {shlex.quote(str(work))}']
    if summary:
        lines.append('srun python -u -m gnss_field.orchestration status --campaign-dir ' + shlex.quote(str(root)))
    else:
        lines.append('srun python -u -m gnss_field worker --task-file ' + shlex.quote(str(task_file)) +
                     ' --index "${SLURM_ARRAY_TASK_ID}"')
    return '\n'.join(lines) + '\n'


def write_campaign(directory, campaign: Campaign, runtime: RuntimeConfig | None = None, *,
                   environment='gnss-field', working_directory=None) -> dict:
    """Create dry-run artifacts. This function performs no subprocess calls or preflight claims."""
    runtime = runtime or RuntimeConfig()
    _validate_runtime(runtime)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'slurm').mkdir(exist_ok=True)
    execution = _execution_settings(root, environment, working_directory)
    immutable_json(root / 'campaign.json', campaign.to_dict())
    immutable_json(root / 'runtime.json', asdict(runtime))
    immutable_json(root / 'execution.json', execution)
    for phase in sorted({task.phase for task in campaign.tasks}):
        tasks = tuple(task for task in campaign.tasks if task.phase == phase)
        task_file = root / 'tasks' / f'phase_{phase:03d}.json'
        write_task_file(task_file, campaign, tasks)
        for summary in (False, True):
            suffix = '_summary' if summary else ''
            atomic_bytes(root / 'slurm' / f'phase_{phase:03d}{suffix}.sh',
                         render_script(root, tasks, runtime, task_file=task_file,
                                       environment=execution['environment'],
                                       working_directory=execution['working_directory'], summary=summary).encode('utf-8'))
    path = root / 'manifest.json'
    if not path.exists():
        write_json(path, {'campaign_id': campaign.campaign_id, 'campaign_hash': fingerprint(campaign.to_dict()),
                          'state': 'PLANNED', 'jobs': [], 'attempts': [], 'submission_intents': [],
                          'reserved_card_hours': 0., 'stage_reserved_card_hours': {}})
    manifest = read_json(path)
    if manifest.get('campaign_hash') != fingerprint(campaign.to_dict()):
        raise ValueError('Immutable campaign differs from coordinator manifest')
    return manifest


class Coordinator:
    """One coordinator owns campaign.json and manifest.json. Workers own per-task outputs."""
    def __init__(self, directory, campaign: Campaign, *, runtime=None, client=None,
                 environment=None, working_directory=None, _initialize=True):
        self.directory = Path(directory).resolve()
        self.campaign = campaign
        self.runtime = runtime or RuntimeConfig()
        self.client = client or SlurmClient()
        execution = _execution_settings(self.directory, environment, working_directory)
        self.environment = execution['environment']
        self.working_directory = Path(execution['working_directory'])
        if _initialize:
            write_campaign(self.directory, campaign, self.runtime, environment=self.environment,
                           working_directory=self.working_directory)
        elif self._read().get('campaign_hash') != fingerprint(campaign.to_dict()):
            raise ValueError('Immutable campaign differs from coordinator manifest')

    @classmethod
    def from_directory(cls, directory, *, client=None):
        root = Path(directory)
        return cls(root, Campaign.from_dict(read_json(root / 'campaign.json')),
                   runtime=RuntimeConfig(**read_json(root / 'runtime.json')), client=client, _initialize=False)

    def _read(self):
        return read_json(self.directory / 'manifest.json')

    def _write(self, manifest):
        write_json(self.directory / 'manifest.json', manifest)

    def _refresh(self, manifest):
        for job in manifest['jobs']:
            # A completed receipt never substitutes for scheduler completion.
            job['slurm_state'] = _state(self.client.status(job['job_id']))
        return manifest

    def _task_completed(self, manifest, task):
        attempts = [attempt for attempt in manifest['attempts'] if attempt['task_id'] == task.task_id]
        if not attempts:
            raise ValueError(f'No submitted dependency attempt for {task.task_id}')
        latest = attempts[-1]
        if not latest.get('job_id'):
            raise ValueError('Unreconciled dependency submission')
        if hasattr(self.client, 'task_status'):
            state = _state(self.client.task_status(latest['job_id'], latest['index']))
        else:
            state = _state(self.client.status(latest['job_id']))
        if state != 'COMPLETED':
            raise ValueError(f'Dependency {task.task_id} Slurm state is {state}')
        return verified_receipt(self.directory, task)

    def _authorize(self, tasks, manifest, preflight, budget_lock, benchmark_cap):
        benchmark_reservation = sum(task.card_hours * task.max_attempts for task in self.campaign.tasks
                                    if task.stage == 'benchmark')
        if benchmark_reservation > 4 + 1e-9:
            raise ValueError('Declared initial benchmark attempts exceed four L40 card hours')
        validate_preflight(preflight, self.campaign, self.runtime, require_driver=budget_lock is not None)
        if isinstance(self.client, SlurmClient):
            if os.name != 'posix' or preflight.get('hostname') != socket.gethostname():
                raise ValueError('Submission must run on the preflight-verified Tongji host')
        if budget_lock is not None:
            limits = validate_budget(self.campaign, budget_lock)
            total = limits['total_card_hours']
            stage_limits = limits['stage_card_hours']
        else:
            if (isinstance(benchmark_cap, bool) or not isinstance(benchmark_cap, (float, int)) or
                    not math.isfinite(benchmark_cap) or not 0 < benchmark_cap <= 4):
                raise ValueError('Initial benchmark needs an explicit cap of at most four L40 card hours')
            if any(task.gpu and task.stage != 'benchmark' for task in tasks):
                raise ValueError('A benchmark cap unlocks benchmark GPU work only')
            total = float(benchmark_cap)
            stage_limits = {'benchmark': float(benchmark_cap)}
        needed = sum(task.card_hours for task in tasks)
        if manifest['reserved_card_hours'] + needed > total + 1e-9:
            raise ValueError('Remaining total reservation is insufficient')
        stage_needed = defaultdict(float)
        for task in tasks:
            if task.gpu and task.card_hours <= 0:
                raise ValueError('GPU task requires declared worst-case card hours')
            if task.card_hours > self.runtime.wall_hours + 1e-9:
                raise ValueError('One-GPU task reservation exceeds its job segment')
            stage_needed[task.stage] += task.card_hours
        for stage, needed in stage_needed.items():
            if manifest['stage_reserved_card_hours'].get(stage, 0) + needed > stage_limits.get(stage, 0) + 1e-9:
                raise ValueError(f'Remaining stage reservation is insufficient for {stage}')

    def submit_phase(self, phase: int, *, execute=False, preflight=None, budget_lock=None,
                     benchmark_cap=None, _resume=False) -> dict:
        tasks = tuple(task for task in self.campaign.tasks if task.phase == phase)
        if not tasks:
            raise ValueError('Unknown phase')
        if not execute:
            return {'dry_run': True, 'phase': phase, 'tasks': [task.task_id for task in tasks],
                    'manifest': str(self.directory / 'manifest.json')}
        with single_owner(self.directory / '.coordinator.lock'):
            manifest = self._refresh(self._read())
            if manifest['state'] == 'PAUSED' and not _resume:
                raise ValueError('Campaign is paused. Use explicit resume')
            if any(intent['state'] != 'REGISTERED' for intent in manifest['submission_intents']):
                raise ValueError('An active or unreconciled submission intent blocks duplicate jobs')
            if any(job['slurm_state'] not in TERMINAL for job in manifest['jobs']):
                raise ValueError('An active or unknown campaign job blocks phase submission')
            by_id = {task.task_id: task for task in self.campaign.tasks}
            for task in tasks:
                for dependency in task.dependencies:
                    self._task_completed(manifest, by_id[dependency])
            remaining = []
            for task in tasks:
                attempts = [item for item in manifest['attempts'] if item['task_id'] == task.task_id]
                if attempts:
                    try:
                        self._task_completed(manifest, task)
                        continue
                    except ValueError:
                        pass
                if len(attempts) >= task.max_attempts:
                    raise ValueError(f'Task {task.task_id} exhausted its attempt limit')
                remaining.append(task)
            tasks = tuple(remaining)
            if not tasks:
                return {'dry_run': False, 'phase': phase, 'complete': True, 'tasks': []}
            self._authorize(tasks, manifest, preflight, budget_lock, benchmark_cap)
            phase_attempt = 1 + sum(job['kind'] == 'array' and job['phase'] == phase for job in manifest['jobs'])
            if phase_attempt == 1:
                stem = f'phase_{phase:03d}'
            else:
                stem = f'phase_{phase:03d}_attempt_{phase_attempt:03d}'
            task_file = self.directory / 'tasks' / f'{stem}.json'
            write_task_file(task_file, self.campaign, tasks)
            script = self.directory / 'slurm' / f'{stem}.sh'
            atomic_bytes(script, render_script(self.directory, tasks, self.runtime, task_file=task_file,
                                               environment=self.environment,
                                               working_directory=self.working_directory).encode())
            previous_phases = [task.phase for task in self.campaign.tasks if task.phase < phase]
            barrier_ids = []
            if previous_phases:
                previous = max(previous_phases)
                barrier_ids = [job['job_id'] for job in manifest['jobs']
                               if job['kind'] == 'array' and job['phase'] == previous and
                               job['slurm_state'] == 'COMPLETED']
            dependency = 'afterok:' + ':'.join(barrier_ids) if barrier_ids else None
            intent = {'token': f'{phase}:{phase_attempt}', 'phase': phase, 'state': 'SUBMITTING',
                      'task_ids': [task.task_id for task in tasks]}
            manifest['submission_intents'].append(intent)
            attempt_rows = []
            for index, task in enumerate(tasks):
                row = {'task_id': task.task_id, 'run_id': task.run_id, 'index': index,
                       'number': 1 + sum(a['task_id'] == task.task_id for a in manifest['attempts']),
                       'card_hours': task.card_hours, 'stage': task.stage, 'job_id': None}
                manifest['attempts'].append(row)
                attempt_rows.append(row)
                manifest['reserved_card_hours'] += task.card_hours
                stage = manifest['stage_reserved_card_hours']
                stage[task.stage] = stage.get(task.stage, 0.) + task.card_hours
            # Reservation and intent precede sbatch. An uncertain response cannot trigger another submission.
            self._write(manifest)
            job_id = str(self.client.submit(script, dependency=dependency))
            if not re.fullmatch(r'\d+', job_id):
                raise ValueError('Invalid job ID from submission adapter')
            intent['state'] = 'REGISTERED'
            intent['job_id'] = job_id
            for row in attempt_rows:
                row['job_id'] = job_id
            manifest['jobs'].append({'job_id': job_id, 'phase': phase, 'kind': 'array',
                                     'slurm_state': 'PENDING', 'task_ids': intent['task_ids']})
            manifest['state'] = 'RUNNING'
            self._write(manifest)
            summary_script = self.directory / 'slurm' / f'phase_{phase:03d}_summary.sh'
            # The summary is a CPU-only afterany job. Record its intent before sending it as well.
            summary_intent = {'token': intent['token'] + ':summary', 'phase': phase,
                              'state': 'SUBMITTING', 'task_ids': []}
            manifest['submission_intents'].append(summary_intent)
            self._write(manifest)
            summary_id = str(self.client.submit(summary_script, dependency=f'afterany:{job_id}'))
            if not re.fullmatch(r'\d+', summary_id):
                raise ValueError('Invalid summary job ID')
            summary_intent.update(state='REGISTERED', job_id=summary_id)
            manifest['jobs'].append({'job_id': summary_id, 'phase': phase, 'kind': 'summary',
                                     'slurm_state': 'PENDING', 'task_ids': []})
            self._write(manifest)
            return {'dry_run': False, 'phase': phase, 'job_id': job_id, 'summary_job_id': summary_id,
                    'tasks': intent['task_ids']}

    def resume(self, phase, **kwargs):
        """Regenerate only incomplete array members and retain each task's run ID and attempt ledger."""
        return self.submit_phase(phase, _resume=True, **kwargs)

    def status(self, *, query_slurm=True, write=False):
        with single_owner(self.directory / '.coordinator.lock') if write else nullcontext():
            manifest = self._read()
            if query_slurm:
                self._refresh(manifest)
            states = {}
            for task in self.campaign.tasks:
                path = self.directory / 'results' / task.task_id / 'status.json'
                states[task.task_id] = read_json(path)['state'] if path.exists() else 'PLANNED'
            result = {**manifest, 'task_states': states}
            if write:
                self._write(manifest)
            return result

    def pause(self, *, job_id=None):
        with single_owner(self.directory / '.coordinator.lock'):
            manifest = self._refresh(self._read())
            if job_id is not None and str(job_id) not in {job['job_id'] for job in manifest['jobs']}:
                raise ValueError('A pause can target only campaign-owned job IDs')
            for job in manifest['jobs']:
                if job_id is not None and job['job_id'] != str(job_id):
                    continue
                if job['slurm_state'] in ACTIVE:
                    for task_id in job['task_ids']:
                        task = next(task for task in self.campaign.tasks if task.task_id == task_id)
                        try:
                            self._task_completed(manifest, task)
                            continue
                        except ValueError:
                            pass
                        set_task_state(self.directory, task, 'PAUSED', detail='Campaign-owned job cancelled')
                    self.client.cancel(job['job_id'])
                    job['slurm_state'] = 'CANCELLED'
            manifest['state'] = 'PAUSED'
            self._write(manifest)
            return manifest
