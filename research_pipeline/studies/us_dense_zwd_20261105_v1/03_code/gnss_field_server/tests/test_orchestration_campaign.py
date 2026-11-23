from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from gnss_field.artifacts import read_json, sha256
from gnss_field.contracts import RuntimeConfig


def api():
    from gnss_field import orchestration
    return orchestration


HASHES = {name: 'a' * 64 for name in ('plan', 'data', 'code', 'protocol')}


def tiny_campaign():
    o = api()
    return o.Campaign('test', (
        o.Task('benchmark', 'benchmark', 0, ('benchmark',), card_hours=2),
        o.Task('fit', 'fit', 1, ('fit',), ('benchmark',), card_hours=3, max_attempts=2),
    ), HASHES)


class Slurm:
    def __init__(self):
        self.states = {}
        self.calls = []
        self.cancelled = []

    def submit(self, script, *, dependency=None):
        job_id = str(100 + len(self.calls))
        self.calls.append((Path(script), dependency))
        self.states[job_id] = 'PENDING'
        return job_id

    def status(self, job_id):
        return self.states[job_id]

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.states[job_id] = 'CANCELLED'


def preflight():
    return {'verified': True, 'site': 'Tongji', 'profile': 'tongji_l40',
            'gpu': 'L40', 'partition': 'L40', 'gres': 'gpu:l40:1',
            'cpus': 7, 'memory_gb': 64, 'driver_ok': True, 'account_ok': True,
            'hashes': HASHES}


def lock(total=8, stages=None):
    return {'locked': True, 'hashes': HASHES, 'total_card_hours': total,
            'stage_card_hours': stages or {'benchmark': 2, 'fit': 6}}


def test_default_dag_enforces_scientific_phase_barriers():
    o = api()
    campaign = o.build_default_campaign('campaign', HASHES)
    by_id = {task.task_id: task for task in campaign.tasks}
    assert len(by_id) == len(campaign.tasks)
    assert by_id['mean_base'].phase == by_id['mean_all'].phase
    assert by_id['mean_cache'].dependencies == ('mean_base', 'mean_all')
    assert {by_id[f'basis_cv_{i}'].phase for i in range(3)} == {by_id['basis_cv_0'].phase}
    assert by_id['representation_gate'].dependencies == ('basis_cv_0', 'basis_cv_1', 'basis_cv_2')
    assert by_id['basis_final'].dependencies == ('representation_gate',)
    assert by_id['evaluate2020'].dependencies == ('product_lock',)
    assert by_id['direction_gate'].scientific_gate
    assert all(task.run_id.startswith('campaign/') for task in campaign.tasks)
    assert by_id['benchmark'].max_attempts == 1
    assert by_id['mean_base'].max_attempts == 2


def test_dag_rejects_duplicate_missing_cycle_and_same_phase_dependency():
    o = api()
    a = o.Task('a', 'a', 0, ('a',))
    for tasks in ((a, a), (replace(a, dependencies=('missing',)),),
                  (replace(a, dependencies=('b',)), o.Task('b', 'b', 1, ('b',), ('a',))),
                  (a, o.Task('b', 'b', 0, ('b',), ('a',)))):
        with pytest.raises(ValueError):
            o.Campaign('x', tasks, HASHES)


def test_dry_run_writes_immutable_task_files_and_never_submits(tmp_path, monkeypatch):
    o = api()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('external command'))
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=client)
    result = coordinator.submit_phase(0)
    assert result['dry_run'] is True
    assert client.calls == []
    assert (tmp_path / 'logs').is_dir()
    task_file = tmp_path / 'tasks' / 'phase_000.json'
    assert read_json(task_file)['tasks'][0]['argv'] == ['benchmark']
    script = (tmp_path / 'slurm' / 'phase_000.sh').read_text()
    assert '#SBATCH --signal=USR1@180' in script
    assert '#SBATCH --array=0-0%2' in script
    assert '#SBATCH --time=02:00:00' in script
    assert 'srun python -u -m gnss_field worker --task-file ' in script
    assert '--index "${SLURM_ARRAY_TASK_ID}"' in script
    assert 'B:USR1' not in script
    assert '--requeue' not in script
    assert read_json(tmp_path / 'manifest.json')['jobs'] == []
    changed = replace(tiny_campaign(), tasks=(replace(tiny_campaign().tasks[0], argv=('changed',)), tiny_campaign().tasks[1]))
    with pytest.raises(ValueError, match='Immutable'):
        o.Coordinator(tmp_path, changed)


def test_tongji_bootstrap_uses_cpu_partition_for_checks_and_summaries(tmp_path):
    o = api()
    campaign = o.build_default_campaign('bootstrap', HASHES)
    runtime = replace(RuntimeConfig(), cpu_partition='intel')
    for stage in ('pack', 'audit', 'doctor', 'benchmark'):
        task = next(task for task in campaign.tasks if task.stage == stage)
        for summary in (False, True):
            script = o.render_script(tmp_path, (task,), runtime,
                                     task_file=tmp_path / 'tasks.json', summary=summary)
            if stage == 'benchmark' and not summary:
                assert '#SBATCH --partition=L40\n' in script
                assert '#SBATCH --gres=gpu:l40:1\n' in script
                assert '#SBATCH --time=04:00:00\n' in script
            else:
                assert '#SBATCH --partition=intel\n' in script
                assert '#SBATCH --gres=' not in script
                wall_limit = '00:10:00' if summary else '00:15:00'
                assert f'#SBATCH --time={wall_limit}\n' in script


def test_sbatch_rejection_shows_platform_error(tmp_path, monkeypatch):
    from gnss_field.orchestration.scheduler import SlurmClient

    def rejected(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr='sbatch: error: GPU resource required')

    monkeypatch.setattr(subprocess, 'run', rejected)
    with pytest.raises(RuntimeError, match='GPU resource required'):
        SlurmClient().submit(tmp_path / 'phase_000.sh')


def test_a800_campaign_requires_matching_gpu_evidence_and_requests_one_card(tmp_path):
    o = api()
    runtime = replace(RuntimeConfig(), partition='A800', gres='gpu:a800:1', max_parallel=1)
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), runtime=runtime, client=client)
    a800 = dict(preflight(), profile='tongji_a800', gpu='A800', partition='A800', gres='gpu:a800:1')
    with pytest.raises(ValueError, match='profile'):
        coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    with pytest.raises(ValueError, match='profile'):
        coordinator.submit_phase(0, execute=True, preflight=a800,
                                 budget_lock=dict(lock(), allocated_profile='tongji_l40'))
    assert client.calls == []
    coordinator.submit_phase(0, execute=True, preflight=a800, benchmark_cap=4)
    script = client.calls[0][0].read_text()
    assert '#SBATCH --partition=A800\n' in script
    assert '#SBATCH --gres=gpu:a800:1\n' in script
    assert '#SBATCH --array=0-0%1\n' in script


def test_submission_requires_verified_server_and_locked_bounds(tmp_path):
    o = api()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=Slurm())
    for kwargs in ({}, {'preflight': preflight()},
                   {'preflight': dict(preflight(), profile='local_rtx'), 'budget_lock': lock()},
                   {'preflight': preflight(), 'benchmark_cap': 4.01},
                   {'preflight': preflight(), 'budget_lock': lock(total=7)},
                   {'preflight': preflight(), 'budget_lock': lock(stages={'benchmark': 2, 'fit': 5})},
                   {'preflight': dict(preflight(), hashes=dict(HASHES, code='b' * 64)), 'budget_lock': lock()}):
        with pytest.raises(ValueError):
            coordinator.submit_phase(0, execute=True, **kwargs)
    assert coordinator.client.calls == []


def test_submit_registers_jobs_and_refuses_duplicate_active_array(tmp_path):
    o = api()
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=client)
    result = coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    assert result['job_id'] == '100'
    manifest = read_json(tmp_path / 'manifest.json')
    assert manifest['jobs'][0]['job_id'] == '100'
    assert manifest['jobs'][1]['kind'] == 'summary'
    assert client.calls[1][1] == 'afterany:100'
    with pytest.raises(ValueError, match='active'):
        coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    with pytest.raises(ValueError):
        coordinator.submit_phase(1, execute=True, preflight=preflight(), budget_lock=lock())


def test_dependencies_require_artifact_hash_and_completed_slurm_and_pass_gate(tmp_path):
    o = api()
    client = Slurm()
    campaign = tiny_campaign()
    coordinator = o.Coordinator(tmp_path, campaign, client=client)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), budget_lock=lock())
    output = tmp_path / 'result.bin'
    output.write_bytes(b'good')
    o.commit_task_result(tmp_path, campaign.tasks[0], {'result': output})
    with pytest.raises(ValueError):
        coordinator.submit_phase(1, execute=True, preflight=preflight(), budget_lock=lock())
    client.states['100'] = 'COMPLETED'
    client.states['101'] = 'COMPLETED'
    output.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact'):
        coordinator.submit_phase(1, execute=True, preflight=preflight(), budget_lock=lock())
    output.write_bytes(b'good')
    result = coordinator.submit_phase(1, execute=True, preflight=preflight(), budget_lock=lock())
    assert result['job_id'] == '102'
    assert client.calls[2][1] == 'afterok:100'


@pytest.mark.parametrize('state', ['PAUSED', 'FAILED', 'STOPPED_NO_GAIN'])
def test_pause_fail_and_scientific_stop_block_dependants(tmp_path, state):
    o = api()
    client = Slurm()
    campaign = tiny_campaign()
    coordinator = o.Coordinator(tmp_path, campaign, client=client)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), budget_lock=lock())
    client.states.update({'100': 'COMPLETED', '101': 'COMPLETED'})
    o.set_task_state(tmp_path, campaign.tasks[0], state)
    with pytest.raises(ValueError):
        coordinator.submit_phase(1, execute=True, preflight=preflight(), budget_lock=lock())


def test_resume_preserves_run_id_and_charges_failed_attempts(tmp_path):
    o = api()
    client = Slurm()
    campaign = tiny_campaign()
    coordinator = o.Coordinator(tmp_path, campaign, client=client)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), budget_lock=lock())
    client.states.update({'100': 'FAILED', '101': 'COMPLETED'})
    o.set_task_state(tmp_path, campaign.tasks[0], 'FAILED')
    # A max_attempts=1 reservation remains consumed after failure.
    with pytest.raises(ValueError, match='attempt'):
        coordinator.resume(0, execute=True, preflight=preflight(), budget_lock=lock())
    assert read_json(tmp_path / 'manifest.json')['reserved_card_hours'] == 2


def test_pause_cancels_only_campaign_owned_active_ids(tmp_path):
    o = api()
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=client)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    client.states['999'] = 'RUNNING'
    coordinator.pause()
    assert set(client.cancelled) == {'100', '101'}
    assert client.states['999'] == 'RUNNING'
    assert read_json(tmp_path / 'manifest.json')['state'] == 'PAUSED'


def test_pause_preserves_finished_array_members_and_resume_only_rebuilds_incomplete_tasks(tmp_path):
    o = api()

    class MemberSlurm(Slurm):
        def task_status(self, job_id, index):
            return 'COMPLETED' if job_id == '100' and index == 0 else self.status(job_id)

    client = MemberSlurm()
    campaign = o.Campaign('test', (
        o.Task('done', 'fit', 0, ('fit',), card_hours=1, max_attempts=2),
        o.Task('remaining', 'fit', 0, ('fit',), card_hours=1, max_attempts=2),
    ), HASHES)
    coordinator = o.Coordinator(tmp_path, campaign, client=client)
    limits = lock(total=4, stages={'fit': 4})
    coordinator.submit_phase(0, execute=True, preflight=preflight(), budget_lock=limits)
    output = tmp_path / 'done.bin'
    output.write_bytes(b'done')
    o.commit_task_result(tmp_path, campaign.tasks[0], {'result': output})
    client.states['100'] = 'RUNNING'
    coordinator.pause()
    assert read_json(tmp_path / 'results' / 'done' / 'status.json')['state'] == 'SUCCEEDED'
    resumed = coordinator.resume(0, execute=True, preflight=preflight(), budget_lock=limits)
    assert resumed['tasks'] == ['remaining']
    payload = read_json(tmp_path / 'tasks' / 'phase_000_attempt_002.json')
    assert payload['tasks'][0]['run_id'] == 'test/remaining'
    manifest = read_json(tmp_path / 'manifest.json')
    assert manifest['reserved_card_hours'] == 3
    assert len([row for row in manifest['attempts'] if row['task_id'] == 'done']) == 1


def test_unknown_submission_response_retains_reservation_and_blocks_duplicates(tmp_path):
    o = api()

    class LostResponse(Slurm):
        def submit(self, script, *, dependency=None):
            raise ConnectionError('scheduler accepted, connection lost')

    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=LostResponse())
    with pytest.raises(ConnectionError):
        coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    manifest = read_json(tmp_path / 'manifest.json')
    assert manifest['reserved_card_hours'] == 2
    assert manifest['submission_intents'][0]['state'] == 'SUBMITTING'
    with pytest.raises(ValueError, match='unreconciled'):
        coordinator.resume(0, execute=True, preflight=preflight(), benchmark_cap=4)


def test_one_l40_job_rejects_eight_cpus_as_documented_by_tongji(tmp_path):
    o = api()
    with pytest.raises(ValueError, match='seven'):
        o.Coordinator(tmp_path, tiny_campaign(), runtime=RuntimeConfig(cpus=8))


def test_owner_lock_releases_after_process_crash(tmp_path):
    import os
    import sys
    import gnss_field
    from gnss_field.orchestration.state import single_owner

    path = tmp_path / 'owner.lock'
    environment = dict(os.environ, PYTHONPATH=str(Path(gnss_field.__file__).parent.parent))
    completed = subprocess.run([sys.executable, '-c',
                               'import os,sys; from gnss_field.orchestration.state import single_owner; '
                               'lock=single_owner(sys.argv[1]); lock.__enter__(); os._exit(7)', str(path)],
                              env=environment, check=False, capture_output=True)
    assert completed.returncode == 7
    with single_owner(path):
        with pytest.raises(ValueError, match='owner lock'):
            with single_owner(path):
                pytest.fail('a concurrent owner acquired the same file')


def test_bootstrap_static_preflight_allows_gpu_benchmark_before_driver_check(tmp_path):
    o = api()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=Slurm())
    static = dict(preflight(), driver_ok=False)
    with pytest.raises(ValueError):
        coordinator.submit_phase(0, execute=True, preflight=static, budget_lock=lock())
    result = coordinator.submit_phase(0, execute=True, preflight=static, benchmark_cap=4)
    assert result['job_id'] == '100'


def test_default_dag_covers_each_registered_ablation():
    o = api()
    campaign = o.build_default_campaign('campaign', HASHES)
    variants = {task.argv[2] for task in campaign.tasks if task.stage == 'ablation'}
    assert variants == {'pretrain_only', 'no_geometry_bias', 'current_only', 'rank64', 'rank256', 'joint_encoder'}


def test_per_job_pause_refuses_foreign_job_ids(tmp_path):
    o = api()
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=client)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    with pytest.raises(ValueError, match='owned'):
        coordinator.pause(job_id='999')
    assert client.cancelled == []
    coordinator.pause(job_id='100')
    assert client.cancelled == ['100']


@pytest.mark.parametrize('hours,attempts', [(5., 1), (3., 2)])
def test_full_budget_lock_cannot_expand_initial_benchmark_ceiling(tmp_path, hours, attempts):
    o = api()
    campaign = o.Campaign('test', (o.Task('benchmark', 'benchmark', 0, ('benchmark',),
                                         card_hours=hours, max_attempts=attempts),), HASHES)
    client = Slurm()
    coordinator = o.Coordinator(tmp_path, campaign, client=client)
    with pytest.raises(ValueError, match='benchmark'):
        coordinator.submit_phase(0, execute=True, preflight=preflight(),
                                 budget_lock=lock(total=10, stages={'benchmark': 10}))
    assert client.calls == []
    assert read_json(tmp_path / 'manifest.json')['reserved_card_hours'] == 0


def test_reload_status_and_resume_preserve_persisted_execution_configuration(tmp_path, monkeypatch):
    o = api()
    campaign = o.Campaign('test', (o.Task('benchmark', 'benchmark', 0, ('benchmark',),
                                         card_hours=1, max_attempts=2),), HASHES)
    work = tmp_path / 'deployment with spaces'
    work.mkdir()
    client = Slurm()
    coordinator = o.Coordinator(tmp_path / 'campaign', campaign, client=client,
                                environment='custom-env', working_directory=work)
    coordinator.submit_phase(0, execute=True, preflight=preflight(), benchmark_cap=4)
    client.states.update({'100': 'FAILED', '101': 'COMPLETED'})
    o.set_task_state(coordinator.directory, campaign.tasks[0], 'PAUSED')
    original_script = coordinator.directory / 'slurm' / 'phase_000.sh'
    original = original_script.read_bytes()
    original_mtime = original_script.stat().st_mtime_ns
    other_directory = tmp_path / 'different caller'
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    loaded = o.Coordinator.from_directory(coordinator.directory, client=client)
    loaded.status(query_slurm=False)
    loaded.status(write=True)
    assert loaded.environment == 'custom-env'
    assert Path(loaded.working_directory) == work.resolve()
    assert original_script.read_bytes() == original
    assert original_script.stat().st_mtime_ns == original_mtime
    loaded.resume(0, execute=True, preflight=preflight(), benchmark_cap=4)
    retry = (loaded.directory / 'slurm' / 'phase_000_attempt_002.sh').read_text()
    assert 'conda activate custom-env' in retry
    assert str(work.resolve()) in retry


def test_pure_status_does_not_recreate_or_write_campaign_files(tmp_path, monkeypatch):
    o = api()
    import gnss_field.orchestration.scheduler as scheduler

    coordinator = o.Coordinator(tmp_path, tiny_campaign(), client=Slurm())
    deleted_script = tmp_path / 'slurm' / 'phase_000.sh'
    deleted_script.unlink()
    before = {path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob('*') if path.is_file()}

    def unexpected_write(*args, **kwargs):
        pytest.fail('status wrote a planned artifact')

    monkeypatch.setattr(scheduler, 'atomic_bytes', unexpected_write)
    monkeypatch.setattr(scheduler, 'write_json', unexpected_write)
    monkeypatch.setattr(scheduler, 'immutable_json', unexpected_write)
    loaded = o.Coordinator.from_directory(tmp_path, client=Slurm())
    assert loaded.status(query_slurm=False)['state'] == 'PLANNED'
    after = {path.relative_to(tmp_path): (path.read_bytes(), path.stat().st_mtime_ns)
             for path in tmp_path.rglob('*') if path.is_file()}
    assert after == before
    assert not deleted_script.exists()


def test_mixed_cpu_gpu_phases_are_rejected_before_array_allocation(tmp_path):
    o = api()
    gpu = o.Task('gpu', 'fit', 0, ('fit',), card_hours=1)
    cpu = o.Task('cpu', 'audit', 0, ('audit',), gpu=False)
    with pytest.raises(ValueError, match='resource'):
        o.Campaign('test', (gpu, cpu), HASHES)
    with pytest.raises(ValueError, match='resource'):
        o.render_script(tmp_path, (gpu, cpu), RuntimeConfig(), task_file=tmp_path / 'tasks.json')
