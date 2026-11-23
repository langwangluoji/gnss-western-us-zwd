import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help_is_portable_and_does_not_start_training():
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, '-m', 'gnss_field', '--help'],
                             env={**__import__('os').environ, 'PYTHONPATH': str(root/'src')},
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'campaign' in result.stdout and 'benchmark' in result.stdout


def test_local_doctor_cannot_authorize_tongji_submission(tmp_path):
    from gnss_field.runtime import doctor
    from gnss_field.contracts import RuntimeConfig
    result = doctor(RuntimeConfig(device='cpu'), output=tmp_path/'doctor.json')
    assert result['verified'] is False
    assert result['site'] != 'Tongji' or not result['account_ok']
    assert (tmp_path/'doctor.json').exists()


def test_benchmark_rejects_local_allocation_before_training(tmp_path, monkeypatch):
    import pytest
    from gnss_field.runtime import benchmark
    from gnss_field.contracts import RuntimeConfig, ScienceConfig
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='allocation'):
        benchmark(ScienceConfig(), RuntimeConfig(), tmp_path, hashes={}, static_preflight={})


def test_step_signal_requires_real_step_identifier(monkeypatch):
    import pytest
    from gnss_field.runtime import _send_step_usr1
    monkeypatch.delenv('SLURM_STEP_ID', raising=False)
    with pytest.raises(RuntimeError, match='step'):
        _send_step_usr1()


def test_step_signal_targets_python_step_without_shell(monkeypatch):
    from gnss_field.runtime import _send_step_usr1
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    monkeypatch.setenv('SLURM_STEP_ID', '0')
    seen = []
    def run(args, **kwargs):
        seen.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', run)
    assert _send_step_usr1() == '12345.0'
    assert seen[0][0] == ['scancel', '--signal=USR1', '12345.0']
    assert 'shell' not in seen[0][1]



def test_release_hashes_checks_registered_data_and_all_content(legacy_fixture, tmp_path):
    import pytest
    from gnss_field.data import pack_legacy
    from gnss_field.runtime import verified_release_hashes
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    from gnss_field.artifacts import fingerprint, read_json, write_json
    packed = tmp_path/'packed'
    pack_legacy(legacy_fixture, packed, registered=False)
    runtime = RuntimeConfig(data_root=str(packed))
    with pytest.raises(ValueError, match='synthetic'):
        verified_release_hashes(ScienceConfig(), runtime, tmp_path)
    manifest = read_json(packed/'manifest.json')
    manifest['registered'] = True
    manifest.pop('fingerprint')
    manifest['fingerprint'] = fingerprint(manifest)
    write_json(packed/'manifest.json', manifest)
    import shutil
    import gnss_field
    shutil.copytree(Path(gnss_field.__file__).parent, tmp_path/'src/gnss_field', ignore=shutil.ignore_patterns('__pycache__'))
    hashes = verified_release_hashes(ScienceConfig(), runtime, tmp_path)
    assert hashes['data'] == manifest['fingerprint']
    from dataclasses import replace
    assert verified_release_hashes(replace(ScienceConfig(), rank=64), runtime, tmp_path)['plan'] != hashes['plan']
    (tmp_path/'src').mkdir(exist_ok=True)
    (tmp_path/'src/change.py').write_text('changed = True')
    assert verified_release_hashes(ScienceConfig(), runtime, tmp_path)['code'] != hashes['code']
    shard = packed/next(iter(manifest['files']))
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Corrupt'):
        verified_release_hashes(ScienceConfig(), runtime, tmp_path)



def test_direct_benchmark_checks_actual_release_before_training(tmp_path, monkeypatch):
    import pytest
    import types
    import gnss_field.runtime as runtime_module
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    hashes = dict(plan='old', data='data', code='code', protocol='protocol')
    monkeypatch.setattr(runtime_module, 'os', types.SimpleNamespace(name='posix', environ={'SLURM_JOB_ID': '123'}))
    monkeypatch.setattr(runtime_module, 'verified_release_hashes', lambda *a: {**hashes, 'plan': 'new'})
    monkeypatch.setattr(runtime_module, 'engineering_smoke', lambda *a, **k: pytest.fail('training reached'))
    with pytest.raises(ValueError, match='differs'):
        runtime_module.benchmark(ScienceConfig(), RuntimeConfig(), tmp_path,
                                 hashes=hashes, static_preflight={'verified': True, 'hashes': hashes})



@pytest.mark.parametrize(('gpu', 'rate'), [('L40', 2), ('A800', 4)])
def test_benchmark_cost_estimates_use_registered_effective_hours(tmp_path, monkeypatch, gpu, rate):
    import types
    import gnss_field.runtime as module
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    hashes = dict(plan='plan', data='data', code='code', protocol='protocol')
    profile = {arm+suffix: 2 for arm in ('det', 'gaussian', 'edm', 'gan')
               for suffix in ('_pretrain_batch', '_observation_batch')}
    profile.update(m2_full_field_observation_batch=2,
                   mean_microbatch_sweep=[dict(micro_hours=1, patch_chunk=8, seconds=8)])
    smoke = dict(profile=profile, mean_seconds=2, ridge_seconds=2, es_seconds=2,
                 resume={'slurm_step_signal': {'passed': True}}, verified_for_tongji_budget=False)
    monkeypatch.setattr(module, 'os', types.SimpleNamespace(name='posix', environ={'SLURM_JOB_ID': '1'}))
    monkeypatch.setattr(module, 'verified_release_hashes', lambda *a: hashes)
    monkeypatch.setattr(module, 'doctor', lambda *a, **k: dict(driver_ok=True, actual_gpu=gpu, actual_memory_gb=80))
    monkeypatch.setattr(module, 'engineering_smoke', lambda *a, **k: smoke)
    runtime = RuntimeConfig(partition=gpu, gres=f'gpu:{gpu.lower()}:1')
    preflight = dict(verified=True, hashes=hashes, profile=f'tongji_{gpu.lower()}',
                     partition=gpu, gres=runtime.gres)
    report = module.benchmark(ScienceConfig(), runtime, tmp_path,
                              hashes=hashes, static_preflight=preflight)
    assert report['allocated_profile'] == f'tongji_{gpu.lower()}'
    stages = {row['stage']: row for row in report['stage_cost_estimates']}
    for stage, hours in [('mean_base', 8), ('basis', 16), ('m2_f', 8), ('m2_j', 8), ('edm_refine', 32)]:
        assert stages[stage]['effective_hours'] == hours
        assert stages[stage]['estimated_update_seconds'] == hours
        assert stages[stage]['gpu_only_CNY'] == stages[stage]['estimated_card_hours']*rate
