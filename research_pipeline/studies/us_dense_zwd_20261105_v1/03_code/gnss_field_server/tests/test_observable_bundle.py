"""Server coverage study excludes complete weather blocks and preserves roles."""
import numpy as np
import pytest


def test_coverage_manifest_excludes_full_week_and_both18hour_buffers():
    from gnss_field.training.observable import coverage_split
    hours = np.arange(100*168, 105*168)
    anchor = np.array([102*168+60])
    train = coverage_split(hours, np.ones(len(hours), bool), anchor)
    removed = ~train
    assert hours[removed].min() == 102*168-18
    assert hours[removed].max() == 103*168+17
    assert not train[np.searchsorted(hours, anchor)].any()


def test_bundle_loader_requires_training_B_and_verified_hash(tmp_path):
    from gnss_field.training.observable import load_bundle
    from gnss_field.artifacts import write_json, sha256
    path = tmp_path/'data.npz'
    np.savez(path, hours=np.array([420768]))
    write_json(tmp_path/'manifest.json', dict(role='C', year=2019, data_sha256=sha256(path)))
    with pytest.raises(PermissionError):
        load_bundle(tmp_path)


def test_campaign_lease_rejects_a_second_live_process(tmp_path):
    from gnss_field.training.observable import acquire_study_lease
    path = acquire_study_lease(tmp_path)
    try:
        with pytest.raises(FileExistsError):
            acquire_study_lease(tmp_path)
    finally:
        path.unlink()


def test_development_evaluator_refuses_sealed_D2020(tmp_path, monkeypatch):
    from gnss_field.artifacts import write_json
    from gnss_field.training.observable_evaluate import main
    import sys
    write_json(tmp_path/'manifest.json', dict(role='D', year=2020))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--bundle', str(tmp_path), '--run', str(tmp_path),
        '--arm', 'mean', '--output', str(tmp_path/'out')])
    with pytest.raises(PermissionError):
        main()


def test_local_benchmark_cannot_unlock_formal_A800_budget(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    from lock_observable_budget import main
    from gnss_field.artifacts import write_json
    write_json(tmp_path/'benchmark.json', dict(formal_verification=False))
    write_json(tmp_path/'manifest.json', dict(maximum_updates_per_arm=1572))
    output = tmp_path/'budget.json'
    monkeypatch.setattr(sys, 'argv', ['budget', '--benchmark', str(tmp_path/'benchmark.json'),
        '--bundle', str(tmp_path), '--output', str(output)])
    with pytest.raises(ValueError):
        main()
    assert not output.exists()


def test_published_release_rejects_modified_shared_engine(tmp_path):
    from gnss_field.training.release import verify_release
    from gnss_field.artifacts import write_json, sha256
    _published_fixture(tmp_path)
    engine = tmp_path/'src/gnss_field/training/engine.py'
    assert verify_release(tmp_path) == sha256(tmp_path/'release.json')
    engine.write_text('modified engine')
    with pytest.raises(ValueError, match='engine.py'):
        verify_release(tmp_path)


def _published_fixture(root):
    from gnss_field.training.release import REQUIRED_FILES
    from gnss_field.artifacts import write_json, sha256
    files = {}
    for name in REQUIRED_FILES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixed dependency')
        files[name] = sha256(path)
    write_json(root/'release.json', dict(files=files))


def test_published_release_rejects_omitted_mandatory_dependency(tmp_path):
    from gnss_field.training.release import verify_release
    from gnss_field.artifacts import read_json, write_json
    _published_fixture(tmp_path)
    manifest = read_json(tmp_path/'release.json')
    del manifest['files']['src/gnss_field/training/engine.py']
    write_json(tmp_path/'release.json', manifest)
    with pytest.raises(ValueError, match='mandatory'):
        verify_release(tmp_path)


def test_initial_missing_after_updates_is_rejected(tmp_path):
    import torch
    from gnss_field.training.observable import restore_initial_checkpoint
    from gnss_field.training.engine import TrainEngine, StageSpec
    from gnss_field.contracts import ScienceConfig
    engine = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8),
        StageSpec('observable_mean_refine', 4, 4, 2, 2, 42, {'test': 'missing-initial'}), ScienceConfig())
    engine.step = 1
    with pytest.raises(ValueError, match='initial.pt'):
        restore_initial_checkpoint(engine)


def test_formal_release_requires_manifest(tmp_path):
    from gnss_field.training.release import verify_release
    with pytest.raises(ValueError, match='release.json'):
        verify_release(tmp_path)
    assert verify_release(tmp_path, required=False) is None


def test_A800_budget_rejects_missing_real_slurm_signal_probe(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    from lock_observable_budget import main
    from gnss_field.artifacts import write_json, sha256
    write_json(tmp_path/'manifest.json', dict(maximum_updates_per_arm=1572))
    write_json(tmp_path/'benchmark.json', dict(formal_verification=True,
        real_100_plus100_equals200=True, gpu='NVIDIA A800', updates=20,
        bundle_manifest_sha256=sha256(tmp_path/'manifest.json'), real_slurm_signal_resume=False))
    monkeypatch.setattr(sys, 'argv', ['budget', '--benchmark', str(tmp_path/'benchmark.json'),
        '--bundle', str(tmp_path), '--output', str(tmp_path/'budget.json')])
    with pytest.raises(ValueError, match='Slurm signal'):
        main()


def test_initial_only_restart_preserves_step_zero_candidate(tmp_path):
    import torch
    from gnss_field.training.observable import restore_initial_checkpoint
    from gnss_field.training.engine import TrainEngine, StageSpec, atomic_torch
    from gnss_field.contracts import ScienceConfig
    spec = StageSpec('observable_mean_refine', 4, 4, 2, 2, 42, {'test': 'initial-only'})
    cfg = ScienceConfig()
    first = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8), spec, cfg)
    first.products = {'joint': [{'value': 1., 'step': 0, 'checkpoint': str(tmp_path/'initial.pt')}]}
    atomic_torch(tmp_path/'initial.pt', first.state())
    restarted = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8), spec, cfg)
    assert not restarted.products and not (tmp_path/'last.pt').exists()
    restore_initial_checkpoint(restarted)
    assert restarted.products == first.products
    assert restarted.step == 0
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, restarted.model.state_dict()[key])
def test_month_cache_is_bound_to_evaluation_lock(tmp_path):
    from gnss_field.training.observable_evaluate import save_month, load_month
    path = tmp_path/'forecast_2019-01.npz'
    values = dict(hours=np.arange(8), station_ids=np.array([1, 3]), samples=np.ones((8, 4, 2)))
    save_month(path, 'evaluation-one', **values)
    saved = load_month(path, 'evaluation-one')
    np.testing.assert_array_equal(saved['samples'], values['samples'])
    with pytest.raises(ValueError, match='evaluation'):
        load_month(path, 'evaluation-two')

