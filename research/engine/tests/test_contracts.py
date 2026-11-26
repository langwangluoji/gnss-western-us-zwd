import pytest


def test_temporal_roles_and_budgets_are_validated():
    from gnss_field.contracts import ScienceConfig, RuntimeConfig
    assert ScienceConfig().rank == 128
    assert ScienceConfig().base_years == (2016, 2017)
    assert ScienceConfig().residual_years == (2018,)
    assert RuntimeConfig().max_parallel == 2
    with pytest.raises(ValueError):
        ScienceConfig(base_years=(2019,))
    with pytest.raises(ValueError):
        ScienceConfig(rank=300)


def test_canonical_hash_is_order_independent_and_nonfinite_is_rejected():
    from gnss_field.artifacts import fingerprint
    assert fingerprint({'x': 1, 'y': 2}) == fingerprint({'y': 2, 'x': 1})
    with pytest.raises(ValueError):
        fingerprint({'score': float('nan')})


def test_resolved_json_round_trip_preserves_worker_science_hash(tmp_path):
    from gnss_field.artifacts import fingerprint, write_json
    from gnss_field.contracts import ScienceConfig, load_config

    science = ScienceConfig()
    resolved = tmp_path / 'resolved.json'
    write_json(resolved, {'science': science})
    assert '1e-06' in resolved.read_text(encoding='utf-8')

    worker_science, _, _ = load_config(resolved)
    assert worker_science.minimum_lr == science.minimum_lr
    assert fingerprint(worker_science) == fingerprint(science)
