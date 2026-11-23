"""The bounded convergence study must preserve legacy schedules and B-only gates."""
import numpy as np
import pytest


def test_legacy_coverage_schedule_is_unchanged():
    from gnss_field.training.observable import study_schedule
    maximum, science = study_schedule({'maximum_updates_per_arm': 1572}, 6288)
    assert maximum == 1572
    assert (science.warmup, science.evaluate_every, science.minimum_refinement_steps, science.patience) == (100, 100, 300, 3)


def test_convergence_schedule_is_bounded_and_manifest_consistent():
    from gnss_field.training.observable import study_schedule
    meta = dict(maximum_updates_per_arm=5000, convergence_protocol='mean_convergence_v1')
    maximum, science = study_schedule(meta, 6288)
    assert maximum == 5000
    assert (science.minimum_refinement_steps, science.patience, science.ema) == (1600, 10, .999)
    with pytest.raises(ValueError, match='schedule'):
        study_schedule(dict(meta, maximum_updates_per_arm=5001), 6288)
    with pytest.raises(ValueError, match='protocol'):
        study_schedule(dict(meta, convergence_protocol='unknown'), 6288)


def test_B_gate_uses_complete_weeks_and_fails_without_evidence():
    from gnss_field.training.observable_convergence_gate import paired_week_gate
    hours = np.repeat(np.arange(12)*168, 24)+np.tile(np.arange(24), 12)
    old = np.linspace(3., 6., len(hours))
    good = paired_week_gate(hours, old-.04, old)
    assert good['passed'] and good['interval95_mm'][1] < 0
    uncertain = old+np.repeat(np.array([-.3, .25]*6), 24)
    assert not paired_week_gate(hours, uncertain, old)['passed']
    with pytest.raises(ValueError, match='finite'):
        paired_week_gate(hours, old*np.nan, old)


def test_raw_diagnostic_does_not_create_a_competing_selection_product(tmp_path):
    import torch
    from gnss_field.contracts import ScienceConfig
    from gnss_field.training.engine import StageSpec, TrainEngine
    model = torch.nn.Linear(1, 1)
    engine = TrainEngine(model, tmp_path, np.arange(8),
        StageSpec('observable_mean_refine', 2, 2, 2, 2, 42, {}),
        ScienceConfig(startup_evaluation=0))
    engine.step = 1
    engine._evaluate(lambda current, step: {'joint': {'value': 2., 'raw_ES_mm': 1.}})
    assert set(engine.products) == {'joint'}
    assert engine.products['joint'][0]['value'] == 2.


def test_hourly_gate_energy_matches_registered_masked_objective():
    import torch
    from gnss_field.training.observable_convergence_gate import hourly_energy
    from gnss_field.objectives.scores import energy_score
    torch.manual_seed(42)
    samples = 100+torch.randn(3, 64, 7)
    truth = 100+torch.randn(3, 7)
    valid = torch.rand(3, 7) > .3
    truth[~valid] = float('nan')
    actual = hourly_energy(samples, truth, valid)
    expected = torch.stack([energy_score(samples[i:i+1], truth[i:i+1], valid[i:i+1]) for i in range(3)])
    torch.testing.assert_close(actual, expected)
