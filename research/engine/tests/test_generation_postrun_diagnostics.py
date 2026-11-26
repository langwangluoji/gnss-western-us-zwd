"""Properties of the post-run decomposition, independent of model implementation."""
import numpy as np
import pytest

from generation_postrun_diagnostics import matched_decomposition, rank_histogram, fair_crps, independent_centering, paired_block_rmse


def forecast(samples, mean):
    return dict(samples=samples, mean=mean, truth=np.ones(mean.shape),
                population=np.ones(mean.shape, bool), hours=np.arange(len(mean)),
                station_ids=np.arange(mean.shape[1]))


def test_pure_translation_has_no_distribution_shape_change():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(4, 32, 6)); base = np.zeros((4, 6))
    shift = rng.normal(size=(4, 6))
    gaussian = forecast(z, base)
    generated = forecast(z + shift[:, None], shift)
    result = matched_decomposition(gaussian, generated)
    np.testing.assert_allclose(result['gaussian_shifted'], generated['samples'])
    np.testing.assert_allclose(result['generated_recentered'], gaussian['samples'])
    np.testing.assert_allclose(result['mean_shift'], shift)


def test_cannot_compare_changed_cohort_or_drop_nonfinite_samples():
    a = forecast(np.zeros((2, 4, 3)), np.zeros((2, 3)))
    b = {k: v.copy() for k, v in a.items()}
    b['population'][0, 0] = False
    with pytest.raises(ValueError, match='population'):
        matched_decomposition(a, b)
    b = {k: v.copy() for k, v in a.items()}
    b['samples'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        matched_decomposition(a, b)


def test_exchangeable_rank_histogram_is_uniform_with_ties():
    rng = np.random.default_rng(82)
    # Deliberate ties: randomized ordering among equal values is required.
    values = rng.integers(0, 4, size=(20000, 17, 1))
    hist = rank_histogram(values[:, 0], values[:, 1:],
                          np.ones((20000, 1), bool), seed=19)
    assert len(hist['counts']) == 17
    assert max(abs(np.asarray(hist['frequency']) - 1/17)) < .006


def test_fair_crps_equals_one_dimensional_unbiased_energy_score():
    from gnss_field.objectives.scores import energy_score
    import torch
    z = np.array([[[-3., 1.], [0., 4.], [2., 5.], [4., 8.]]])
    y = np.array([[1., 3.]])
    actual = fair_crps(y, z)
    for j in range(2):
        expected = energy_score(torch.tensor(z[:, :, j:j+1]), torch.tensor(y[:, j:j+1]))
        assert actual[0, j] == pytest.approx(float(expected))


def test_independent_centering_removes_a_pure_translation_without_reusing_members():
    rng = np.random.default_rng(53)
    z = rng.normal(size=(3, 32, 4)); base = np.zeros((3, 4))
    shift = np.arange(12).reshape(3, 4)
    a, b = forecast(z, base), forecast(z+shift[:, None], shift)
    result = independent_centering(a, b, np.arange(16), np.arange(16, 32))
    np.testing.assert_allclose(result['samples'], z[:, 16:], atol=1e-14)
    np.testing.assert_allclose(result['mean'], base, atol=1e-14)
    np.testing.assert_allclose(result['scale'], np.ones(3), atol=1e-14)
    with pytest.raises(ValueError, match='overlap'):
        independent_centering(a, b, np.arange(16), np.arange(8, 24))


def test_block_rmse_interval_keeps_all_stations_and_recovers_constant_error():
    truth = np.zeros((24, 3)); mask = np.ones((24, 3), bool)
    block = np.repeat(np.arange(12), 2)
    result = paired_block_rmse(truth, truth, np.ones_like(truth), mask, block)
    assert result['difference_mm'] == pytest.approx(1.)
    assert result['interval95_mm'] == pytest.approx([1., 1.])
    assert result['blocks'] == 12

