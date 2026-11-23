import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_B_assisted_diagnostic_never_uses_held_group_labels():
    from architecture_bottleneck_diagnostics import oracle_prediction
    h = np.array([[1., 0.], [0., 1.], [1., 1.]])
    truth = np.array([[2., 3., 5.], [4., 1., 5.]])
    mean = np.zeros_like(truth); valid = np.ones_like(truth, dtype=bool)
    fitted = np.array([True, True, False])
    first = oracle_prediction(h, truth, mean, valid, fitted, np.ones(2), .001)
    truth[:, 2] += 1000
    second = oracle_prediction(h, truth, mean, valid, fitted, np.ones(2), .001)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, 2], [5., 5.], atol=1e-5)


def test_oracle_zero_context_means_zero_correction():
    from architecture_bottleneck_diagnostics import oracle_prediction
    mean = np.ones((2, 3))*4
    pred = oracle_prediction(np.ones((3, 2)), mean*2, mean,
        np.zeros_like(mean, dtype=bool), np.array([True, True, False]), np.ones(2), 1.)
    np.testing.assert_array_equal(pred, mean)


def test_aggregated_RMSE_bootstrap_matches_existing_direct_resampling():
    from architecture_result_analysis import rmse_interval
    from gnss_field.evaluation import paired_block_bootstrap
    rng = np.random.default_rng(5)
    hours = np.arange(2000, 2000+24*21)
    truth = rng.normal(size=(len(hours), 3))
    baseline = truth+rng.normal(size=truth.shape)
    candidate = truth+.9*(baseline-truth)
    population = np.ones_like(truth, dtype=bool)
    population[::3, 1] = False
    expected = paired_block_bootstrap(hours, truth, candidate, baseline, population, replicates=100)
    result = rmse_interval(hours, truth, candidate, baseline, population, replicates=100)
    np.testing.assert_allclose(result['ci95_mm'], expected['ci95_mm'], atol=1e-12)
