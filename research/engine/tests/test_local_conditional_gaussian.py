"""Conditional Gaussian scale has an auditable convex observation objective."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_conditional_scale_gradient_matches_finite_difference():
    from local_conditional_gaussian import scale_objective
    rng = np.random.default_rng(4)
    q, dim, x = rng.chisquare(7, 40), np.full(40, 7), rng.normal(size=40)
    data = np.c_[q, dim, x]
    p = np.array([.2, -.1])
    _, gradient = scale_objective(p, data)
    expected = [(scale_objective(p+direction, data)[0]-scale_objective(p-direction, data)[0])/2e-5 for direction in np.eye(2)*1e-5]
    np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-7)


def test_conditional_scale_recovers_simulated_weather_dependence():
    from local_conditional_gaussian import scale_objective
    rng = np.random.default_rng(42)
    x = rng.normal(size=20000)
    q = rng.chisquare(8, len(x))*np.exp(2*(-.1+.3*x))
    data = np.c_[q, np.full(len(x), 8), x]
    fit = minimize(scale_objective, np.zeros(2), args=(data,), jac=True, method='BFGS')
    np.testing.assert_allclose(fit.x, [-.1, .3], atol=.015)


def test_wet_condition_distinguishes_real_zero_from_missing_all_stations():
    from local_conditional_gaussian import wet_feature
    raw = np.full((3, 4, 5), np.nan)
    raw[0] = 0.
    raw[2, 0] = [5., 0., 0., 0., 0.]
    wet, present = wet_feature(raw)
    np.testing.assert_array_equal(present, [True, False, True])
    np.testing.assert_array_equal(wet, [0., 0., 5.])
