"""Joint-field likelihood and reproducible observation-ensemble checks."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_composite_gaussian_gradient_matches_finite_difference():
    from local_joint_field import covariance_objective
    rng = np.random.default_rng(8)
    a = rng.normal(size=(3, 5, 7))
    kernels = a@a.transpose(0, 2, 1)
    cross = rng.normal(size=(30, 5))
    records = [(30, cross.T@cross/30, kernels)]
    p = np.array([.3, -.2, .1])
    value, gradient = covariance_objective(p, records)
    expected = np.empty(3)
    for j in range(3):
        direction = np.eye(3)[j]*1e-5
        expected[j] = (covariance_objective(p+direction, records)[0]-covariance_objective(p-direction, records)[0])/2e-5
    np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-7)


def test_joint_members_are_prefix_and_station_subset_invariant():
    from local_joint_field import sample_factors
    rng = np.random.default_rng(2)
    h = [rng.normal(size=(7, 4)), rng.normal(size=(7, 3))]
    ids = np.arange(7)+3
    full = sample_factors(h, np.array([2., 1., 3.]), [422000, 422006], ids, 16, 42)
    selected = [1, 4, 6]
    part = sample_factors([v[selected] for v in h], np.array([2., 1., 3.]), [422006], ids[selected], 8, 42)
    np.testing.assert_array_equal(full[1, :8, selected].T, part[0])


def test_member_shuffle_preserves_every_marginal_exactly():
    from local_joint_field import shuffle_members
    rng = np.random.default_rng(42)
    values = rng.normal(size=(3, 64, 8))
    shuffled = shuffle_members(values, [120, 121, 122], np.arange(8), 42)
    np.testing.assert_array_equal(np.sort(values, axis=1), np.sort(shuffled, axis=1))


def test_diagonal_control_matches_declared_gaussian_variances():
    from local_joint_field import sample_diagonal
    std = np.array([2., 4., 7.])
    members = sample_diagonal(std, [422000], np.array([3, 5, 7]), 20000, 42)[0]
    np.testing.assert_allclose(members.std(0), std, rtol=.02)
    assert abs(np.corrcoef(members.T)[0, 1]) < .03


def test_joint_monthly_scores_keep_fixed_group_ids_when_one_group_has_no_hours():
    import pandas as pd
    from local_probability_experiment import joint_series
    metadata = pd.DataFrame(dict(dense_group=['a']*4+['b']*4, x=np.arange(8)*100., y=np.zeros(8)))
    truth = np.ones((2, 8))
    members = np.ones((2, 4, 8))
    population = np.ones((2, 8), bool)
    population[:, 4:] = False
    series, groups = joint_series(truth, members, population, metadata)
    assert groups == ['a', 'b']
    assert series.shape == (2, 2, 2)
    assert np.isnan(series[:, 1]).all()
