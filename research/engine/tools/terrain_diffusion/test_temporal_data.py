"""Contracts: no overlapping dates, role leakage, stale values, or coast units."""
import numpy as np
import pytest


def test_window_never_crosses_split():
    from .temporal_data import anchors_for_period
    h = np.arange(240)
    anchors = anchors_for_period(h, 48, 6, 100, 200)
    assert anchors[0] == 147 and anchors[-1] == 193
    assert np.all(anchors - 47 >= 100)
    assert np.all(anchors + 6 < 200)


def test_noncontinuous_axis_rejected():
    from .temporal_data import anchors_for_period
    with pytest.raises(ValueError, match='continuous'):
        anchors_for_period(np.array([0, 1, 3]), 1, 1, 0, 4)


def test_history_does_not_read_targets_or_future():
    from .temporal_data import spatial_history
    v = np.ones((12, 4, 3), dtype='float32') * 100
    v[:, :, 1] = 280
    v[:, :, 2] = 900
    xy = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype='float32')
    grid = np.array([[25, 25], [75, 75]], dtype='float32')
    kwargs = dict(values=v, hours=np.arange(12), anchor=8, length=6,
                  context=np.array([0, 1]), xy=xy, height=np.zeros(4), grid=grid)
    before = spatial_history(**kwargs)
    v[:, 2:] = 99999  # Target station history and labels must remain isolated.
    v[9:, :2] = 88888  # Future context also unavailable.
    after = spatial_history(**kwargs)
    np.testing.assert_array_equal(before, after)


def test_no_observations_are_distinct_from_zero_and_not_nan():
    from .temporal_data import spatial_history
    features = spatial_history(np.full((8, 2, 3), np.nan), np.arange(8), 7, 4,
                               np.array([0]), np.zeros((2, 2)), np.zeros(2), np.zeros((1, 2)))
    assert np.isfinite(features).all()
    assert features[..., 3].max() == 0  # ZWD availability channel


def test_coast_distance_km_and_segment_interior():
    from .coast_features import distance_to_coast
    line = np.array([[0, -10], [0, 10]], dtype=float)
    result = distance_to_coast(np.array([[0, 0], [3, 0]], dtype=float), [line])
    np.testing.assert_allclose(result, [0, 3], atol=1e-6)
