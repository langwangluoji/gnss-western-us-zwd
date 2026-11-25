import numpy as np


def test_station_macro_differs_from_micro_for_unequal_station_error():
    from .temporal_experiment import score
    truth = np.zeros((2, 6, 2))
    prediction = np.zeros_like(truth)
    prediction[:, :, 1] = 20
    metrics = score(prediction, truth)
    assert metrics['macro_rmse_mm'] == 10
    np.testing.assert_allclose(metrics['micro_rmse_mm'], np.sqrt(200))
    assert metrics['increment_macro_rmse_mm'] == 0


def test_score_masks_missing_targets():
    from .temporal_experiment import score
    truth = np.zeros((2, 6, 2))
    truth[:, :, 0] = np.nan
    prediction = np.full_like(truth, 3)
    metrics = score(prediction, truth)
    assert metrics['macro_rmse_mm'] == 3
    assert metrics['station_count'] == 1
