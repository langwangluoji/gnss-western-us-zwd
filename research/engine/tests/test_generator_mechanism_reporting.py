import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))


def test_fixed_panel_metrics_keep_member_station_axes_and_macro_weights():
    from validate_generator_mechanisms import diagnostic_metrics
    samples = np.array([[[1., 3., 2.], [2., 4., 3.], [3., 5., 4.], [4., 6., 5.]],
                        [[2., 4., 3.], [3., 5., 4.], [4., 6., 5.], [5., 7., 6.]]])
    truth = np.array([[2., np.nan, 4.], [3., 5., 5.]])
    mask = np.isfinite(truth)
    xy = np.array([[0., 0.], [300000., 0.], [600000., 0.]])
    result = diagnostic_metrics(samples, truth, mask, xy)
    assert result['macro_mse_mm2'] == .25
    assert result['cross_cell_hours'] == 2
    assert np.isfinite(result['cross_cell_ES_mm'])
    assert result['cross_cell_station_columns'] == [0, 1, 2]


@pytest.mark.parametrize('rmse,es,crps,coverage,expected', [
    (7.8, 6.0, 4.0, .7, (True, False)),
    (7.8, 5.9, 3.95, .7, (True, True)),
    (8.2, 5.8, 3.8, .7, (False, False)),
    (7.8, 6.2, 4.0, .7, (False, False)),
    (8., 5.8, 3.8, .66, (False, False)),
])
def test_promotion_does_not_ignore_accuracy_score_or_coverage_regression(rmse, es, crps, coverage, expected):
    from report_generator_mechanisms import screen
    def record(r, e, c, p):
        return {'scores': {'withheld_weather_withheld': {'macro_rmse': r, 'group_ES_mm': e, 'macro_CRPS_mm': c, 'coverage90': p}}}
    result = screen(record(8., 6., 4., .7), record(rmse, es, crps, coverage))
    assert (result['point_screen'], result['probability_screen']) == expected


def test_point_confirmation_requires_all_replication_guardrails_and_interval():
    from report_generator_mechanisms import confirmed_point_direction
    good = dict(point_screen=True, delta_macro_rmse_ci95=[-.3, -.1])
    failed_replication = dict(point_screen=False, delta_macro_rmse_mm=-.001)
    assert confirmed_point_direction(good, good, good, good)
    assert not confirmed_point_direction(good, good, good, failed_replication)
    assert not confirmed_point_direction(dict(good, delta_macro_rmse_ci95=[-.3, .1]), good, good, good)
