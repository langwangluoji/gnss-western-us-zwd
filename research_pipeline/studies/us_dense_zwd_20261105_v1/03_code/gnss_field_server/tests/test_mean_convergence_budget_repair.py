"""Reuse real warmed checkpoint timing without extrapolating CUDA startup."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def observations():
    return dict(updates=20, seconds_per_update=.17456666,
        development_seconds=.49108785, checkpoint_seconds=.0083468)


def test_cold_start_is_charged_once_and_steady_timing_is_measured():
    from lock_mean_convergence_repair import measured_estimate
    timing = [dict(step=200, elapsed=.7419286), dict(step=200, elapsed=.7667148)]
    result = measured_estimate(observations(), timing)
    assert 250 < result['estimated_seconds_with_margin'] < 260
    assert result['steady_seconds_per_update'] == pytest.approx(.7667148/200)
    changed = dict(observations(), seconds_per_update=.22456666)
    # One extra cold second changes the margin estimate by 1.5 seconds, not 375.
    assert measured_estimate(changed, timing)['estimated_seconds_with_margin']-result['estimated_seconds_with_margin'] == pytest.approx(1.5)


@pytest.mark.parametrize('timing', [
    [dict(step=20, elapsed=.74), dict(step=200, elapsed=.76)],
    [dict(step=200, elapsed=float('nan')), dict(step=200, elapsed=.76)],
    [dict(step=200, elapsed=-1.), dict(step=200, elapsed=.76)],
])
def test_missing_or_invalid_warm_measurement_cannot_unlock_budget(timing):
    from lock_mean_convergence_repair import measured_estimate
    with pytest.raises(ValueError):
        measured_estimate(observations(), timing)


def test_slow_measured_updates_still_exceed_same_fifteen_minute_cap():
    from lock_mean_convergence_repair import measured_estimate
    result = measured_estimate(observations(), [dict(step=200, elapsed=50.)]*2)
    assert result['estimated_seconds_with_margin'] > 900
