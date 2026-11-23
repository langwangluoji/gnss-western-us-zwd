"""Scientific contracts for the standalone local U-Net spike."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('weather_pilot', Path(__file__).parents[1]/'tools/simple_weather_unet_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_context_is_permutation_invariant_and_ignores_target_values():
    xy = np.array([[0., 0.], [25000., 0.], [0., 25000.], [25000., 25000.]])
    values = np.arange(20, dtype='f4').reshape(4, 5)+270
    values[:, 0] = [80., 100., 110., 130.]
    queries = np.array([[8000., 9000.], [14000., 18000.]])
    first = pilot.idw_fields(xy, values, queries, np.zeros(5))
    p = np.array([2, 0, 3, 1])
    second = pilot.idw_fields(xy[p], values[p], queries, np.zeros(5))
    np.testing.assert_allclose(first[0], second[0], atol=1e-6)
    missing, support = pilot.idw_fields(xy, np.full_like(values, np.nan), queries, np.ones(5))
    np.testing.assert_array_equal(missing, np.ones((2, 5)))
    assert not support.any()


def test_initial_mean_and_gradients_are_well_defined():
    torch.manual_seed(42)
    model = pilot.SimpleUNet(42)
    x = torch.randn(2, 42, 32, 24)
    result = model(x)
    assert result.shape == (2, 1, 32, 24)
    assert torch.equal(result, torch.zeros_like(result))
    (result-1).square().mean().backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_fail_on_nonfinite_in_scored_population():
    y = np.array([[1., 2.], [2., 3.]])
    mask = np.ones_like(y, dtype=bool)
    try:
        pilot.metrics(np.array([[1., np.nan], [2., 3.]]), y, mask)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite prediction must fail, not delete a sample')


def test_weather_buffers_include_cross_year_boundary():
    calendar = pilot.select_calendar()
    for start in calendar['inner_starts']+calendar['outer_starts']:
        assert not ((calendar['train'] >= start-18) & (calendar['train'] < start+72+18)).any()
