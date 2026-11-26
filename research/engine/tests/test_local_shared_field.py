"""Contracts for an A-only local residual experiment."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_setconv_preserves_constants_and_ignores_invalid_values():
    from local_shared_field import setconv
    kernel = torch.tensor([[1., 2., 0.], [0., 0., 0.]])
    value = torch.tensor([[7., 7., float('nan')]])
    valid = torch.tensor([True, True, False])
    field, density = setconv(kernel, value, valid)
    torch.testing.assert_close(field, torch.tensor([[7., 0.]]))
    torch.testing.assert_close(density, torch.tensor([3., 0.]))
    empty, _ = setconv(kernel, value, torch.zeros(3, dtype=torch.bool))
    assert torch.isfinite(empty).all() and not empty.any()


def test_month_balanced_split_keeps_whole_nominal_weeks():
    from local_shared_field import balanced_blocks
    import pandas as pd
    dates = pd.date_range('2018-01-01', '2018-12-31', freq='6h')
    hours = dates.to_numpy().astype('datetime64[h]').astype('i8')
    train, development = balanced_blocks(hours)
    assert not set(hours[train]//168) & set(hours[development]//168)
    assert set(dates[development].month) == set(range(1, 13))
    assert set(dates[train].month) == set(range(1, 13))


def test_nested_block_selection_never_sees_outer_weather_hours():
    from local_shared_field import nested_blocks
    import pandas as pd
    dates = pd.date_range('2018-01-01', '2018-12-31', freq='6h')
    hours = dates.to_numpy().astype('datetime64[h]').astype('i8')
    core, inner, outer = nested_blocks(hours)
    assert not (core & inner).any() and not (inner & outer).any()
    assert not (core & outer).any() and (core | inner | outer).all()
    blocks = [set(hours[mask]//168) for mask in (core, inner, outer)]
    assert not blocks[0] & blocks[1] and not blocks[1] & blocks[2]


def test_ridge_ignores_held_labels_and_finds_known_shared_correction():
    from local_shared_field import fit_shared_ridge, apply_shared_ridge
    rng = np.random.default_rng(42)
    features = rng.normal(size=(24, 7, 3))
    truth = 2.*features[..., 0]-.5*features[..., 2]
    valid = np.ones(truth.shape, bool)
    valid[12:] = False
    first = fit_shared_ridge(features, truth, valid, 0.)
    changed = truth.copy()
    changed[~valid] = 1e9
    second = fit_shared_ridge(features, changed, valid, 0.)
    np.testing.assert_array_equal(first['weights'], second['weights'])
    np.testing.assert_allclose(apply_shared_ridge(features, first), truth, atol=1e-10)


def test_shared_affine_field_commutes_with_height_query():
    from local_shared_field import fit_shared_ridge, apply_shared_ridge
    from gnss_field.geometry import surface_query
    rng = np.random.default_rng(1)
    grid = torch.tensor(rng.normal(size=(2, 3, 8, 8)), dtype=torch.float64)
    xy = torch.tensor([[1.2, 3.4], [5.1, 4.2]], dtype=torch.float64)
    height = torch.tensor([100., 1800.], dtype=torch.float64)
    # Explicit constant is a Z0 field, so its height transform is retained.
    grid[:, 0] = 1
    queried = surface_query(grid, xy, height).numpy().transpose(0, 2, 1)
    weights = np.array([.8, -.7, .1])
    truth = queried@weights
    state = fit_shared_ridge(queried, truth, np.ones(truth.shape, bool), 0.)
    # Centering after query would add a surface intercept and break this.
    correction = apply_shared_ridge(grid.numpy().transpose(0, 2, 3, 1), state)
    published = surface_query(torch.tensor(correction[:, None]), xy, height).numpy()[:, 0]
    np.testing.assert_allclose(published, apply_shared_ridge(queried, state), atol=1e-10)
