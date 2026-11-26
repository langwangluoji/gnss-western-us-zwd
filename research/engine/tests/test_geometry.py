import numpy as np
import torch


def test_mosaic_constant_and_all_contributors_receive_gradient():
    from gnss_field.geometry import mosaic
    origins = np.array([[0, 0], [0, 12], [12, 0], [12, 12]])
    p = torch.full((2, 4, 1, 24, 24), 3., requires_grad=True)
    image = mosaic(p, origins, (36, 36))
    torch.testing.assert_close(image, torch.full_like(image, 3.))
    image.square().sum().backward()
    assert (p.grad.abs().sum((0, 2, 3, 4)) > 0).all()


def test_published_query_and_projected_basis_agree_after_height_restore():
    from gnss_field.geometry import query_grid, basis_observation_matrix
    torch.manual_seed(1)
    phi = torch.randn(24 * 24, 8, dtype=torch.float64)
    c = torch.randn(3, 4, 8, dtype=torch.float64)
    xy = torch.tensor([[3.2, 5.6], [12.3, 10.1]], dtype=torch.float64)
    heights = torch.tensor([100., 1800.], dtype=torch.float64)
    field = torch.einsum('gr,bkr->bkg', phi, c).reshape(3, 4, 24, 24)
    actual = query_grid(field, xy) * torch.exp(-heights / 3000)
    h = basis_observation_matrix(phi, (24, 24), xy, heights)
    torch.testing.assert_close(actual, torch.einsum('nr,bkr->bkn', h, c))


def test_aligned_coarsening_preserves_constant_and_rejects_incomplete_shape():
    import pytest
    from gnss_field.geometry import coarsen_100km
    x = torch.ones(2, 1, 24, 24) * 8
    torch.testing.assert_close(coarsen_100km(x), x)
    with pytest.raises(ValueError):
        coarsen_100km(torch.ones(1, 1, 23, 24))


def test_single_and_empty_station_geometry_is_explicit():
    from gnss_field.geometry import input_geometry
    q = np.array([[0., 0.], [100000., 0.]])
    one = input_geometry(np.array([[0., 0.]]), q)
    assert not one['support'].any()
    assert one['count'].tolist() == [1, 1]
    assert np.isfinite(one['weights']).all()
    empty = input_geometry(np.empty((0, 2)), q)
    assert not empty['support'].any()
    assert (empty['count'] == 0).all()
