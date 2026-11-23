import torch


def test_decoder_fast_query_equals_published_grid_with_training_gradients():
    from gnss_field.training.system import FrozenDecoder
    from gnss_field.objectives import CoefficientTransform
    from gnss_field.geometry import surface_query
    torch.manual_seed(11)
    phi = torch.randn(24*24, 8)
    transform = CoefficientTransform.fit(torch.randn(30, 8))
    decoder = FrozenDecoder(phi, (24, 24), transform, 40.)
    z = torch.randn(2, 4, 8, requires_grad=True)
    mu = torch.rand(2, 1, 24, 24)*200
    xy = torch.tensor([[5.3, 6.7], [10.1, 12.2]])
    heights = torch.tensor([100., 1100.])
    grid = decoder.field(z, mu)
    fast = decoder.station(z, surface_query(mu, xy, heights).squeeze(1), xy, heights)
    actual = surface_query(grid.squeeze(2), xy, heights)
    torch.testing.assert_close(fast, actual, atol=.0001, rtol=1e-5)
    fast.square().sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
