"""Whole-grid conditioning, spatial capacity, and coefficient distributions."""
from dataclasses import replace
import importlib

import pytest
import torch

from gnss_field.contracts import ContextBatch

torch.set_num_threads(2)


def component(name):
    try:
        return importlib.import_module('gnss_field.models.' + name)
    except ModuleNotFoundError as exc:
        pytest.fail(f'Standalone {name} component is missing: {exc}')


def context(n=5):
    torch.manual_seed(30)
    return ContextBatch(hours=torch.tensor([11, 27]), stations=torch.randn(2, n, 17),
                        padding=torch.zeros(2, n, dtype=torch.bool),
                        coarse=torch.randn(2, 4, 15, 6, 7), maps=torch.randn(2, 6, 24, 28),
                        baseline=torch.randn(2, 1, 24, 28))


def test_condition_encoder_ignores_padded_values_and_station_order():
    model = component('condition').ConditionEncoder().eval()
    b = context()
    b.padding[:, -2:] = True
    reference = model(b)
    bad = b.stations.clone(); bad[:, -2:] = float('nan')
    changed_padding = model(replace(b, stations=bad))
    order = torch.tensor([3, 1, 4, 0, 2])
    reordered = model(replace(b, stations=b.stations[:, order], padding=b.padding[:, order]))
    assert reference.shape == (2, 128)
    torch.testing.assert_close(reference, changed_padding, rtol=0, atol=0)
    torch.testing.assert_close(reference, reordered, rtol=2e-5, atol=2e-6)


def test_condition_encoder_empty_station_set_matches_all_padding_and_backpropagates():
    model = component('condition').ConditionEncoder()
    b = context()
    padded = replace(b, stations=torch.full_like(b.stations, float('nan')),
                     padding=torch.ones_like(b.padding))
    empty = replace(b, stations=b.stations[:, :0], padding=b.padding[:, :0])
    first = model(padded); second = model(empty)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    first.square().mean().backward()
    assert model.null_station.grad is not None
    assert model.null_station.grad.abs().sum() > 0


def test_condition_encoder_uses_supplied_mean_and_fixed_support_maps():
    model = component('condition').ConditionEncoder().eval()
    b = context()
    default = model(b)
    torch.testing.assert_close(default, model(replace(b, mean=b.baseline)), rtol=0, atol=0)
    supplied = model(replace(b, mean=b.baseline + 5))
    assert not torch.allclose(default, supplied)
    newmaps = b.maps.clone(); newmaps[:, 3:] += 2
    assert not torch.allclose(default, model(replace(b, maps=newmaps)))
    default.sum().backward()
    assert model.station_encoder[0].weight.grad.abs().sum() > 0
    assert model.coarse_encoder[0].weight.grad.abs().sum() > 0


def test_spatial_basis_has_true_rank_128_and_fixed_land_rms():
    torch.manual_seed(9)
    model = component('basis').SpatialBasis(rank=128).double()
    xyh = torch.rand(640, 3, dtype=torch.float64) * torch.tensor([1600000., 1400000., 3000.])
    land = torch.arange(640) % 4 != 0
    phi = model(xyh, land)
    assert phi.shape == (640, 128)
    torch.testing.assert_close(phi[land].square().mean(0), torch.ones(128, dtype=phi.dtype),
                               rtol=1e-8, atol=1e-8)
    # A width-64 bottleneck or repeated output columns cannot pass this test.
    assert torch.linalg.matrix_rank(phi).item() == 128
    assert torch.linalg.svdvals(phi)[-1] > 1e-5
    phi[land].mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_spatial_basis_validates_rank_and_nonempty_fixed_land():
    basis = component('basis').SpatialBasis
    with pytest.raises(ValueError):
        basis(rank=257)
    with pytest.raises(ValueError):
        basis(rank=0)
    with pytest.raises(ValueError):
        basis(rank=8)(torch.zeros(10, 3), torch.zeros(10, dtype=torch.bool))


def test_coefficient_heads_shapes_condition_and_noise_gradients():
    heads = component('heads')
    h = torch.randn(3, 128, requires_grad=True)
    noise = torch.randn(3, 19, requires_grad=True)
    det = heads.DeterministicHead(rank=19)
    gaussian = heads.GaussianHead(rank=19)
    generator = heads.GANGenerator(rank=19)
    critic = heads.ConditionalCritic(rank=19)
    coefficients = det(h)
    mu, log_std = gaussian(h)
    generated = generator(noise, h)
    score = critic(generated, h)
    assert coefficients.shape == mu.shape == log_std.shape == generated.shape == (3, 19)
    assert score.shape == (3,)
    assert log_std.min() >= -5 and log_std.max() <= 3
    (coefficients.square().mean() + mu.square().mean() + log_std.square().mean() + score.mean()).backward()
    assert h.grad.abs().sum() > 0 and noise.grad.abs().sum() > 0


def test_gaussian_head_predicts_nonzero_variance_and_clamps_extreme_log_std():
    model = component('heads').GaussianHead(rank=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.log_std.bias.fill_(.69314718056)
    mean, log_std = model(torch.zeros(2, 128))
    assert torch.equal(mean, torch.zeros_like(mean))
    torch.testing.assert_close(log_std.exp().square(), torch.full_like(mean, 4))
    noise = torch.randn(50000, 4, generator=torch.Generator().manual_seed(18))
    samples = mean[:1] + log_std[:1].exp() * noise
    torch.testing.assert_close(samples.var(0), torch.full((4,), 4.), rtol=.025, atol=.025)
    with torch.no_grad():
        model.log_std.bias.copy_(torch.tensor([-20., -5., 3., 20.]))
    _, clipped = model(torch.zeros(1, 128))
    torch.testing.assert_close(clipped, torch.tensor([[-5., -5., 3., 3.]]), rtol=0, atol=0)
