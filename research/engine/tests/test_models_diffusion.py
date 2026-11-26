"""EDM algebra, keyed streams, solver evaluations, and end-to-end gradients."""
import importlib

import pytest
import torch

torch.set_num_threads(2)


def diffusion():
    try:
        return importlib.import_module('gnss_field.models.diffusion')
    except ModuleNotFoundError as exc:
        pytest.fail(f'Coefficient EDM implementation is missing: {exc}')


def test_edm_coefficients_match_hand_computed_standard_formula():
    got = diffusion().edm_coefficients(torch.tensor([.5, 1.], dtype=torch.float64))
    expected = ([.5, .2], [.3535533905932738, .4472135954999579],
                [1.4142135623730951, .8944271909999159], [-.17328679513998632, 0.], [8., 5.])
    assert len(got) == 5
    for actual, numbers in zip(got, expected):
        torch.testing.assert_close(actual, torch.tensor(numbers, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_keyed_noise_is_exactly_prefix_batch_and_order_independent():
    noise = diffusion().keyed_noise
    torch.manual_seed(3)
    before = torch.random.get_rng_state().clone()
    all_draws = noise([5, 12, 99], 5, seed=22, shape=(9,))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(all_draws[:, :2], noise([5, 12, 99], 2, seed=22, shape=(9,)))
    assert torch.equal(all_draws[[2, 0]], noise([99, 5], 5, seed=22, shape=(9,)))
    assert torch.equal(all_draws[1:2], noise([12], 5, seed=22, shape=(9,)))
    assert not torch.equal(all_draws, noise([5, 12, 99], 5, seed=23, shape=(9,)))
    assert all_draws.shape == (3, 5, 9)
    assert noise([12], 1, 22, shape=(9,), dtype=torch.float64).dtype == torch.float64


def test_heun_generic_tensor_uses_35_calls_and_preserves_all_call_gradients():
    fn = diffusion().heun_sample
    gains = torch.linspace(.01, .04, 35, dtype=torch.float64, requires_grad=True)
    calls = []
    def denoise(x, sigma):
        index = len(calls); calls.append(float(sigma))
        return x / (1 + sigma) + gains[index]
    output = fn(denoise, torch.randn(2, 7, dtype=torch.float64))
    assert output.shape == (2, 7)
    assert len(calls) == 35
    assert min(calls) > 0
    output.square().sum().backward()
    assert torch.isfinite(gains.grad).all()
    assert (gains.grad.abs() > 0).all()


def test_heun_one_dimension_known_constant_clean_solution():
    fn = diffusion().heun_sample
    clean = torch.tensor([[.25, -.75]])
    result = fn(lambda x, sigma: clean.expand_as(x), torch.tensor([[1., 2.]]))
    torch.testing.assert_close(result, clean, atol=2e-6, rtol=0)


def test_edm_head_uses_preconditioning_and_supports_scalar_and_batch_sigma():
    model = diffusion().EDMHead(rank=7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias.fill_(2.)
    x = torch.randn(2, 7); h = torch.randn(2, 128)
    torch.testing.assert_close(model.denoise(x, .5, h), .5 * x + .7071067811865476,
                               rtol=1e-6, atol=1e-6)
    sigma = torch.tensor([.5, 1.])
    expected = x * torch.tensor([.5, .2])[:, None] + torch.tensor([.7071067811865476, .8944271909999159])[:, None]
    torch.testing.assert_close(model(x, sigma, h), expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        model(x, torch.ones(3), h)
    with pytest.raises(ValueError):
        model(x, 0., h)


def test_edm_samples_keep_prefixes_and_batch_identity():
    torch.manual_seed(93)
    model = diffusion().EDMHead(rank=9).eval()
    h = torch.randn(2, 128)
    with torch.no_grad():
        full = model.sample(h, [103, 509], members=4, seed=12)
        prefix = model.sample(h, [103, 509], members=2, seed=12)
        alone = model.sample(h[1:2], [509], members=4, seed=12)
    assert full.shape == (2, 4, 9)
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full[:, :2], prefix, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(full[1:2], alone, rtol=2e-5, atol=2e-5)
    assert not torch.allclose(full[:, 0], full[:, 1])


def test_real_edm_sample_backpropagates_through_all_35_denoiser_calls():
    torch.manual_seed(24)
    model = diffusion().EDMHead(rank=8)
    h = torch.randn(1, 128, requires_grad=True)
    activations = []
    def retain_output(module, args, output):
        output.retain_grad(); activations.append(output)
    handle = model.output.register_forward_hook(retain_output)
    samples = model.sample(h, [110], members=2, seed=71)
    handle.remove()
    assert len(activations) == 35
    # A score-style pair term supplies a training gradient through the ensemble.
    target = torch.linspace(-1, 1, 8)
    score = (samples - target).square().sum(-1).sqrt().mean()
    score -= .5 * (samples[:, 0] - samples[:, 1]).square().sum(-1).sqrt().mean()
    score.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert all(a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
               for a in activations)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('args', [dict(members=0), dict(members=2, hour_ids=[]),
                                  dict(members=2, steps=1)])
def test_edm_rejects_invalid_ensemble_or_solver_requests(args):
    model = diffusion().EDMHead(rank=4)
    arguments = dict(hour_ids=[1], members=2, seed=4)
    arguments.update(args)
    with pytest.raises(ValueError):
        model.sample(torch.zeros(1, 128), **arguments)
