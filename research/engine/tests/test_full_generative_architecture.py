import pytest
import torch

from gnss_field.generative.architecture import (
    ConditionEncoder, ConditionalFieldVAE, FieldDecoder, LatentCritic,
    LatentDenoiser, LatentGenerator, ObservationCritic,
)


@pytest.fixture
def batch():
    torch.manual_seed(73)
    torch.set_num_threads(2)
    y, x = torch.meshgrid(torch.arange(32) * 25000., torch.arange(32) * 25000., indexing='ij')
    geometry = torch.stack((x, y, torch.full_like(x, 100.)), -1)
    return dict(grid_condition=torch.randn(2, 37, 32, 32), station_tokens=torch.randn(2, 5, 31),
                station_mask=torch.tensor([[True, False, True, True, False], [False] * 5]),
                grid_geometry=geometry, input_geometry=geometry.reshape(-1, 3)[[0, 60, 120, 180, 240]],
                background=torch.full((2, 1, 32, 32), 100.),
                query_indices=torch.tensor([[0, 1, 32, 33], [400, 401, 432, 433], [800, 801, 832, 833]]),
                query_weights=torch.full((3, 4), .25), target_geometry=geometry.reshape(-1, 3)[[0, 400, 800]],
                target_mask=torch.tensor([[True, True, False], [True, False, True]]))


def encode(model, batch):
    return model(*(batch[key] for key in ('grid_condition', 'station_tokens', 'station_mask', 'grid_geometry', 'input_geometry')))


def test_encoder_geometry_attention_permutation_empty_set_and_masked_nan(batch):
    model = ConditionEncoder()
    original = encode(model, batch)
    assert [tuple(x.shape) for x in original.pyramid] == [(2, 32, 32, 32), (2, 64, 16, 16), (2, 96, 8, 8), (2, 128, 4, 4), (2, 192, 2, 2)]
    assert original.global_features.shape == (2, 192)
    permutation = torch.tensor([3, 0, 4, 2, 1])
    changed = {**batch, 'station_tokens': batch['station_tokens'][:, permutation],
               'station_mask': batch['station_mask'][:, permutation], 'input_geometry': batch['input_geometry'][permutation]}
    torch.testing.assert_close(encode(model, changed).global_features, original.global_features, atol=1e-6, rtol=1e-5)
    changed = {**batch, 'station_tokens': batch['station_tokens'].clone()}
    changed['station_tokens'][~batch['station_mask']] = float('nan')
    torch.testing.assert_close(encode(model, changed).global_features, original.global_features)
    changed = {**batch, 'station_tokens': batch['station_tokens'][:, :0], 'station_mask': batch['station_mask'][:, :0], 'input_geometry': batch['input_geometry'][:0]}
    assert torch.isfinite(encode(model, changed).global_features).all()


def test_encoder_input_gradients_and_invalid_shape(batch):
    batch['station_tokens'].requires_grad_()
    model = ConditionEncoder()
    encode(model, batch).global_features.square().mean().backward()
    assert batch['station_tokens'].grad[0, batch['station_mask'][0]].abs().sum() > 0
    assert batch['station_tokens'].grad[1].abs().sum() == 0
    assert model.geometry_bias.weight.grad.abs().sum() > 0
    assert model.null_token.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        encode(model, {**batch, 'grid_condition': torch.zeros(2, 36, 32, 32)})


def test_decoder_random_dependence_all_scales_and_gradients(batch):
    encoder = ConditionEncoder()
    decoder = FieldDecoder(latent_dim=8, grid_shape=(32, 32))
    z = torch.randn(2, 3, 8, requires_grad=True)
    fields = decoder(encode(encoder, batch), z, batch['background'])
    assert fields.shape == (2, 3, 1, 32, 32)
    assert (fields[:, 0] - fields[:, 1]).abs().mean() > 1e-5
    fields.square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_vae_posterior_is_masked_permutation_invariant_and_inference_label_free(batch):
    model = ConditionalFieldVAE(latent_dim=8, grid_shape=(32, 32))
    encoded = model.encode(batch)
    residual = torch.tensor([[2., -1., float('nan')], [3., float('nan'), 1.]])
    arguments = (encoded, residual, batch['query_indices'], batch['query_weights'], batch['target_geometry'], batch['target_mask'])
    mu, logstd = model.posterior_parameters(*arguments)
    permutation = torch.tensor([2, 0, 1])
    permuted = model.posterior_parameters(encoded, residual[:, permutation], batch['query_indices'][permutation],
                                          batch['query_weights'][permutation], batch['target_geometry'][permutation], batch['target_mask'][:, permutation])
    torch.testing.assert_close(mu, permuted[0])
    torch.testing.assert_close(logstd, permuted[1])
    noise = torch.randn(2, 2, 8)
    fields, *parameters = model.posterior_sample(encoded, batch['background'], *arguments[1:], noise)
    assert fields.shape == (2, 2, 1, 32, 32)
    (fields.mean() + sum(x.square().mean() for x in parameters)).backward()
    for part in (model.encoder, model.decoder, model.prior, model.posterior):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    clean = model.sample(model.encode(batch), batch['background'], noise)
    changed = model.sample(model.encode({**batch, 'target': torch.full((2, 3), float('nan'))}), batch['background'], noise)
    torch.testing.assert_close(clean, changed)
    with pytest.raises(ValueError, match='empty|valid'):
        model.posterior_parameters(*arguments[:-1], torch.zeros_like(batch['target_mask']))


def test_generator_critic_and_denoiser_have_condition_noise_gradient_paths():
    torch.manual_seed(12)
    noise = torch.randn(2, 3, 8, requires_grad=True)
    condition = torch.randn(2, 192, requires_grad=True)
    generator = LatentGenerator(latent_dim=8, width=32)
    critic = LatentCritic(latent_dim=8)
    generated = generator(noise, condition)
    assert generated.shape == noise.shape
    assert (generated[:, 0] - generated[:, 1]).abs().mean() > 1e-4
    critic(generated[:, 0], condition).sum().backward()
    assert noise.grad.abs().sum() > 0 and condition.grad.abs().sum() > 0
    denoiser = LatentDenoiser(latent_dim=8, width=32)
    sample = denoiser.denoise(noise[:, 0].detach(), torch.tensor([.1, 1.]), condition.detach())
    sample.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in denoiser.parameters())


def test_edm_sampling_prefix_batch_independence_and_35_evaluations():
    torch.manual_seed(12)
    model = LatentDenoiser(latent_dim=8, width=32)
    condition = torch.randn(2, 192)
    calls = []
    original = model.denoise
    def counted(*args):
        calls.append(1)
        return original(*args)
    model.denoise = counted
    together = model.sample(condition, torch.tensor([93, 62]), 3, 42)
    assert len(calls) == 35
    first = model.sample(condition[:1], torch.tensor([93]), 2, 42)
    torch.testing.assert_close(together[:1, :2], first, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(together).all()
    assert (together[:, 0] - together[:, 1]).abs().sum() > 0
    together.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_observation_critic_mask_pair_geometry_permutation_and_gradient():
    torch.manual_seed(1)
    model = ObservationCritic()
    values = torch.randn(2, 12, requires_grad=True)
    local = torch.randn(2, 12, 32)
    geometry = torch.stack((torch.arange(12) * 21000., torch.zeros(12), torch.arange(12) * 20.), -1)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[0, 1] = False
    mask[1] = False
    original = model(values, local, geometry, mask)
    permutation = torch.randperm(12)
    changed = model(values[:, permutation], local[:, permutation], geometry[permutation], mask[:, permutation])
    torch.testing.assert_close(original, changed, atol=2e-6, rtol=2e-5)
    corrupt = values.detach().clone()
    corrupt[~mask] = float('nan')
    torch.testing.assert_close(original, model(corrupt, local, geometry, mask))
    original.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad[~mask].abs().sum() == 0
    assert values.grad[mask].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.pair.parameters())


def test_vae_log_standard_deviation_limits_are_applied_to_both_distributions(batch):
    model = ConditionalFieldVAE(latent_dim=8, grid_shape=(32, 32))
    with torch.no_grad():
        model.prior[-1].weight.zero_()
        model.prior[-1].bias[8:] = 100.
        model.posterior.output[-1].weight.zero_()
        model.posterior.output[-1].bias[8:] = -100.
    encoded = model.encode(batch)
    mu_p, logstd_p = model.prior_parameters(encoded)
    mu_q, logstd_q = model.posterior_parameters(encoded, torch.zeros(2, 3), batch['query_indices'], batch['query_weights'],
                                               batch['target_geometry'], batch['target_mask'])
    assert (logstd_p == 2.).all() and (logstd_q == -6.).all()
    assert torch.isfinite(mu_p).all() and torch.isfinite(mu_q).all()
