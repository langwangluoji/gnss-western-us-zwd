import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from gnss_field.models.conditioned_unet import (ConditionedUNet, FieldGAN, FieldVAE,
                                               FieldDiffusion, fair_scores)


def inputs():
    torch.manual_seed(42)
    return torch.randn(2, 42, 16, 12), torch.zeros(2, 1, 16, 12)


def test_encoder_grows_but_decoder_stays_light():
    small, wide = ConditionedUNet((16, 32, 64)), ConditionedUNet((32, 64, 128))
    assert wide.decoder.up2[0].out_channels == small.decoder.up2[0].out_channels == 32
    assert wide.decoder.up1[0].out_channels == small.decoder.up1[0].out_channels == 16
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in small.parameters())


@pytest.mark.parametrize('family', [FieldGAN, FieldVAE, FieldDiffusion])
def test_full_grid_and_sampler_gradient(family):
    x, base = inputs()
    model = family()
    # Production generation starts from a trained mean; exercise its nonzero head.
    with torch.no_grad():
        model.core.decoder.head.weight.normal_(0, .01)
    members = model.sample(x, base, [10, 20], 3, 42)
    assert members.shape == (2, 3, 1, 16, 12)
    assert torch.isfinite(members).all()
    (members-1).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if isinstance(model, FieldDiffusion):
        assert model.last_calls == 11
        assert all(p.grad is not None for p in model.noise_encoder.parameters())


@pytest.mark.parametrize('family', [FieldGAN, FieldVAE, FieldDiffusion])
def test_member_identity_does_not_depend_on_batch_or_prefix(family):
    x, base = inputs()
    model = family().eval()
    with torch.no_grad():
        all_members = model.sample(x, base, [10, 20], 4, 42)
        prefix = model.sample(x[:1], base[:1], [10], 2, 42)
    torch.testing.assert_close(all_members[:1, :2], prefix, atol=1e-5, rtol=1e-5)


def test_three_models_do_not_share_vae_dependency():
    assert not any(isinstance(m, FieldVAE) for m in FieldGAN().modules())
    assert not any(isinstance(m, FieldVAE) for m in FieldDiffusion().modules())


def test_vae_posterior_never_changes_deployment_prior_path():
    x, base = inputs()
    model = FieldVAE().eval()
    with torch.no_grad():
        model.core.decoder.head.weight.normal_(0, .01)
        model.posterior[-1].weight.normal_(0, .01)
        encoded = model.encode(x)
        first = model.sample_encoded(encoded, base, [10, 20], 2, 42)
        posterior = torch.randn(2, 2, 16, 12)
        a, _ = model.posterior_sample(encoded, base, posterior, [10, 20], 2, 42)
        b, _ = model.posterior_sample(encoded, base, posterior+100, [10, 20], 2, 42)
        second = model.sample_encoded(encoded, base, [10, 20], 2, 42)
    assert not torch.equal(a, b)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_nonfinite_condition_fails_before_sampling():
    x, base = inputs()
    x[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        FieldDiffusion().sample(x, base, [10, 20], 2, 42)


def test_energy_and_crps_use_off_diagonal_estimates():
    # For K2, draws0/2 and observation1: E|X-y|=1; half pair term=1.
    samples = torch.tensor([[[0.], [2.]]], requires_grad=True)
    es, crps = fair_scores(samples, torch.tensor([[1.]]), torch.tensor([[True]]))
    torch.testing.assert_close(es, torch.tensor(0.))
    torch.testing.assert_close(crps, torch.tensor(0.))
    with pytest.raises(ValueError):
        fair_scores(torch.tensor([[[float('nan')], [2.]]]), torch.tensor([[1.]]), torch.tensor([[True]]))


def test_training_qualification_uses_only_fitting_labels():
    sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
    from train_independent_generators import qualified_training_rows
    mask = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [0, 0, 0, 0, 1]], bool)
    rows, ids = np.arange(3), np.arange(4)
    np.testing.assert_array_equal(qualified_training_rows(mask, rows, ids), [0])
    mask[:, 4] = ~mask[:, 4]
    np.testing.assert_array_equal(qualified_training_rows(mask, rows, ids), [0])


def test_spatial_evaluation_uses_original_calendar_not_training_qualification():
    sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
    from train_independent_generators import fixed_spatial_rows
    original = np.arange(864)
    expected = original[np.rint(np.linspace(0, 863, 48)).astype(int)]
    np.testing.assert_array_equal(fixed_spatial_rows(original), expected)
