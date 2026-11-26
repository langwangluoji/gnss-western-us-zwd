import math

import pytest
import torch

from gnss_field.generative.observations import (
    diagonal_gaussian_kl, query_features, query_surface, sample_surface, variational_loss,
)


def test_query_is_bilinear_then_surface_height_and_differentiable():
    fields = torch.tensor([[[[2., 4.], [6., 8.]]]], requires_grad=True)
    indices = torch.tensor([[0, 1, 2, 3]])
    weights = torch.tensor([[.25, .25, .25, .25]])
    result = query_surface(fields, indices, weights, torch.tensor([3000.]))
    torch.testing.assert_close(result, torch.tensor([[[5. / math.e]]]))
    result.sum().backward()
    torch.testing.assert_close(fields.grad, torch.full_like(fields, .25 / math.e))
    torch.testing.assert_close(query_features(fields.detach(), indices, weights), torch.tensor([[[5.]]]))


def test_query_rejects_nonfinite_fields_and_invalid_weights():
    indices = torch.zeros((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match='finite'):
        query_surface(torch.full((1, 1, 2, 2), float('nan')), indices, torch.full((1, 4), .25), torch.zeros(1))
    with pytest.raises(ValueError, match='weight'):
        query_surface(torch.ones(1, 1, 2, 2), indices, torch.ones(1, 4), torch.zeros(1))


def test_diagonal_kl_sums_dimensions_against_hand_calculation():
    mu = torch.tensor([[1., 2.]])
    zero = torch.zeros_like(mu)
    torch.testing.assert_close(diagonal_gaussian_kl(mu, zero, zero, zero), torch.tensor([2.5]))
    torch.testing.assert_close(diagonal_gaussian_kl(zero, zero, zero, zero), torch.zeros(1))
    # q standard deviation 2 and p standard deviation 1: 0.5*(4-1)-log(2).
    torch.testing.assert_close(diagonal_gaussian_kl(torch.zeros(1, 1), torch.full((1, 1), math.log(2.)),
                                                    torch.zeros(1, 1), torch.zeros(1, 1)),
                               torch.tensor([1.5 - math.log(2.)]))


def test_gaussian_elbo_common_reference_count_and_invalid_slot_sanitization():
    predictions = torch.tensor([[[2., float('nan')]], [[0., 2.]]], requires_grad=True)
    targets = torch.tensor([[0., float('nan')], [0., 0.]])
    mask = torch.tensor([[True, False], [True, True]])
    mu = torch.tensor([[1., 2.], [1., 2.]])
    zero = torch.zeros_like(mu)
    result = variational_loss(predictions, targets, mask, mu, zero, zero, zero, 2., reference_count=4.)
    c = math.log(2.) + .5 * math.log(2 * math.pi)
    expected_nll = ((c + .5) + (2 * c + .5)) / 2 / 4
    assert result['nll'].item() == pytest.approx(expected_nll)
    assert result['kl'].item() == pytest.approx(2.5 / 4)
    assert result['kl_per_dim'].item() == pytest.approx(1.25)
    result['loss'].backward()
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad[0, 0, 1] == 0


def test_observation_noise_added_once_and_empty_loss_rejected():
    batch = dict(query_indices=torch.zeros(1, 4, dtype=torch.long), query_weights=torch.full((1, 4), .25),
                 target_geometry=torch.tensor([[0., 0., 3000.]]))
    fields = torch.full((1, 2, 1, 2, 2), 10.)
    actual = sample_surface(fields, batch, torch.ones(1, 2, 1), 2.)
    torch.testing.assert_close(actual, torch.full((1, 2, 1), 10. / math.e + 2.))
    zero = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='valid|empty'):
        variational_loss(torch.zeros(1, 1, 1), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool),
                         zero, zero, zero, zero, 2.)
