import pytest
import torch


def test_unbiased_mean_mse_matches_cross_products_and_gradient():
    from gnss_field.objectives.ensemble_mean import unbiased_mean_mse
    x = torch.tensor([[[-2.], [2.]]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, 1, dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.bool)
    value = unbiased_mean_mse(x, target, mask)
    assert value.item() == -4.
    torch.testing.assert_close(torch.autograd.grad(value, x)[0], torch.tensor([[[2.], [-2.]]], dtype=torch.float64))


def test_unbiased_mean_mse_uses_hour_macro_and_ignores_missing_labels():
    from gnss_field.objectives.ensemble_mean import unbiased_mean_mse
    x = torch.tensor([[[1., 3.], [2., 4.]], [[5., 7.], [6., 8.]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0., float('nan')], [1., 2.]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True]])
    # Two members: cross error product; station mean per hour, then hour mean.
    expected = (2. + (4.*5.+5.*6.)/2)/2
    assert unbiased_mean_mse(x, target, mask).item() == expected
    gradient = torch.autograd.grad(unbiased_mean_mse(x, target, mask), x)[0]
    assert gradient[0, :, 1].abs().sum().item() == 0.
    assert torch.autograd.gradcheck(lambda z: unbiased_mean_mse(z, target, mask), (x,))


def test_unbiased_mean_mse_rejects_bad_predictions_and_empty_hours():
    from gnss_field.objectives.ensemble_mean import unbiased_mean_mse
    x = torch.zeros(1, 4, 2)
    y = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        unbiased_mean_mse(x, y, torch.zeros_like(y, dtype=torch.bool))
    x[0, 0, 1] = float('nan')
    with pytest.raises(ValueError):
        unbiased_mean_mse(x, y, torch.tensor([[True, False]]))
