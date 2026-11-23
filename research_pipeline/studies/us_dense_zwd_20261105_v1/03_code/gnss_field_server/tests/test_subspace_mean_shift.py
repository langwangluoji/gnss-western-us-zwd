"""Mean-only control preserves Gaussian shape and the common initial product."""
import torch


def test_mean_shift_control_changes_only_the_conditional_location():
    from gnss_field.models.mean_shift import SubspaceMeanShift
    model = SubspaceMeanShift(5, 3, width=12)
    condition, noise = torch.randn(2, 3), torch.randn(2, 7, 5)
    torch.testing.assert_close(model.sample_noise(condition, noise), noise, rtol=0, atol=0)
    with torch.no_grad():
        model.net[-1].bias.fill_(.25)
    result = model.sample_noise(condition, noise)
    torch.testing.assert_close(result-noise, torch.full_like(noise, .25))
    torch.testing.assert_close(result[:, 1]-result[:, 0], noise[:, 1]-noise[:, 0])
