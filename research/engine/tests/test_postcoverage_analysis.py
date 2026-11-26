import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_independent_integration_recovers_fixed_location_without_observation_noise():
    from postcoverage_analysis import latent_mean_shift
    class Location(torch.nn.Module):
        rank = 4
        def sample_noise(self, condition, noise):
            return noise+condition[:, None]
    condition = torch.tensor([[.2, -.1, .5, .1], [.3, .1, -.2, .6]])
    projection = torch.arange(20).reshape(5, 4).float()/10
    amplitude = torch.tensor([1., 1.3])
    shift, half, error = latent_mean_shift(Location(), condition, [420800, 420801],
        projection, amplitude, members=32, seed=20261121)
    expected = condition@projection.T*amplitude[:, None]
    torch.testing.assert_close(shift, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(half, expected, atol=2e-6, rtol=0)
    assert error.abs().max() < 1e-6


def test_analytic_Gaussian_integrated_shift_is_zero():
    from postcoverage_analysis import latent_mean_shift
    from gnss_field.models.subspace import SubspaceEDM
    model = SubspaceEDM(rank=4, condition_dim=3, width=8).eval()
    shift, half, error = latent_mean_shift(model, torch.zeros(2, 3), [420800, 420801],
        torch.ones(5, 4), torch.ones(2), members=16, seed=20261121)
    assert shift.abs().max() < 2e-6 and half.abs().max() < 2e-6


def test_analysis_refuses_sealed_D2020(tmp_path):
    from postcoverage_analysis import load_evaluation
    (tmp_path/'manifest.json').write_text('{"role":"D","year":2020}')
    with pytest.raises(PermissionError):
        load_evaluation(tmp_path)
