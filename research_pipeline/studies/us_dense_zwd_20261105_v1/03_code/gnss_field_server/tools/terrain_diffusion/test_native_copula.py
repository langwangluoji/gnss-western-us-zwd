import torch
from .native_copula import spatial_correlation,correlate_noise


def test_gaussian_copula_has_unit_marginals_and_permutation_consistency():
    static=torch.zeros(4,9);static[:,:2]=torch.tensor([[0.,0.],[.05,0.],[.3,0.],[.8,.1]])
    correlation=spatial_correlation(static,.6)
    torch.testing.assert_close(correlation.diag(),torch.ones(4))
    assert torch.linalg.eigvalsh(correlation).min()>0
    order=torch.tensor([3,1,0,2])
    torch.testing.assert_close(spatial_correlation(static[order],.6),correlation[order][:,order])


def test_noise_factor_applies_registered_space_and_time_covariance():
    static=torch.zeros(2,9);static[1,0]=.2
    time=torch.tensor([[1.,.5],[.5,1.]])
    # Identity basis noises let us inspect the exact linear map rather than a Monte Carlo test.
    noise=torch.eye(4).reshape(4,2,2)
    transformed=correlate_noise(noise,static,.4,time).reshape(4,4)
    covariance=transformed.T@transformed
    expected=torch.kron(spatial_correlation(static,.4),time)
    torch.testing.assert_close(covariance,expected,atol=2e-6,rtol=2e-6)
