import torch


def test_energy_score_u_statistic_matches_explicit_distances_and_has_gradient():
    from gnss_field.objectives import energy_score
    x = torch.tensor([[[0., 0.], [2., 0.], [0., 2.], [2., 2.]]], requires_grad=True)
    y = torch.tensor([[1., 1.]])
    first = (x[0]-y[0]).norm(dim=-1).mean()
    pair = sum((x[0, i]-x[0, j]).norm() for i in range(4) for j in range(4) if i != j)/24
    expected = (first-pair)/(2**.5)
    torch.testing.assert_close(energy_score(x, y), expected)
    energy_score(x, y).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_float64_ridge_is_differentiable_and_normalization_reversible():
    from gnss_field.objectives import fit_latent, CoefficientTransform
    torch.manual_seed(2)
    h = torch.randn(40, 8, dtype=torch.float64, requires_grad=True)
    truth = torch.randn(8, dtype=torch.float64)
    d = h.detach() @ truth
    c, diagnostic = fit_latent(h, d)
    assert c.dtype == torch.float64
    assert (h@c-d).square().mean() < 1e-4
    c.square().sum().backward()
    assert torch.isfinite(h.grad).all()
    values = torch.randn(20, 8)
    t = CoefficientTransform.fit(values)
    torch.testing.assert_close(t.decode(t.encode(values)), values)
    torch.testing.assert_close(t.encode(values).square().mean(0).sqrt(), torch.full((8,), .5))


def test_nonfinite_scores_fail_instead_of_deleting_predictions():
    import pytest
    from gnss_field.objectives import energy_score
    x = torch.ones(1, 4, 5)
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        energy_score(x, torch.ones(1, 5))
def test_unbiased_ES_ignores_numerical_self_distance(monkeypatch):
    import torch
    from gnss_field.objectives.scores import energy_score
    torch.manual_seed(42)
    samples = torch.randn(2, 4, 7)
    truth = torch.randn(2, 7)
    expected = energy_score(samples, truth)
    original = torch.cdist
    def rounded_self_distances(x, y, **kwargs):
        return original(x, y, **kwargs) + .25*torch.eye(x.shape[-2], device=x.device)
    monkeypatch.setattr(torch, 'cdist', rounded_self_distances)
    torch.testing.assert_close(energy_score(samples, truth), expected, rtol=0, atol=0)

def test_K128_ES_with_large_common_mean_matches_FP64_reference():
    from gnss_field.objectives.scores import energy_score
    torch.manual_seed(42)
    samples = 100000.+torch.randn(1, 128, 12)
    truth = 100000.+torch.randn(1, 12)
    torch.testing.assert_close(energy_score(samples, truth),
        energy_score(samples.double(), truth.double()).float(), rtol=2e-5, atol=2e-5)

