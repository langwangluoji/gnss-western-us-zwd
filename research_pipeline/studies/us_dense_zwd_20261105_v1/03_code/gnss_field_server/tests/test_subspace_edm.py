"""Observable-subspace EDM must preserve the frozen Gaussian starting point."""
import numpy as np
import torch


def test_calibrated_initial_sampler_is_exact_standard_normal_path():
    from gnss_field.models.subspace import SubspaceEDM
    torch.manual_seed(42)
    model = SubspaceEDM(7, 3, width=16).double()
    noise = torch.randn(2, 4, 7, dtype=torch.float64)
    condition = torch.randn(2, 3, dtype=torch.float64)
    actual = model.sample_noise(condition, noise)
    torch.testing.assert_close(actual, noise, rtol=1e-12, atol=1e-12)
    assert model.gaussian_gain.item() > 0


def test_sampler_backward_reaches_trainable_head_through35_calls():
    from gnss_field.models.subspace import SubspaceEDM
    model = SubspaceEDM(7, 3, width=16)
    calls = []
    original = model.denoise
    def record(*args):
        calls.append(1)
        return original(*args)
    model.denoise = record
    prediction = model.sample_noise(torch.randn(2, 3), torch.randn(2, 4, 7))
    prediction.square().mean().backward()
    assert len(calls) == 35
    assert torch.isfinite(model.net[-1].weight.grad).all()
    assert model.net[-1].weight.grad.abs().sum() > 0


def test_station_assembly_equals_query_of_common_grid_realizations():
    from gnss_field.models.subspace import station_members
    torch.manual_seed(1)
    phi = torch.randn(15, 9, dtype=torch.float64)
    q = torch.randn(4, 15, dtype=torch.float64)
    v = torch.linalg.qr(torch.randn(9, 3, dtype=torch.float64)).Q.T
    h = q@phi
    projection, remainder = h@v.T, h-(h@v.T)@v
    z, eps, eta = [torch.randn(2, 5, n, dtype=torch.float64) for n in (3, 9, 4)]
    base = torch.randn(2, 4, dtype=torch.float64)
    g = torch.tensor([.8, 1.4], dtype=torch.float64)
    actual = station_members(base, z, eps, eta, projection, remainder, g, 3.)
    coefficients = z@v+eps-(eps@v.T)@v
    expected = base[:, None]+g[:, None, None]*(coefficients@phi.T@q.T+3.*eta)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_training_only_pca_ignores_changed_development_conditions():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    from local_subspace_data import fit_condition_pca, transform_condition
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, 12))
    features[:40, 0] = 3.
    mask = np.arange(60) < 40
    first = fit_condition_pca(features, mask, 5)
    changed = features.copy()
    changed[~mask] *= 1e6
    second = fit_condition_pca(changed, mask, 5)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.isfinite(transform_condition(changed, first)).all()
