import numpy as np
import pytest


def test_frozen_ridge_readout_matches_registered_formula():
    from gnss_field.generative.background import frozen_ridge_correction
    rng = np.random.default_rng(17)
    features = rng.normal(size=(9, 93)).astype('f4')
    center = rng.normal(size=93);std = rng.uniform(.3, 2., size=93)
    beta = rng.normal(size=94)
    row = dict(center=center.tolist(), std=std.tolist(), beta=beta.tolist())
    expected = 12. * ((features.astype('f8')-center)/std @ beta[:-1] + beta[-1])
    np.testing.assert_allclose(frozen_ridge_correction(features, row, 12.), expected, atol=1e-10)
    with pytest.raises(ValueError):
        frozen_ridge_correction(features[:, :90], row, 12.)


def test_frozen_moments_obey_missing_input_and_target_subsets():
    from gnss_field.generative.background import FrozenInnovationMoments
    rng = np.random.default_rng(4)
    axy = rng.normal(size=(7, 2))*50000;ah = rng.uniform(0, 3000, 7)
    xy = rng.normal(size=(11, 2))*50000;h = rng.uniform(0, 3000, 11)
    residual = rng.normal(size=7);valid = np.array([1, 0, 1, 1, 0, 1, 1], bool)
    full = FrozenInnovationMoments(xy, h, axy, ah)(residual, valid)
    subset = FrozenInnovationMoments(xy[[1, 7]], h[[1, 7]], axy, ah)(residual, valid)
    np.testing.assert_allclose(subset, full[[1, 7]], atol=1e-6)
    assert np.array_equal(FrozenInnovationMoments(xy, h, axy, ah)(residual, np.zeros(7, bool)), np.zeros((11, 24)))


def test_background_assets_are_verified_before_model_loading(tmp_path):
    from gnss_field.generative.background import FusionBackgroundAssets
    path = tmp_path/'weights.pt';path.write_bytes(b'changed')
    assets = FusionBackgroundAssets(network_checkpoint=str(path),
        geometry_coefficients=str(path), fusion_settings=str(path),
        expected_sha256={str(path): '0'*64})
    with pytest.raises(ValueError, match='hash'):
        assets.verify()
