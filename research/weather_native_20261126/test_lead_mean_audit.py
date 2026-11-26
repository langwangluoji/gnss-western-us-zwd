import numpy as np
import pytest

from lead_mean_audit import mean_centers, move_center, verified_geometry


def test_hybrid_center_uses_only_known_lead_and_keeps_sources_unchanged():
    base=np.arange(14,dtype='float32').reshape(2,7);gaussian=base+3
    centers=mean_centers(base,gaussian)
    np.testing.assert_array_equal(centers['hybrid'][...,0],base[...,0])
    np.testing.assert_array_equal(centers['hybrid'][...,1:],gaussian[...,1:])
    np.testing.assert_array_equal(centers['base'],base)
    np.testing.assert_array_equal(centers['gaussian'],gaussian)
    centers['hybrid'][0,0]=-10
    assert base[0,0]==0 and gaussian[0,0]==3


def test_pure_shift_preserves_each_member_residual_and_full_sample_covariance():
    rng=np.random.default_rng(8);x=rng.normal(size=(16,7));old=np.arange(7.);new=old+np.linspace(-3,4,7)
    y=move_center(x+old,old,new)
    np.testing.assert_allclose(y-new,x,atol=2e-15)
    np.testing.assert_allclose(np.cov(y,rowvar=False),np.cov(x,rowvar=False),atol=2e-15)


def test_center_recipe_and_shapes_fail_closed():
    with pytest.raises(ValueError,match='seven'):
        mean_centers(np.zeros((2,6)),np.zeros((2,6)))
    with pytest.raises(ValueError,match='shape'):
        mean_centers(np.zeros((2,7)),np.zeros((3,7)))
    with pytest.raises(ValueError,match='finite'):
        mean_centers(np.zeros((2,7)),np.full((2,7),np.nan))
    with pytest.raises(ValueError,match='shape'):
        move_center(np.zeros((16,3,7)),np.zeros((2,7)),np.zeros((2,7)))


def test_geometry_must_match_frozen_manifest_before_distance_scoring(tmp_path):
    import hashlib,json
    path=tmp_path/'geometry.json';path.write_text(json.dumps({'dev_static':np.zeros((137,9)).tolist()}),encoding='utf-8')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'geometry_sha256':digest}),encoding='utf-8')
    actual,registered=verified_geometry(tmp_path)
    assert actual.shape==(137,9) and registered==digest
    path.write_text(json.dumps({'dev_static':np.ones((137,9)).tolist()}),encoding='utf-8')
    with pytest.raises(ValueError,match='geometry'):
        verified_geometry(tmp_path)
