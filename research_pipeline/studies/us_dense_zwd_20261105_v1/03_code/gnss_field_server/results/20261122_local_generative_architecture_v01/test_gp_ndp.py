import numpy as np
import torch
import pytest


def test_gaussian_correlation_noise_has_registered_covariance():
    from gp_ndp import correlation_factor
    rng=np.random.default_rng(4);a=rng.normal(size=(7,3));b=rng.normal(size=(7,5));tau=.7
    covariance=a@a.T+b@b.T+tau**2*np.eye(7);std=np.sqrt(np.diag(covariance));expected=covariance/std[:,None]/std[None]
    L=correlation_factor(a,b,tau)
    assert np.max(abs(L@L.T-expected))<2e-7
    x=torch.randn(2,4,7);whole=x@torch.tensor(L.T)
    assert torch.allclose(whole[:,:2],x[:,:2]@torch.tensor(L.T),atol=1e-6)
    with pytest.raises(ValueError):correlation_factor(a,b,-1.)


def test_sampling_audit_checks_frozen_checkpoint(tmp_path):
    from sampling_stability import audit_dependencies
    from gnss_field.artifacts import write_json,sha256
    (tmp_path/'pilot_models.py').write_text('frozen model');(tmp_path/'GP_correlation_factor.npy').write_bytes(b'factor')
    write_json(tmp_path/'data.lock.json',dict(data='fixed'))
    directory=tmp_path/'fold_0/N-set';directory.mkdir(parents=True)
    checkpoint=directory/'selected.pt';checkpoint.write_bytes(b'frozen checkpoint')
    write_json(directory/'result.json',dict(checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint)))
    write_json(directory/'lock.json',dict(model=sha256(tmp_path/'pilot_models.py')))
    bound=audit_dependencies(tmp_path,['fold_0/N-set']);assert bound['artifacts']['fold_0/N-set']['checkpoint']==sha256(checkpoint)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='checkpoint'):audit_dependencies(tmp_path,['fold_0/N-set'])
