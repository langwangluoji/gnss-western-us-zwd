import numpy as np
import torch
from tools.terrain_diffusion.residual_field import spatial_folds, Foundation, common_shift, masked_mse, ResidualDenoiser


def test_group_fold_isolation():
    geo=np.array([[0,0],[10,0],[210,0],[220,0]],dtype=float)
    groups=np.array([1,1,2,2]); f=spatial_folds(geo,groups)
    assert f[0]==f[1] and f[2]==f[3]


def test_mask_and_common_shift():
    p=torch.tensor([1.,2.],requires_grad=True); y=torch.tensor([2.,float('nan')]); m=torch.tensor([True,False])
    loss=masked_mse(p,y,m);loss.backward();assert torch.isfinite(loss) and p.grad[1]==0
    x=np.arange(24.).reshape(3,2,4);z=common_shift(x,np.ones((2,4)))
    np.testing.assert_allclose(z-z.mean(0),x-x.mean(0))


def test_foundation_query_invariant():
    for kind in ('mlp','cnn_lstm'):
        torch.manual_seed(2);model=Foundation(kind).eval();s=torch.randn(1,7,41);h=torch.randn(1,7,12,24)
        with torch.no_grad():
            all=model(s,h);part=model(s[:,:3],h[:,:3])
        torch.testing.assert_close(all[:,:3],part,atol=1e-6,rtol=1e-5)


def test_residual_denoiser_gradient():
    model=ResidualDenoiser();x=torch.randn(1,8);f=torch.randn(1,8,114);g=torch.randn(1,8,4)
    y=model(x,torch.tensor([12]),f,g);y.square().mean().backward()
    assert y.shape==x.shape and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_padded_queries_excluded_from_diffusion_sampling():
    from types import SimpleNamespace
    from tools.terrain_diffusion.residual_experiment import probability_predict
    torch.manual_seed(7); model=ResidualDenoiser().cuda().eval()
    rng=np.random.default_rng(8);f=rng.normal(size=(1,5,114)).astype('f4');g=rng.normal(size=(1,5,4)).astype('f4');b=np.zeros((1,5),dtype='f4')
    data=SimpleNamespace(mean=np.array([0.]),std=np.array([1.]))
    a=probability_predict(model,'diffusion',f,g,b,1.,data,2,9,query_valid=np.array([[True,True,True,False,False]]))
    c=probability_predict(model,'diffusion',f[:,:3],g[:,:3],b[:,:3],1.,data,2,9)
    np.testing.assert_allclose(a[:,:,:3],c,atol=1e-5)


def test_sync_failure_does_not_abort_training(tmp_path,monkeypatch):
    from tools.terrain_diffusion import residual_queue as q
    def fail(*a):raise RuntimeError('network unavailable')
    monkeypatch.setattr(q,'sync',fail)
    assert q.sync_safely(tmp_path,'test') is False
    assert (tmp_path/'sync_pending.json').exists()


def test_tree_rejects_empty_monitor(tmp_path):
    import pytest
    from tools.terrain_diffusion.residual_experiment import fit_tree
    with pytest.raises(ValueError,match='empty fit or monitor'):
        fit_tree(np.zeros((2,3,2)),np.zeros((2,3)),np.ones((2,3),bool),np.zeros(2,bool),tmp_path,1,42)
