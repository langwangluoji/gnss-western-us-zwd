import importlib
import importlib.util
import pytest
import numpy as np
import torch


def module():
    name=__package__+'.native_inducing_flow_experiment'
    assert importlib.util.find_spec(name) is not None,'formal inducing-flow runner is missing'
    return importlib.import_module(name)


def test_formal_budget_and_train_test_contract_are_explicit():
    m=module()
    m.require_budget(False,3000,3000,False,True)
    for args in [(True,3000,3000,False,True),(False,3000,2999,False,True),(False,3000,3000,False,False)]:
        with pytest.raises(ValueError):m.require_budget(*args)
    m.require_budget(False,3000,4,True,True)
    with pytest.raises(ValueError):m.require_budget(False,3000,5,True,True)


def test_coupled_draw_has_declared_linear_map_and_repeatable_rng():
    m=module();B,Q,D,K=2,3,8,20000
    torch.manual_seed(9);W=torch.randn(B,Q*7,D*7,dtype=torch.float64)*.02
    residual=torch.eye(Q*7,dtype=torch.float64)[None].expand(B,-1,-1)*.7
    torch.manual_seed(6);path,eta=m.coupled_draw(residual,W,K)
    assert eta.shape==(K,B,D,7) and path.shape==(K,B,Q,7)
    torch.manual_seed(6);again,other=m.coupled_draw(residual,W,K)
    torch.testing.assert_close(path,again,rtol=0,atol=0);torch.testing.assert_close(eta,other,rtol=0,atol=0)
    actual=path[:,0].double().flatten(1);cov=actual.T@actual/K
    torch.testing.assert_close(cov,residual[0]+W[0]@W[0].T,atol=.025,rtol=.025)
    cross=actual.T@eta[:,0].double().flatten(1)/K
    torch.testing.assert_close(cross,W[0],atol=.03,rtol=.03)


def test_geometry_pairs_cover_available_distance_strata_without_self_pairs():
    m=module();static=torch.zeros(8,9);static[:,0]=torch.tensor([0,.01,.02,.06,.12,.20,.40,.8])
    a=m.geometry_pairs(static,np.random.default_rng(2));b=m.geometry_pairs(static,np.random.default_rng(2))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert (a[:,0]<a[:,1]).all() and len(torch.unique(a,dim=0))==len(a)
    distance=(static[a[:,0],:2]-static[a[:,1],:2]).norm(dim=-1)*1000
    assert all(((distance>=lo)&(distance<hi)).any() for lo,hi in [(0,50),(50,150),(150,300),(300,float('inf'))])


def test_evaluation_rejects_partial_calendar_reordered_queries_or_changed_cache():
    m=module();protocol=dict(config=dict(seed=42,months=12,calendar_day=8))
    ids=np.broadcast_to(np.arange(137),(216,137)).copy()
    m.require_calendar(protocol,216,ids,42,'cache','cache')
    for p,n,q,s,a,e in [(dict(config=dict(seed=42,months=1,calendar_day=8)),216,ids,42,'cache','cache'),
        (protocol,18,ids[:18],42,'cache','cache'),(protocol,216,ids[:,::-1],42,'cache','cache'),
        (protocol,216,ids,17,'cache','cache'),(protocol,216,ids,42,'changed','cache')]:
        with pytest.raises(ValueError):m.require_calendar(p,n,q,s,a,e)


def test_affine_analytic_std_equals_complete_basis_enumeration():
    m=module();torch.manual_seed(30);B,Q,D,H=2,3,8,7;N=Q*H;J=D*H;K=N+J
    basis=torch.randn(B,N,J)*.03;residual=torch.eye(N,dtype=torch.float64)[None].expand(B,-1,-1)*.7
    eta=torch.zeros(K,B,D,H);eta[N:]=torch.eye(J).reshape(J,1,D,H)
    noise=torch.zeros(K,B,N);noise[:N]=torch.eye(N)[:,None]*.7**.5
    path=noise+torch.einsum('bnd,kbd->kbn',basis,eta.flatten(2))
    b=dict(condition=torch.randn(B,Q,265),global_condition=torch.randn(B,90),mean=torch.zeros(B,Q,H),
        std=torch.ones(B,Q,H),gp_path=path.reshape(K,B,Q,H),latent_noise=eta,basis=basis)
    for domain in ('time','frequency'):
        model=m.InducingLatentFlow('affine',domain)
        with torch.no_grad():model.flow.net[-1].weight.normal_(0,.01);model.query.net[-1].weight.normal_(0,.01)
        expected=model(**b).square().sum(0).sqrt()
        torch.testing.assert_close(m.affine_std(model,b,residual),expected.double(),rtol=1e-5,atol=2e-6)
