import importlib
import importlib.util
import numpy as np
import pytest
import torch
from scipy.stats import multivariate_t


def module():
    name=__package__+'.native_student_process'
    assert importlib.util.find_spec(name) is not None,'covariance-matched Student process is missing'
    return importlib.import_module(name)


def test_covariance_matched_density_agrees_with_scipy_and_gaussian_limit():
    m=module();torch.manual_seed(4);d=5
    a=torch.randn(d,d,dtype=torch.float64);C=a@a.T+torch.eye(d,dtype=torch.float64)
    x=torch.randn(3,d,dtype=torch.float64);nu=torch.tensor(7.,dtype=torch.float64,requires_grad=True)
    q,ld,dimension=m.masked_radius(x,C[None].expand(3,-1,-1))
    actual=m.log_density(q,ld,dimension,nu)
    expected=multivariate_t.logpdf(x.numpy(),loc=np.zeros(d),shape=C.numpy()*5/7,df=7)
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=1e-12,rtol=1e-12)
    actual.sum().backward();assert torch.isfinite(nu.grad) and nu.grad!=0
    torch.testing.assert_close(m.log_density(q,ld,dimension,1e7),m.log_density(q,ld,dimension),atol=2e-5,rtol=0)
    with pytest.raises(ValueError):m.log_density(q,ld,dimension,2.)


def test_masked_radius_uses_observed_submatrix_and_preserves_empty_origins():
    m=module();C=torch.tensor([[2.,.5,.3],[.5,3.,.2],[.3,.2,4.]],dtype=torch.float64)
    error=torch.tensor([[1.,float('nan'),-2.],[float('nan')]*3],dtype=torch.float64)
    q,ld,d=m.masked_radius(error,C[None].expand(2,-1,-1))
    indices=torch.tensor([0,2]);sub=C[indices][:,indices];v=error[0,indices]
    torch.testing.assert_close(q[0],v@torch.linalg.solve(sub,v),atol=1e-12,rtol=0)
    torch.testing.assert_close(ld[0],torch.linalg.slogdet(sub).logabsdet,atol=1e-12,rtol=0)
    assert d.tolist()==[2,0] and torch.isnan(q[1]) and torch.isnan(ld[1])


def test_full_affine_covariance_equals_complete_basis_enumeration():
    m=module();from .native_inducing_flow import InducingLatentFlow
    torch.manual_seed(18);B,Q,D,H=2,3,8,7;N=Q*H;J=D*H;K=N+J
    basis=torch.randn(B,N,J)*.03;residual=torch.eye(N,dtype=torch.float64)[None].expand(B,-1,-1)*.7
    eta=torch.zeros(K,B,D,H);eta[N:]=torch.eye(J).reshape(J,1,D,H)
    epsilon=torch.zeros(K,B,N);epsilon[:N]=torch.eye(N)[:,None]*.7**.5
    path=epsilon+torch.einsum('bnd,kbd->kbn',basis,eta.flatten(2))
    b=dict(condition=torch.randn(B,Q,265),global_condition=torch.randn(B,90),mean=torch.zeros(B,Q,H),
        std=torch.rand(B,Q,H)+1,gp_path=path.reshape(K,B,Q,H),latent_noise=eta,basis=basis)
    for domain in ('time','frequency'):
        model=InducingLatentFlow('affine',domain)
        with torch.no_grad():model.flow.net[-1].weight.normal_(0,.01);model.query.net[-1].weight.normal_(0,.01)
        output=model(**b).flatten(2).double();expected=torch.einsum('kbi,kbj->bij',output,output)
        torch.testing.assert_close(m.affine_covariance(model,b,residual),expected,atol=2e-6,rtol=1e-5)
