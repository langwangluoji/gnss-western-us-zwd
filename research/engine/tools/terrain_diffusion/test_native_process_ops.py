import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).with_name('native_process_ops.py')
spec=importlib.util.spec_from_file_location('native_process_ops',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_process_ops as module


def setup():
    st=torch.zeros(4,9);st[2,:3]=torch.tensor([.1,-.1,.5]);st[3,:3]=torch.tensor([.5,.4,.8])
    t=torch.arange(7);Kt=.95*.8**(t[:,None]-t[None]).abs()+.05*torch.eye(7)
    return st,dict(spatial_rho=.8,time_correlation=Kt.tolist())


def test_principal_factors_and_heteroscedastic_restore_include_repeated_coordinates():
    assert hasattr(module,'principal_factors'), 'process covariance operators are missing'
    st,cp=setup();As,Ai,_=module.space_factors(st,cp,torch.arange(4));At,Ati,_=module.time_factors(cp)
    x=torch.randn(3,4,7);mu=torch.randn_like(x);sigma=torch.exp(torch.randn_like(x)*.2)
    z=module.whiten((x-mu)/sigma,Ai,Ati)
    restored=mu+sigma*module.color(z,As,At)
    torch.testing.assert_close(restored,x,rtol=2e-5,atol=2e-5)
    identity=module.principal_factors(torch.ones(1,1))[0]
    torch.testing.assert_close(identity,torch.ones(1,1))
    with pytest.raises(ValueError,match='positive'):
        module.principal_factors(torch.tensor([[1.,2.],[2.,1.]]))
    with pytest.raises(ValueError,match='unique'):
        module.space_factors(st,cp,torch.tensor([1,1,2,3]))


def test_principal_space_transform_is_station_permutation_equivariant():
    assert hasattr(module,'space_factors'), 'canonical covariance root is missing'
    st,cp=setup();order=torch.tensor([3,1,0,2]);ids=torch.arange(4)
    As,_,_=module.space_factors(st,cp,ids);Ps,_,_=module.space_factors(st[order],cp,ids[order])
    torch.testing.assert_close(Ps,As[order][:,order],atol=2e-6,rtol=2e-6)


def test_heun0_never_calls_the_untrained_endpoint_and_preserves_zero_velocity():
    assert hasattr(module,'heun0'), 'analytic endpoint sampler is missing'
    class Model:
        alpha=torch.linspace(1,0,101)
        calls=[]
        def denoise(self,x,c,k,U,eigen):
            assert (k>0).all(), 'k0 must be analytic'
            self.calls.append(k.clone());return torch.zeros_like(x)
    model=Model();noise=torch.randn(2,4,7)
    result=module.heun0(model,torch.zeros(2,4,3),noise,steps=20)
    torch.testing.assert_close(result,noise,rtol=0,atol=0)
    assert len(model.calls)==39


def test_missing_trajectory_loss_whitens_only_observed_subsets():
    assert hasattr(module,'masked_process_loss'), 'masked process objective is missing'
    st,cp=setup();u=torch.randn(3,4,7);u[0,1,3]=torch.nan;u[1,1:3,2]=torch.nan;u[2]=torch.nan
    c=torch.randn(3,4,5);saved=c.clone()
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.ones(()))
        def loss(self,x,condition,U,eigen):
            assert torch.isfinite(x).all();return self.p*x.square().mean()
    model=Model();loss,count,origins=module.masked_process_loss(model,u,c,st,cp,'full',torch.arange(4))
    _,Ati,_=module.time_factors(cp);expected=0
    for i in (0,1):
        keep=torch.isfinite(u[i]).all(-1);_,Ai,_=module.space_factors(st[keep],cp,torch.arange(4)[keep])
        expected+=module.whiten(u[i:i+1,keep],Ai,Ati).square().mean()*int(keep.sum())
    torch.testing.assert_close(loss,expected/5,atol=2e-6,rtol=2e-6)
    assert count==5 and origins==2
    torch.testing.assert_close(c,saved,rtol=0,atol=0)
    loss.backward();assert torch.isfinite(model.p.grad)
