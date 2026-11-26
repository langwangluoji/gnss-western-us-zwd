import importlib
import importlib.util
import torch
import pytest


def module():
    name=__package__+'.native_weather_calibration'
    assert importlib.util.find_spec(name) is not None,'weather calibration controls missing'
    return importlib.import_module(name)


def test_matched_parameters_and_exact_covariance_warm_start():
    m=module();x=torch.randn(3,90);torch.manual_seed(2);g=m.WeatherCalibrator('scale')
    torch.manual_seed(2);t=m.WeatherCalibrator('tail')
    assert sum(p.numel() for p in g.parameters())==sum(p.numel() for p in t.parameters())
    torch.testing.assert_close(g(x),torch.ones(3),rtol=0,atol=0)
    torch.testing.assert_close(t(x),torch.full((3,),20.),rtol=1e-6,atol=1e-6)
    with pytest.raises(ValueError):g(torch.randn(3,91))


def test_scaled_log_density_uses_double_dimension_and_mask_only_for_scores():
    m=module();from .native_student_process import log_density
    q=torch.tensor([12.,float('nan'),8.],dtype=torch.float64);ld=torch.tensor([3.,float('nan'),2.],dtype=torch.float64)
    d=torch.tensor([5,0,3]);scale=torch.tensor([1.23456789,1.2,.9],dtype=torch.float64,requires_grad=True)
    loss,count=m.loss(q,ld,d,scale)
    expected=(-log_density(q[[0,2]]/scale[[0,2]],ld[[0,2]]+d[[0,2]].double()*scale[[0,2]].log(),d[[0,2]])/d[[0,2]]).mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0);assert count==2
    loss.backward();assert torch.isfinite(scale.grad).all() and scale.grad[1]==0


def test_empty_block_has_zero_finite_gradient_without_fabricated_labels():
    m=module();s=torch.ones(2,requires_grad=True);v=torch.full((2,),float('nan'));d=torch.zeros(2,dtype=torch.long)
    value,count=m.loss(v,v,d,s);assert count==0 and value==0
    value.backward();torch.testing.assert_close(s.grad,torch.zeros_like(s))


def test_full_field_mixture_warm_output_and_query_subset_share_radial_noise():
    m=module();torch.manual_seed(9);x=torch.randn(32,2,3,7);mu=torch.randn(2,3,7);s=torch.ones(2)
    torch.testing.assert_close(m.transform_ensemble(x,mu,s),x,rtol=0,atol=0)
    full=m.transform_ensemble(x,mu,s,torch.tensor([8.,20.]),31)
    ids=torch.tensor([2,0]);small=m.transform_ensemble(x[:,:,ids],mu[:,ids],s,torch.tensor([8.,20.]),31)
    torch.testing.assert_close(small,full[:,:,ids],rtol=0,atol=0)


def test_student_scale_mixture_preserves_population_covariance():
    m=module();torch.manual_seed(8);K=100000
    C=torch.tensor([[2.,.6],[.6,1.]],dtype=torch.float64)
    x=(torch.randn(K,2,dtype=torch.float64)@torch.linalg.cholesky(C).T).reshape(K,1,2,1)
    y=m.transform_ensemble(x,torch.zeros(1,2,1,dtype=torch.float64),torch.tensor([1.5],dtype=torch.float64),torch.tensor([8.]),44).flatten(1)
    torch.testing.assert_close(y.mean(0),torch.zeros(2,dtype=torch.float64),rtol=0,atol=.03)
    torch.testing.assert_close(y.T@y/K,1.5*C,rtol=.03,atol=.025)
