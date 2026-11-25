"""Guard row-vector whitening, projective sampling and Gaussian attribution."""
import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).with_name('native_weather_gp_diffusion.py')
spec=importlib.util.spec_from_file_location('weather_diffusion',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_weather_gp_diffusion as module


def fixture():
    from .test_native_weather_gp import setup
    from .native_weather_gp import WeatherProjectiveGP
    c,h,st,ids,cp=setup();gp=WeatherProjectiveGP(265,cp)
    for net in (gp.node,gp.time):torch.nn.init.normal_(net[-1].weight,std=.02)
    K=gp.correlation(c,h,st,ids).detach();blocks=gp.query_time_blocks(c,h,st,ids).detach()
    return c,h,st,ids,gp,K,blocks


def test_batched_principal_roots_whiten_and_restore_heteroscedastic_rows():
    assert hasattr(module,'trajectory_factors'),'batched query-specific roots missing'
    _,_,_,_,_,K,blocks=fixture();root,inverse=module.trajectory_factors(blocks)
    torch.testing.assert_close(root@root,blocks,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(root@inverse,torch.eye(7,dtype=torch.float64).expand_as(root),atol=1e-10,rtol=1e-10)
    x=torch.randn(2,4,7,dtype=torch.float64);scale=torch.exp(x*.2)
    white=module.row_transform(x/scale,inverse)
    torch.testing.assert_close(module.row_transform(white,root)*scale,x,atol=1e-10,rtol=1e-10)
    with pytest.raises(ValueError,match='positive'):module.trajectory_factors(torch.ones(2,7,7))


def test_zero_velocity_inherits_gp_and_active_local_flow_preserves_common_path_subsets():
    assert hasattr(module,'transform_gp_path'),'GP-coupled local transformation missing'
    c,h,st,ids,gp,K,blocks=fixture();root,_=module.trajectory_factors(blocks)
    condition=module.condition265(c,h);model=module.make_model('local')
    eta=torch.randn(2,28,dtype=torch.float64)
    path=torch.einsum('bij,bj->bi',torch.linalg.cholesky(K),eta).reshape(2,4,7)
    result=module.transform_gp_path(model,condition,path,root,steps=20,solver='heun0')
    torch.testing.assert_close(result,path,atol=2e-6,rtol=2e-6)
    torch.nn.init.normal_(model.net[-1].weight,std=.02)
    full=module.transform_gp_path(model,condition,path,root,steps=20,solver='heun0')
    sub=torch.tensor([3,1]);smallroot,_=module.trajectory_factors(gp.query_time_blocks(c[:,sub],h,st[sub],ids[sub]).detach())
    small=module.transform_gp_path(model,condition[:,sub],path[:,sub],smallroot,steps=20,solver='heun0')
    torch.testing.assert_close(small,full[:,sub],atol=3e-6,rtol=3e-6)
    assert not torch.allclose(full,path,atol=1e-3,rtol=0)


def test_affine_physical_marginals_equal_direct_linear_map_and_mc():
    assert hasattr(module,'affine_marginals'),'conditional Gaussian physical marginals missing'
    c,h,_,_,_,_,blocks=fixture();root,_=module.trajectory_factors(blocks)
    c=module.condition265(c,h);model=module.make_model('affine')
    torch.nn.init.normal_(model.net[-1].weight,std=.01)
    mu=torch.randn(2,4,7);std=torch.exp(mu*.3)
    mean,sigma=module.affine_marginals(model,c,root,mu,std,20,'heun0')
    m,M=model.affine_map(c,20,'heun0')
    direct_factor=std.double()[...,None]*(root@M.double())
    torch.testing.assert_close(mean,mu.double()+std.double()*torch.einsum('bqij,bqj->bqi',root,m.double()),atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(sigma,direct_factor.square().sum(-1).sqrt(),atol=1e-9,rtol=1e-9)
    noise=torch.randn(40000,2,4,7,dtype=torch.float64)
    samples=mean[None]+torch.einsum('bqij,kbqj->kbqi',direct_factor,noise)
    torch.testing.assert_close(samples.mean(0),mean,atol=.025,rtol=.025)
    torch.testing.assert_close(samples.std(0),sigma,atol=.025,rtol=.025)


def test_masked_training_ignores_incomplete_trajectories_and_uses_real_loss():
    assert hasattr(module,'masked_loss'),'local frozen-GP masked objective missing'
    c,h,_,_,_,_,blocks=fixture();_,inverse=module.trajectory_factors(blocks)
    c=module.condition265(c,h);model=module.make_model('local')
    u=torch.randn(2,4,7);u[0,1,3]=torch.nan;u[1]=torch.nan
    loss,count,origins=module.masked_loss(model,u,c,inverse)
    assert count==3 and origins==1 and torch.isfinite(loss)
    loss.backward();assert model.net[-1].weight.grad.abs().sum()>0
    loss,count,origins=module.masked_loss(model,torch.full_like(u,torch.nan),c,inverse)
    assert count==origins==0 and loss==0
