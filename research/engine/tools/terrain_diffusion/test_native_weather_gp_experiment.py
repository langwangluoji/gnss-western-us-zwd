import importlib.util
from pathlib import Path
import torch
from .test_native_weather_gp import setup
from .native_weather_gp import WeatherProjectiveGP,gaussian_nll

path=Path(__file__).with_name('native_weather_gp_experiment.py')
spec=importlib.util.spec_from_file_location('gp_experiment',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_weather_gp_experiment as module


def test_gp_supervision_uses_observed_marginals_without_contaminating_conditions():
    assert hasattr(module,'masked_gp_nll'),'masked weather GP runner missing'
    c,h,st,ids,cp=setup();model=WeatherProjectiveGP(265,cp)
    u=torch.randn(2,4,7);u[0,1,3]=torch.nan;u[1,2,1]=torch.nan
    saved=c.clone();loss,count,origins=module.masked_gp_nll(model,u,c,h,st,ids)
    expected=0
    for i in range(2):
        keep=torch.isfinite(u[i]).all(-1)
        R=model.correlation(c[i:i+1,keep],h[i:i+1],st[keep],ids[keep])
        expected=expected+gaussian_nll(u[i:i+1,keep].reshape(1,-1),R)/2
    torch.testing.assert_close(loss,expected,atol=1e-10,rtol=1e-10)
    assert count==6 and origins==2
    torch.testing.assert_close(c,saved,atol=0,rtol=0)
    loss.backward();assert torch.isfinite(model.node[-1].weight.grad).all()
    empty,n,b=module.masked_gp_nll(model,u*torch.nan,c,h,st,ids)
    assert n==b==0 and empty==0 and torch.isfinite(empty)


def test_gaussian_linear_contrast_marginals_match_actual_joint_draws():
    assert hasattr(module,'contrast_marginals'),'analytic Gaussian station/increment contrasts missing'
    c,h,st,ids,cp=setup();model=WeatherProjectiveGP(265,cp)
    R=model.correlation(c,h,st,ids).detach();torch.manual_seed(74)
    mean=torch.randn(2,28,dtype=torch.float64);std=torch.rand(2,28,dtype=torch.float64)+.2
    left=torch.tensor([0,1,2,3]);right=torch.tensor([7,8,3,4])
    am,ast=module.contrast_marginals(mean,std,R,left,right)
    noise=torch.randn(50000,2,28,dtype=torch.float64)
    samples=mean+std*torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(R),noise)
    diff=samples[:,:,left]-samples[:,:,right]
    torch.testing.assert_close(diff.mean(0),am,atol=.02,rtol=.01)
    torch.testing.assert_close(diff.std(0),ast,atol=.02,rtol=.02)
