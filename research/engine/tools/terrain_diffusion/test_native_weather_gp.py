import importlib.util
from pathlib import Path
import pytest
import torch
from .native_copula import spatial_correlation

path=Path(__file__).with_name('native_weather_gp.py')
spec=importlib.util.spec_from_file_location('weather_gp',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_weather_gp as module


def setup():
    torch.manual_seed(19)
    st=torch.randn(4,9)*.1;st[1]=st[0]
    t=torch.arange(7);T=.95*.8**(t[:,None]-t[None]).abs()+.05*torch.eye(7)
    return torch.randn(2,4,175),torch.randn(2,72,5,15),st,torch.arange(4),dict(spatial_rho=.35,time_correlation=T.tolist())


def test_weather_summary_has_exact90_known_history_values_and_ignores_older_prefix():
    assert hasattr(module,'weather_summary'),'known-visible weather summary missing'
    _,history,_,_,_=setup();actual=module.weather_summary(history)
    clean=history.clone()
    clean[...,:3]=torch.where(history[...,3:6]>0,history[...,:3],0)
    clean[...,9:11]=torch.where(history[...,3:4]>0,history[...,9:11],0)
    expected=torch.cat([v for n in (1,24,72) for v in
        (clean[:,-n:].mean((1,2)),clean[:,-n:].std((1,2),unbiased=False))],-1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual,module.weather_summary(torch.cat([history[:,:7]*500,history],1)))
    assert actual.shape==(2,90)


def test_weather_summary_excludes_masked_measurements_and_innovation_placeholders():
    assert hasattr(module,'weather_summary'),'mask-aware visible weather summary missing'
    _,history,_,_,_=setup();history[...,3:6]=1
    history[:,-24:,1,3]=0;history[:,-12:,2,4:6]=0
    poisoned=history.clone();poisoned[:,-24:,1,0]=1e5;poisoned[:,-24:,1,9:11]=-1e5
    poisoned[:,-12:,2,1:3]=2e5
    torch.testing.assert_close(module.weather_summary(history),module.weather_summary(poisoned),rtol=0,atol=0)


@pytest.mark.parametrize('view,count',[(175,43893),(265,44027)])
def test_initial_kernel_inherits_reference_with_unit_marginals_and_spd(view,count):
    assert hasattr(module,'WeatherProjectiveGP'),'weather-conditioned PSD kernel missing'
    c,h,st,ids,cp=setup();model=module.WeatherProjectiveGP(view,cp)
    K=model.correlation(c,h,st,ids)
    expected=torch.kron(spatial_correlation(st,cp['spatial_rho']).double(),torch.tensor(cp['time_correlation'],dtype=torch.float64))
    torch.testing.assert_close(K,expected[None].expand(2,-1,-1),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(K.diagonal(dim1=-2,dim2=-1),torch.ones(2,28,dtype=torch.float64),atol=1e-12,rtol=0)
    assert torch.linalg.eigvalsh(K).min()>0
    assert sum(p.numel() for p in model.parameters())==count
    with pytest.raises(ValueError,match='unique'):
        model.correlation(c,h,st,torch.tensor([0,0,2,3]))


def test_learned_kernel_and_time_blocks_are_query_subset_consistent():
    assert hasattr(module,'WeatherProjectiveGP'),'query-consistent learned kernel missing'
    c,h,st,ids,cp=setup();model=module.WeatherProjectiveGP(265,cp)
    for net in (model.node,model.time):torch.nn.init.normal_(net[-1].weight,std=.025)
    K=model.correlation(c,h,st,ids);sub=torch.tensor([3,1,0]);flat=(sub[:,None]*7+torch.arange(7)).flatten()
    small=model.correlation(c[:,sub],h,st[sub],ids[sub])
    torch.testing.assert_close(small,K[:,flat][:,:,flat],atol=2e-6,rtol=2e-6)
    blocks=model.query_time_blocks(c,h,st,ids)
    for q in range(4):torch.testing.assert_close(blocks[:,q],K[:,q*7:(q+1)*7,q*7:(q+1)*7],atol=1e-12,rtol=0)
    assert torch.linalg.eigvalsh(K).min()>0


def test_covariance_nll_is_proper_and_initial_warp_rho_time_heads_get_gradients():
    assert hasattr(module,'WeatherProjectiveGP'),'differentiable GP objective missing'
    c,h,st,ids,cp=setup();model=module.WeatherProjectiveGP(265,cp)
    K=model.correlation(c,h,st,ids);u=torch.randn(2,28,dtype=torch.float64)
    loss=module.gaussian_nll(u,K)
    expected=-torch.distributions.MultivariateNormal(torch.zeros_like(u),K).log_prob(u).mean()/28-.5*torch.log(torch.tensor(2*torch.pi,dtype=torch.float64))
    torch.testing.assert_close(loss,expected,atol=1e-10,rtol=1e-10)
    loss.backward()
    gradient=model.node[-1].weight.grad
    assert torch.isfinite(gradient).all() and gradient[:3].abs().sum()>0 and gradient[3].abs().sum()>0
    assert torch.isfinite(model.time[-1].weight.grad).all() and model.time[-1].weight.grad.abs().sum()>0


def test_lead_specific_ranges_use_normalized_convolution_and_preserve_translation():
    assert hasattr(module.WeatherProjectiveGP,'spatial_kernel'),'lead-dependent Gaussian convolution kernel missing'
    c,h,st,ids,cp=setup();model=module.WeatherProjectiveGP(175,cp)
    with torch.no_grad():
        model.metric[0]=torch.tensor([-.7,.5,.2]);model.metric[6]=torch.tensor([.4,-.3,.1])
    K=model.correlation(c,h,st,ids)
    moved=st.clone();moved[:,:3]+=torch.tensor([1.,2.,.3])
    torch.testing.assert_close(K,model.correlation(c,h,moved,ids),atol=2e-6,rtol=2e-6)
    length=model.metric.double().exp();l0=length[0];l6=length[6]
    overlap=(2*l0*l6/(l0.square()+l6.square())).sqrt().prod()
    expected=(cp['spatial_rho']*overlap+1-cp['spatial_rho'])*cp['time_correlation'][0][6]
    torch.testing.assert_close(K[:,0,6],expected.expand(2),atol=1e-7,rtol=1e-7)
    loss=module.gaussian_nll(torch.randn(2,28,dtype=torch.float64),K);loss.backward()
    assert torch.isfinite(model.metric.grad).all() and model.metric.grad.abs().sum()>0
