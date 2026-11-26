import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

path=Path(__file__).with_name('native_centered_shape.py')
spec=importlib.util.spec_from_file_location('centered_shape',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_centered_shape as module


@pytest.mark.parametrize('mode,count',[('scale',44239),('shape',44030),('both',43801)])
def test_initial_model_preserves_gaussian_and_fixed_mean(mode,count):
    assert hasattr(module,'CenteredShape'),'mean-preserving nonGaussian model missing'
    model=module.CenteredShape(mode);c=torch.randn(2,4,265);eta=torch.randn(16,2,4,7)
    torch.testing.assert_close(model.transform(c,eta),eta,atol=0,rtol=0)
    mean,std=model.moments(c)
    torch.testing.assert_close(mean,torch.zeros(2,4,7),atol=0,rtol=0)
    torch.testing.assert_close(std,torch.ones(2,4,7),atol=0,rtol=0)
    assert sum(p.numel() for p in model.parameters())==count


def test_active_coefficients_have_exact_zero_mean_known_variance_and_nonzero_skew():
    assert hasattr(module,'CenteredShape'),'Hermite normalization missing'
    model=module.CenteredShape('both').double();c=torch.randn(2,4,265,dtype=torch.float64)
    with torch.no_grad():
        model.net[-1].bias[:7]=.2;model.net[-1].bias[7:14]=.3;model.net[-1].bias[14:]=.15
    nodes,weights=np.polynomial.hermite.hermgauss(16)
    eta=torch.tensor(nodes*2**.5)[:,None,None,None].expand(-1,2,4,7)
    w=torch.tensor(weights/np.sqrt(np.pi))[:,None,None,None]
    x=model.transform(c,eta);mean,std=model.moments(c)
    torch.testing.assert_close((w*x).sum(0),mean,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close((w*x.square()).sum(0),std.square(),atol=1e-12,rtol=1e-12)
    assert (w*x.pow(3)).sum(0).abs().min()>.1
    sub=torch.tensor([3,1]);torch.testing.assert_close(model.transform(c[:,sub],eta[:,:,sub]),x[:,:,sub],atol=1e-12,rtol=1e-12)


def test_fair_crps_uses_distinct_member_pairs_and_preserves_missing_labels():
    assert hasattr(module,'fair_crps_torch'),'differentiable fair CRPS missing'
    x=torch.tensor([[0.,2.],[2.,6.]],requires_grad=True);target=torch.tensor([1.,torch.nan])
    score=module.fair_crps_torch(x,target)
    torch.testing.assert_close(score[0],torch.tensor(0.))
    assert torch.isnan(score[1])
    with pytest.raises(ValueError):module.fair_crps_torch(x[:1],target)


def test_masked_shape_loss_has_gradients_for_new_heads_and_empty_labels():
    assert hasattr(module,'masked_loss'),'masked centered-shape loss missing'
    model=module.CenteredShape('both');c=torch.randn(2,4,265);target=torch.ones(2,4,7)*2
    target[0,1,3]=torch.nan;gs=torch.ones_like(target)
    loss,count=module.masked_loss(model,target,c,gs,16)
    assert count==55 and torch.isfinite(loss)
    loss.backward();g=model.net[-1].weight.grad
    assert torch.isfinite(g).all() and all(g[i:i+7].abs().sum()>0 for i in (0,7,14))
    loss,count=module.masked_loss(model,torch.full_like(target,torch.nan),c,gs,16)
    assert count==0 and loss==0
