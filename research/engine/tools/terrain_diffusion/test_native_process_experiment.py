import importlib.util
from pathlib import Path
import copy
import pytest
import torch

path=Path(__file__).with_name('native_process_experiment.py')
spec=importlib.util.spec_from_file_location('process_runner',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_process_experiment as module


def test_budget_and_source_guard_allow_only_registered_training_or_four_update_smoke():
    assert hasattr(module,'validate_training'), 'independent process runner missing'
    source=dict(data_sha256='a',source_sha256='b',I1_sha256='c')
    module.validate_training(False,3000,3000,False,True,source,source)
    module.validate_training(False,3,4,True,False,source,source)
    cases=[(True,3000,3000,False,True),(False,3,5,True,False),
        (False,3000,3000,False,False),(False,3,4,False,True)]
    for dev,count,steps,smoke,same in cases:
        with pytest.raises(ValueError):module.validate_training(dev,count,steps,smoke,same,source,source)
    changed=copy.deepcopy(source);changed['I1_sha256']='different'
    with pytest.raises(ValueError):module.validate_training(False,3,4,True,False,source,changed)


def test_restore_does_not_commute_heteroscedastic_scale_through_time():
    assert hasattr(module,'restore_mm'), 'physical restoration missing'
    x=torch.randn(2,3,7);base=torch.randn_like(x);bias=torch.randn(7);scale=torch.rand(7)+1
    gm=torch.randn_like(x);gs=torch.rand_like(x)+.1
    restored=module.restore_mm(x,base,bias,scale,gm,gs)
    torch.testing.assert_close(restored,base+bias+scale*(gm+gs*x))


def test_affine_analytic_marginals_match_direct_factor_and_mc():
    assert hasattr(module,'affine_marginals'), 'analytic affine Gaussian control missing'
    torch.manual_seed(17)
    m=torch.randn(2,3,7);M=torch.randn(2,3,7,7)*.15
    t=torch.arange(7);K=.8**(t[:,None]-t[None]).abs()
    from .native_process_ops import principal_factors
    At=principal_factors(K)[0]
    base=torch.randn_like(m);bias=torch.randn(7);scale=torch.rand(7)+1;gm=torch.randn_like(m);gs=torch.rand_like(m)+.1
    mean,std=module.affine_marginals(m,M,At,base,bias,scale,gm,gs)
    factor=(scale*gs)[...,None]*(At@M)
    torch.testing.assert_close(std,factor.square().sum(-1).sqrt())
    z=torch.randn(60000,2,3,7)
    u=(m+torch.einsum('bqij,kbqj->kbqi',M,z))@At
    samples=module.restore_mm(u,base,bias,scale,gm,gs)
    torch.testing.assert_close(samples.mean(0),mean,atol=.025,rtol=.015)
    torch.testing.assert_close(samples.std(0),std,atol=.015,rtol=.02)
