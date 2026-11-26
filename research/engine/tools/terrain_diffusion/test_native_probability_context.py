import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).with_name('native_probability_context.py')
spec=importlib.util.spec_from_file_location('native_probability_context',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_probability_context as module


def test_raw_query_information_is_preserved_exactly_and_respects_query_reordering():
    assert hasattr(module,'condition_view'), 'probability condition views are missing'
    e=torch.randn(2,5,64);raw=torch.randn(2,5,111)
    assert module.condition_view(e,raw,'latent') is e
    assert module.condition_view(e,raw,'raw') is raw
    c=module.condition_view(e,raw,'skip')
    torch.testing.assert_close(c[:,:,:64],e);torch.testing.assert_close(c[:,:,64:],raw)
    index=torch.tensor([4,1,0,2,3])
    torch.testing.assert_close(module.condition_view(e[:,index],raw[:,index],'skip'),c[:,index])


def test_default_factory_is_identical_and_skip_gaussian_has_matched_capacity_control():
    assert hasattr(module,'make_probability'), 'dimension-aware probability factory is missing'
    from .native_frequency_tokens import probability_model
    for domain in ('gaussian','spatial','graph','joint','frequency_tokens'):
        torch.manual_seed(25);old=probability_model(domain)
        torch.manual_seed(25);new=module.make_probability(domain)
        assert old.state_dict().keys()==new.state_dict().keys()
        for name in old.state_dict():torch.testing.assert_close(old.state_dict()[name],new.state_dict()[name],rtol=0,atol=0)
    skip=module.make_probability('gaussian',175,64);wide=module.make_probability('gaussian',64,154)
    assert sum(p.numel() for p in skip.parameters())==12174
    assert sum(p.numel() for p in wide.parameters())==12180
    assert skip(torch.randn(2,3,175))[0].shape==(2,3,7)


def test_preconditioner_defaults_remain_compatible_and_reject_different_information():
    assert hasattr(module,'require_condition_view'), 'condition source guard is missing'
    module.require_condition_view({'config':{}},'latent')
    with pytest.raises(ValueError,match='condition view'):
        module.require_condition_view({'config':{'condition_view':'raw'}},'skip')
    model=module.gaussian_from_protocol({'config':{'condition_view':'skip','width':64}})
    assert model(torch.randn(1,2,175))[0].shape==(1,2,7)


@pytest.mark.parametrize('domain',['spatial','joint','frequency_tokens'])
def test_full_native_diffusion_accepts_skip_information_without_projection(domain):
    model=module.make_probability(domain,175,16)
    x=torch.randn(2,5,7);c=torch.randn(2,5,175);U=torch.eye(5);eigen=torch.linspace(0,2,5)
    loss=model.loss(x,c,U,eigen)
    assert torch.isfinite(loss)
    sample=model.sample(c,torch.randn_like(x),U,eigen,steps=2)
    assert sample.shape==x.shape and torch.isfinite(sample).all()
