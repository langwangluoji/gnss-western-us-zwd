import torch
import pytest
from types import SimpleNamespace

from .native_mean_hs import NativeMeanHS
from .native_mean_fusion_v2 import NativeMeanFusionV2,load_source_state
from .test_native_mean_fusion_v2 import fixture


def test_hs_nonzero_source_warm_start_and_both_branches_match_single_arm_initialization():
    source,batch=fixture();start=torch.get_rng_state()
    model=NativeMeanHS(width=16,length=8).eval()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(start);single_h=NativeMeanFusionV2('H',width=16,length=8)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(start);single_s=NativeMeanFusionV2('S',width=16,length=8)
    for key,value in model.current_residual.state_dict().items():torch.testing.assert_close(value,single_h.current_residual.state_dict()[key],rtol=0,atol=0)
    for key,value in model.time.spatial.state_dict().items():torch.testing.assert_close(value,single_s.time.spatial.state_dict()[key],rtol=0,atol=0)
    load_source_state(model,source.state_dict())
    for a,b in zip(model(**batch,return_aux=True),source(**batch,return_aux=True)):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=0)
    assert any(k.startswith('current_residual.') for k in model.new_parameter_names())
    assert any(k.startswith('time.spatial.') for k in model.new_parameter_names())


def test_both_new_mechanisms_receive_gradients_and_keep_causal_auxiliary_prefix():
    source,batch=fixture();model=NativeMeanHS(width=16,length=8).eval();load_source_state(model,source.state_dict())
    model(**batch,return_aux=True)[0].square().mean().backward()
    assert model.current_residual.weight.grad.abs().max()>0
    assert model.time.spatial.gate.grad.abs()>0
    with torch.no_grad():
        model.time.spatial.gate.fill_(.3);model.current_residual.weight.normal_(std=.05)
    a=model(**batch,return_aux=True)
    history=batch['history'].clone();history[:,5:]+=50
    b=model(**{**batch,'history':history},return_aux=True)
    torch.testing.assert_close(a[2][:,:,:5],b[2][:,:,:5],atol=2e-5,rtol=0)
    part={**batch,'condition':batch['condition'][:,:2],'query_static':batch['query_static'][:2]}
    torch.testing.assert_close(model(**part)[0],a[0][:,:2],atol=2e-5,rtol=0)
    order=torch.arange(len(batch['context_ids'])-1,-1,-1)
    shuffled={**batch,'history':batch['history'][:,:,order],
        'context_static':batch['context_static'][order],'context_ids':batch['context_ids'][order]}
    other=model(**shuffled,return_aux=True)
    torch.testing.assert_close(other[0],a[0],atol=2e-5,rtol=0)
    torch.testing.assert_close(other[2],a[2][:,order],atol=2e-5,rtol=0)


def test_hs_training_rejects_development_and_short_formal_runs(monkeypatch,tmp_path):
    from . import native_mean_hs_experiment as experiment
    args=SimpleNamespace(cache=tmp_path,device='cpu',smoke_cache=False,steps=3000)
    monkeypatch.setattr(experiment,'CachedData',lambda *a,**k:SimpleNamespace(development=True))
    with pytest.raises(ValueError,match='development'):
        experiment.train(args)
    monkeypatch.setattr(experiment,'CachedData',lambda *a,**k:SimpleNamespace(development=False))
    args.steps=4
    with pytest.raises(ValueError,match='formal'):
        experiment.train(args)


def test_export_requires_same_source_and_seed_but_allows_new_calendar():
    from .native_mean_hs_experiment import require_export_contract
    source=dict(data_sha256='data',source_sha256='source',I1_sha256='I1',config=dict(seed=42,calendar_day=0))
    training=dict(source_protocol=source,config=dict(seed=42))
    cache={**source,'config':dict(seed=42,calendar_day=15)}
    require_export_contract(training,cache)
    with pytest.raises(ValueError,match='seed'):
        require_export_contract(training,{**cache,'config':dict(seed=17,calendar_day=15)})
    with pytest.raises(ValueError,match='source'):
        require_export_contract(training,{**cache,'I1_sha256':'other'})


def test_ema_must_equal_sealed_final_checkpoint():
    from .native_mean_hs_experiment import require_final_ema
    final=dict(step=3000,ema={'weight':torch.tensor([1.,2.])})
    require_final_ema({'weight':torch.tensor([1.,2.])},final)
    with pytest.raises(ValueError,match='EMA'):
        require_final_ema({'weight':torch.tensor([1.,3.])},final)
    with pytest.raises(ValueError,match='EMA'):
        require_final_ema({'other':torch.tensor([1.,2.])},final)
