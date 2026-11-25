"""Contracts for independent mean continuation; every model test runs on CPU."""
import importlib
from pathlib import Path
import pytest
import torch
from .native_physics_lm import NativePhysicsLM


torch.set_num_threads(2)


def api():
    name='tools.terrain_diffusion.native_mean_fusion_v2'
    assert importlib.util.find_spec(name) is not None, 'independent v2 mean implementation is missing'
    return importlib.import_module(name)


def fixture():
    torch.manual_seed(117)
    history=torch.randn(2,8,12,15);history[:,:,:,3:6]=1
    cs=torch.randn(12,9);qs=torch.randn(5,9)
    cs[:,:2]*=.1;qs[:,:2]*=.1;cs[:,2]*=.1;qs[:,2]*=.1
    cs[1,:2]=cs[0,:2]
    batch=dict(condition=torch.randn(2,5,111),history=history,context_static=cs,
        context_ids=torch.tensor([23,7,15,2,4,19,5,11,21,1,8,3]),query_static=qs)
    source=NativePhysicsLM('transformer',width=16,length=8).eval()
    for name,p in source.named_parameters():
        if name.startswith(('head.','node_head.','current_gain.','future_gain.','head_mix.')):
            torch.nn.init.normal_(p,std=.07)
    return source,batch


@pytest.mark.parametrize('arm',['C','H','S'])
def test_nonzero_source_warm_start_is_exact(arm):
    v=api();source,b=fixture();model=v.NativeMeanFusionV2(arm,width=16,length=8).eval()
    v.load_source_state(model,source.state_dict())
    expected=source(**b,return_aux=True);actual=model(**b,return_aux=True)
    assert expected[0].abs().max()>.01
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=2e-5,rtol=0)


def test_source_loading_rejects_missing_or_extra_old_keys():
    v=api();source,_=fixture();model=v.NativeMeanFusionV2('H',width=16,length=8)
    state=dict(source.state_dict());del state['head.weight']
    with pytest.raises(ValueError):v.load_source_state(model,state)
    state=dict(source.state_dict());state['unregistered']=torch.ones(1)
    with pytest.raises(ValueError):v.load_source_state(model,state)


@pytest.mark.parametrize('arm',['H','S'])
def test_active_new_branch_preserves_causal_prefixes_and_context_order(arm):
    v=api();source,b=fixture();model=v.NativeMeanFusionV2(arm,width=16,length=8).eval()
    v.load_source_state(model,source.state_dict())
    with torch.no_grad():
        if arm=='S':model.time.spatial.gate.fill_(.3)
        else:torch.nn.init.normal_(model.current_residual.weight,std=.05)
    a=model(**b,return_aux=True)
    altered={**b,'history':b['history'].clone()};altered['history'][:,5:]+=50
    z=model(**altered,return_aux=True)
    torch.testing.assert_close(a[2][:,:,:5],z[2][:,:,:5],atol=2e-5,rtol=0)
    order=torch.tensor([7,1,9,0,3,8,5,10,2,6,11,4])
    shuffled={**b,'history':b['history'][:,:,order],
        'context_static':b['context_static'][order],'context_ids':b['context_ids'][order]}
    p=model(**shuffled,return_aux=True)
    torch.testing.assert_close(p[0],a[0],atol=2e-5,rtol=0)
    torch.testing.assert_close(p[2],a[2][:,order],atol=2e-5,rtol=0)
    chunk={**b,'condition':b['condition'][:,:2],'query_static':b['query_static'][:2]}
    torch.testing.assert_close(model(**chunk)[0],a[0][:,:2],atol=2e-5,rtol=0)


def test_knn_duplicate_coordinates_break_ties_by_fixed_station_id():
    v=api();ids=torch.tensor([20,5,8,1,11,3,9,4,2,7,6,10]);st=torch.zeros(12,9)
    neighbors=v.neighbor_indices(st,ids,8)
    assert ids[neighbors[0]].tolist()==[1,2,3,4,5,6,7,8]
    order=torch.tensor([5,3,8,2,0,11,9,6,4,1,10,7])
    other=v.neighbor_indices(st[order],ids[order],8)
    for i in range(12):
        row=int((ids[order]==ids[i]).nonzero()[0])
        assert ids[order][other[row]].tolist()==ids[neighbors[i]].tolist()


def test_spatial_messages_ignore_masked_neighbor_values():
    v=api();source,b=fixture();model=v.NativeMeanFusionV2('S',width=16,length=8).eval()
    v.load_source_state(model,source.state_dict())
    with torch.no_grad():model.time.spatial.gate.fill_(.5)
    # Neighbor1 still has T/P; masked GNSS and derived innovations cannot enter node0.
    b['history'][:,:,1,3]=0
    a=model.time(b['history'],b['context_static'],b['context_ids'])[:,0]
    altered=b['history'].clone();altered[:,:,1,[0,9,10]]=10000
    z=model.time(altered,b['context_static'],b['context_ids'])[:,0]
    torch.testing.assert_close(a,z,atol=2e-5,rtol=0)
    b['history'][:,:,1,3:6]=0
    a=model.time(b['history'],b['context_static'],b['context_ids'])[:,0]
    altered=b['history'].clone();altered[:,:,1,:3]=10000
    z=model.time(altered,b['context_static'],b['context_ids'])[:,0]
    torch.testing.assert_close(a,z,atol=2e-5,rtol=0)


@pytest.mark.parametrize('arm',['H','S'])
def test_new_zero_branch_has_an_initial_gradient(arm):
    v=api();source,b=fixture();model=v.NativeMeanFusionV2(arm,width=16,length=8)
    v.load_source_state(model,source.state_dict());pred,_,aux=model(**b,return_aux=True)
    (pred.square().mean()+aux.square().mean()).backward()
    parameter=model.current_residual.weight if arm=='H' else model.time.spatial.gate
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum()>1e-9


def test_h_current_shift_enters_all_raw_horizons_before_original_limits():
    v=api();source,b=fixture();model=v.NativeMeanFusionV2('H',width=16,length=8).eval()
    v.load_source_state(model,source.state_dict());before=model(**b)[0]
    with torch.no_grad():model.current_residual.bias[0]=.15
    after=model(**b)[0];limits=model.limits
    raw_before=torch.atanh(before/limits)*limits;raw_after=torch.atanh(after/limits)*limits
    change=raw_after-raw_before
    assert change[:,:,0].abs().max()>1e-6
    torch.testing.assert_close(change,change[:,:,:1].expand_as(change),atol=2e-5,rtol=0)


def runner_api():
    name='tools.terrain_diffusion.native_mean_fusion_v2_experiment'
    assert importlib.util.find_spec(name) is not None, 'independent continuation runner is missing'
    return importlib.import_module(name)


@pytest.mark.parametrize('arm',['C','H','S'])
def test_adam_moments_continue_by_name_and_only_new_parameters_start_at_zero(arm):
    v=api();run=runner_api();source,b=fixture()
    old=torch.optim.AdamW(source.parameters(),lr=3e-4,weight_decay=.01)
    prediction,_,aux=source(**b,return_aux=True)
    (prediction.square().mean()+aux.square().mean()).backward();old.step()
    checkpoint=dict(optimizer=old.state_dict())
    model=v.NativeMeanFusionV2(arm,width=16,length=8)
    v.load_source_state(model,source.state_dict())
    optimizer,scheduler=run.continuation_optimizer(model,source,checkpoint,steps=4)
    target=dict(model.named_parameters())
    for name,p in source.named_parameters():
        expected=old.state.get(p,{})
        actual=optimizer.state.get(target[name],{})
        assert set(actual)==set(expected)
        for field in expected:torch.testing.assert_close(actual[field],expected[field],atol=0,rtol=0)
    for name in model.new_parameter_names():assert not optimizer.state.get(target[name])
    assert [g['lr'] for g in optimizer.param_groups]==([1e-4] if arm=='C' else [1e-4,3e-4])
    for _ in range(4):optimizer.step();scheduler.step()
    assert optimizer.param_groups[0]['lr']==pytest.approx(1e-5)
    if arm!='C':assert optimizer.param_groups[1]['lr']==pytest.approx(3e-5)


def test_export_prediction_is_absolute_millimeters():
    run=runner_api();source,b=fixture();base=torch.full((2,5),123.5)
    prediction,embedding=run.predict_absolute(source,b,base)
    correction,h=source(**b)
    torch.testing.assert_close(prediction,base[:,:,None]+correction)
    torch.testing.assert_close(embedding,h)
    assert prediction.mean()>100


@pytest.mark.parametrize('seed',[42,17])
def test_actual_source_checkpoints_preserve_all_three_arms_on_cpu(seed):
    v=api();_,b=fixture()
    b['history']=b['history'].repeat(1,9,1,1) # Real checkpoint registers all six lags through24h.
    path=Path(__file__).resolve().parents[2]/f'results/20261126_native_weather_means/transformer72_physics_aux_s{seed}/checkpoint_3000.pt'
    if not path.exists():pytest.skip('local source checkpoint not present')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    source=NativePhysicsLM('transformer',64,72).eval();source.load_state_dict(saved['model'])
    assert source.head.weight.abs().sum()>0
    with torch.no_grad():
        expected=source(**b,return_aux=True)
        for arm in ('C','H','S'):
            model=v.NativeMeanFusionV2(arm).eval();v.load_source_state(model,saved['model'])
            for actual,want in zip(model(**b,return_aux=True),expected):
                torch.testing.assert_close(actual,want,atol=2e-5,rtol=0)


def test_independent_gap_cache_is_explicit_and_limited_to_four_smoke_updates():
    run=runner_api()
    assert hasattr(run,'continuation_indices'), 'explicit smoke-cache budget guard is missing'
    assert run.continuation_indices(3,4,False,smoke=True)==[0,1,2,0]
    assert run.continuation_indices(3000,3000,True)==list(range(3000))
    with pytest.raises(ValueError):run.continuation_indices(3,4,False)
    with pytest.raises(ValueError):run.continuation_indices(3,5,False,smoke=True)
    with pytest.raises(ValueError):run.continuation_indices(3,4,True)
