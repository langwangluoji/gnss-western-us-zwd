import importlib
import importlib.util
import torch
from .native_physics_lm import NativePhysicsLM
from .native_mean_fusion_v2 import load_source_state


def module():
    name=__package__+'.native_directed_lag'
    assert importlib.util.find_spec(name) is not None, 'directed station-lag readout is missing'
    return importlib.import_module(name)


def batch():
    torch.manual_seed(19);B,L,C,Q=2,8,4,3
    history=torch.randn(B,L,C,15);history[...,3:6]=1;history[...,6:9]=0
    return dict(condition=torch.randn(B,Q,111),history=history,context_static=torch.randn(C,9)*.02,
        context_ids=torch.tensor([2,9,17,38]),query_static=torch.randn(Q,9)*.02)


def test_nonzero_source_warm_start_and_same_arm_initialization():
    m=module();b=batch();torch.manual_seed(31)
    source=NativePhysicsLM('transformer',64,8).eval()
    with torch.no_grad():source.head.weight.normal_(0,.02)
    models=[]
    for mode in ['factorized','joint']:
        torch.manual_seed(71);model=m.NativeDirectedLag(mode,length=8).eval();load_source_state(model,source.state_dict());models.append(model)
        with torch.no_grad():
            for a,c in zip(source(**b,return_aux=True),model(**b,return_aux=True)):torch.testing.assert_close(a,c,rtol=0,atol=0)
    for key,value in models[0].state_dict().items():torch.testing.assert_close(value,models[1].state_dict()[key],rtol=0,atol=0)


def test_active_readout_query_subset_and_station_permutation():
    m=module();b=batch();model=m.NativeDirectedLag('joint',length=8).eval()
    with torch.no_grad():model.transport.readout.normal_(0,.1)
    full=model(**b)[0];small={**b,'condition':b['condition'][:,[2,0]],'query_static':b['query_static'][[2,0]]}
    torch.testing.assert_close(model(**small)[0],full[:,[2,0]],rtol=1e-5,atol=1e-5)
    order=torch.tensor([3,0,2,1]);permuted={**b,'history':b['history'][:,:,order],
        'context_static':b['context_static'][order],'context_ids':b['context_ids'][order]}
    torch.testing.assert_close(model(**permuted)[0],full,rtol=1e-5,atol=1e-5)


def test_value_masks_and_empty_observations_return_zero_without_nan():
    m=module();b=batch();r=m.DirectedLagReadout('joint',length=8)
    with torch.no_grad():r.readout.normal_(0,.1)
    h=torch.randn(2,3,64);states=torch.randn(2,4,8,64)
    b['history'][...,3:6]=0
    value=r(h,states,b['history'],b['context_static'],b['query_static'])
    torch.testing.assert_close(value,torch.zeros_like(value),rtol=0,atol=0)
    value.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in r.parameters())


def test_new_readout_and_attention_receive_gradients_after_two_updates():
    m=module();b=batch();model=m.NativeDirectedLag('joint',length=8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad();prediction,_=model(**b);loss=(prediction-2).square().mean();loss.backward();optimizer.step()
    for name in ['transport.readout','transport.key.weight','transport.query.weight','transport.edge.0.weight']:
        grad=dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name


def test_uniform_numeric_transport_matches_lag_values_and_variable_mask_denominator():
    m=module();r=m.DirectedLagReadout('joint',length=8)
    with torch.no_grad():
        for p in r.parameters():p.zero_()
        r.readout[0,0,0]=1
    history=torch.zeros(1,8,4,15);history[...,3:6]=1
    history[...,10]=torch.arange(8)[None,:,None]*.01
    h=torch.zeros(1,2,64);states=torch.zeros(1,4,8,64);context=torch.zeros(4,9);query=torch.zeros(2,9)
    output=r(h,states,history,context,query)
    torch.testing.assert_close(output[...,0],torch.full((1,2),3*torch.tanh(torch.tensor(.45/3))))
    history[0,6,2,3]=0
    output=r(h,states,history,context,query)
    torch.testing.assert_close(output[...,0],torch.full((1,2),3*torch.tanh(torch.tensor(.44/3))))
