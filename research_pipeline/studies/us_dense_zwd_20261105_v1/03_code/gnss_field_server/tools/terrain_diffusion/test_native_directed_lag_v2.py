import importlib
import importlib.util
import torch


def readout_class():
    name=__package__+'.native_directed_lag_v2'
    if importlib.util.find_spec(name) is None:
        from .native_directed_lag import DirectedLagReadout
        return DirectedLagReadout
    return importlib.import_module(name).DirectedLagReadoutV2


def test_far_valid_gnss_is_not_diluted_by_nearby_temperature_only_station():
    for mode in ['factorized','joint']:
        model=readout_class()(mode,length=2)
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.readout[0,0,0]=1
        h=torch.zeros(1,1,64);states=torch.zeros(1,2,2,64);history=torch.zeros(1,2,2,15)
        context=torch.zeros(2,9);context[1,0]=.35;query=torch.zeros(1,9)
        history[:,:,1,3]=1;history[:,:,1,10]=.1
        expected=3*torch.tanh(torch.tensor(1/3))
        alone=model(h,states,history,context,query)[0,0,0]
        torch.testing.assert_close(alone,expected,rtol=1e-6,atol=1e-6)
        history[:,:,0,4]=1
        with_temperature=model(h,states,history,context,query)[0,0,0]
        torch.testing.assert_close(with_temperature,expected,rtol=1e-6,atol=1e-6)


def test_per_variable_empty_support_is_zero_and_has_finite_gradients():
    model=readout_class()('joint',length=2)
    with torch.no_grad():model.readout.normal_(0,.1)
    h=torch.zeros(1,1,64);states=torch.zeros(1,2,2,64);history=torch.zeros(1,2,2,15)
    context=torch.zeros(2,9);context[1,0]=.35;query=torch.zeros(1,9)
    output=model(h,states,history,context,query)
    torch.testing.assert_close(output,torch.zeros_like(output),rtol=0,atol=0)
    output.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_v2_retains_v1_initial_parameters_and_nonzero_source_warm_equivalence():
    from .native_directed_lag_v2 import NativeDirectedLagV2
    from .native_directed_lag import NativeDirectedLag
    from .native_physics_lm import NativePhysicsLM
    from .native_mean_fusion_v2 import load_source_state
    from .test_native_directed_lag import batch
    b=batch();source=NativePhysicsLM(length=8).eval()
    with torch.no_grad():source.head.weight.normal_(0,.02)
    for mode in ['factorized','joint']:
        torch.manual_seed(91);old=NativeDirectedLag(mode,length=8)
        torch.manual_seed(91);new=NativeDirectedLagV2(mode,length=8).eval()
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        load_source_state(new,source.state_dict())
        with torch.no_grad():
            for a,c in zip(source(**b,return_aux=True),new(**b,return_aux=True)):torch.testing.assert_close(a,c,rtol=0,atol=0)


def test_v2_active_readout_query_subset_station_permutation_and_gradients():
    from .native_directed_lag_v2 import NativeDirectedLagV2
    from .test_native_directed_lag import batch
    b=batch();model=NativeDirectedLagV2('joint',length=8).eval()
    with torch.no_grad():model.transport.readout.normal_(0,.1)
    full=model(**b)[0];subset={**b,'condition':b['condition'][:,[2,0]],'query_static':b['query_static'][[2,0]]}
    torch.testing.assert_close(model(**subset)[0],full[:,[2,0]],rtol=1e-5,atol=1e-5)
    order=torch.tensor([3,0,2,1]);permuted={**b,'history':b['history'][:,:,order],
        'context_static':b['context_static'][order],'context_ids':b['context_ids'][order]}
    torch.testing.assert_close(model(**permuted)[0],full,rtol=1e-5,atol=1e-5)
    full.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
