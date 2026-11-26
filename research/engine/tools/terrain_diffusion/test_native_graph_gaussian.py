import importlib.util
from pathlib import Path
import torch
from .native_process_ops import heun0,generate_process
from .native_weather import graph_basis

path=Path(__file__).with_name('native_graph_gaussian.py')
spec=importlib.util.spec_from_file_location('graph_gaussian',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_graph_gaussian as module


def test_graph_gaussian_is_affine_in_the_entire_field_at_nonzero_weights():
    assert hasattr(module,'GraphAffineVelocity'),'joint Gaussian graph control missing'
    torch.manual_seed(17);model=module.GraphAffineVelocity()
    c=torch.randn(2,5,175);x=torch.randn(2,5,7);y=torch.randn_like(x);k=torch.tensor([.2,.8])
    eigen,U=graph_basis(torch.randn(5,2)*100,torch.randn(5)*500)
    torch.nn.init.normal_(model.net[-1].weight,std=.01)
    a=.3
    torch.testing.assert_close(model.denoise(a*x+(1-a)*y,c,k,U,eigen),
        a*model.denoise(x,c,k,U,eigen)+(1-a)*model.denoise(y,c,k,U,eigen),atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(heun0(model,c,a*x+(1-a)*y,U,eigen,4),
        a*heun0(model,c,x,U,eigen,4)+(1-a)*heun0(model,c,y,U,eigen,4),atol=6e-6,rtol=6e-6)
    assert sum(p.numel() for p in model.parameters())==45259


def test_graph_control_zero_head_inherits_reference_and_receives_gradients():
    assert hasattr(module,'GraphAffineVelocity'),'zero initialized Gaussian graph control missing'
    model=module.GraphAffineVelocity();c=torch.randn(2,5,175);eta=torch.randn(2,5,7);static=torch.randn(5,9)*.1
    cp=dict(spatial_rho=.6,time_correlation=torch.eye(7).tolist())
    expected=generate_process(None,'reference',c,eta,static,cp,steps=4)
    actual=generate_process(model,'full',c,eta,static,cp,steps=4)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    eigen,U=graph_basis(static[:,:2]*1000,static[:,2]*3000)
    loss=model.loss(eta,c,U,eigen);loss.backward()
    assert torch.isfinite(model.net[-1].weight.grad).all() and model.net[-1].weight.grad.abs().sum()>0


def test_graph_control_passes_other_query_conditions_and_is_permutation_equivariant():
    assert hasattr(module,'GraphAffineVelocity'),'graph conditional control missing'
    torch.manual_seed(42);model=module.GraphAffineVelocity()
    torch.nn.init.normal_(model.net[-1].weight,std=.03)
    c=torch.randn(2,5,175,requires_grad=True);x=torch.zeros(2,5,7);k=torch.tensor([.2,.8])
    eigen,U=graph_basis(torch.randn(5,2)*100,torch.randn(5)*500)
    v=model.denoise(x,c,k,U,eigen)
    gradient=torch.autograd.grad(v[:,0].sum(),c)[0]
    assert gradient[:,1:].abs().sum()>1e-9
    order=torch.tensor([4,1,3,0,2])
    actual=model.denoise(x[:,order],c[:,order],k,U[order],eigen)
    torch.testing.assert_close(actual,v[:,order],atol=2e-6,rtol=2e-6)
