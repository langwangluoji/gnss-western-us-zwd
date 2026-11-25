import torch
from .native_weather import graph_basis
from .native_weather_models import NativeDiffusion
from .native_frequency_tokens import NativeFrequencyTokens


def test_frequency_token_model_is_matched_capacity_and_permutation_equivariant():
    torch.manual_seed(9);model=NativeFrequencyTokens(64,width=64).eval()
    control=NativeDiffusion(64,width=64,domain='graph')
    ratio=sum(p.numel() for p in model.parameters())/sum(p.numel() for p in control.parameters())
    assert .95<ratio<1.05
    torch.nn.init.normal_(model.output.weight,std=.1)
    xy=torch.randn(5,2)*100;eigen,U=graph_basis(xy,torch.zeros(5))
    x=torch.randn(2,5,7);c=torch.randn(2,5,64);k=torch.ones(2)*.4
    a=model.denoise(x,c,k,U,eigen);order=torch.tensor([3,0,4,2,1])
    b=model.denoise(x[:,order],c[:,order],k,U[order],eigen)
    torch.testing.assert_close(a[:,order],b,atol=2e-5,rtol=2e-5)


def test_all_frequency_coefficients_participate_and_zero_velocity_preserves_reference():
    torch.manual_seed(10);model=NativeFrequencyTokens(16,width=16).eval()
    eigen,U=graph_basis(torch.randn(4,2)*100,torch.zeros(4))
    c=torch.randn(2,4,16);noise=torch.randn(2,4,7)
    torch.testing.assert_close(model.sample(c,noise,U,eigen,steps=4,solver='heun'),noise,atol=0,rtol=0)
    torch.nn.init.normal_(model.output.weight,std=.1)
    # The highest temporal coefficient remains an input, with a nonzero physical gradient.
    x=torch.randn(2,4,7,requires_grad=True)
    output=model.denoise(x,c,torch.ones(2)*.5,U,eigen)
    gradient=torch.autograd.grad(output.square().sum(),x)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)>0
    assert (gradient@model.D.T)[...,-1].abs().max()>1e-7
