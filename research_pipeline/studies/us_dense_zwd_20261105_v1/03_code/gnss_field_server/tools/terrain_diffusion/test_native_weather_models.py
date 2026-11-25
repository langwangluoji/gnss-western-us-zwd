import pytest
import torch
from .native_weather import graph_basis
from .native_weather_models import StationEncoder,NativeMean,NativeDiffusion


def inputs():
    torch.manual_seed(19)
    return dict(condition=torch.randn(2,5,111),history=torch.randn(2,8,4,15),
        context_static=torch.randn(4,9),context_ids=torch.arange(4),query_static=torch.randn(5,9))


@pytest.mark.parametrize('kind',['gru','lstm','transformer'])
def test_time_encoders_are_causal_and_equal_history(kind):
    b=inputs();model=StationEncoder(kind,width=16,length=8).eval()
    a=model(b['history'],b['context_static'],b['context_ids'])
    b['history'][:,5:]+=100
    z=model(b['history'],b['context_static'],b['context_ids'])
    torch.testing.assert_close(a[:,:,:5],z[:,:,:5],atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('kind',['weather','gru','lstm','transformer'])
def test_mean_head_initially_preserves_I1_and_station_order(kind):
    b=inputs();model=NativeMean(kind,width=16,length=8).eval()
    result,embedding=model(**b)
    assert torch.count_nonzero(result)==0 and embedding.shape==(2,5,16)
    torch.nn.init.normal_(model.head.weight,std=.1)
    a,_=model(**b);order=torch.tensor([3,1,0,2])
    b['history']=b['history'][:,:,order];b['context_static']=b['context_static'][order];b['context_ids']=b['context_ids'][order]
    z,_=model(**b)
    torch.testing.assert_close(a,z,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('domain',['spatial','graph','joint'])
def test_velocity_diffusion_loss_sampling_and_graph_permutation(domain):
    torch.manual_seed(7);model=NativeDiffusion(16,width=16,domain=domain).eval()
    _,U=graph_basis(torch.randn(5,2)*100,torch.zeros(5))
    eigen,U=graph_basis(torch.randn(5,2)*100,torch.zeros(5))
    c=torch.randn(2,5,16);x=torch.randn(2,5,7);time=torch.ones(2)*.5
    loss=model.loss(x,c,U,eigen)
    assert torch.isfinite(loss)
    a=model.denoise(x,c,time,U,eigen)
    order=torch.tensor([4,2,0,3,1])
    z=model.denoise(x[:,order],c[:,order],time,U[order],eigen)
    torch.testing.assert_close(a[:,order],z,atol=2e-5,rtol=2e-5)
    sample=model.sample(c,torch.randn_like(x),U,eigen,steps=4)
    assert sample.shape==x.shape and torch.isfinite(sample).all()
