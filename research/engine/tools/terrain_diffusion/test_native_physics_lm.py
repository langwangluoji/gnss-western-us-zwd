import pytest
import torch
from .native_physics_lm import NativePhysicsLM,causal_station_loss


def batch():
    torch.manual_seed(42)
    history=torch.randn(2,8,4,15);history[:,:,:,3:6]=1
    cstatic=torch.randn(4,9);qstatic=torch.randn(5,9)
    cstatic[:,:2]*=.1;qstatic[:,:2]*=.1;cstatic[:,2]=0;qstatic[:,2]=0
    return dict(condition=torch.randn(2,5,111),history=history,context_static=cstatic,
        context_ids=torch.arange(4),query_static=qstatic)


@pytest.mark.parametrize('kind',['gru','lstm','transformer'])
def test_physical_lm_initialization_preserves_the_frozen_field(kind):
    model=NativePhysicsLM(kind,width=16,length=8).eval();b=batch()
    correction,h,aux=model(**b,return_aux=True)
    assert torch.count_nonzero(correction)==0 and h.shape==(2,5,16)
    assert aux.shape==(2,4,8,6,3)


@pytest.mark.parametrize('kind',['gru','lstm','transformer'])
def test_station_next_state_auxiliary_predictions_are_strictly_causal(kind):
    b=batch();model=NativePhysicsLM(kind,width=16,length=8).eval()
    torch.nn.init.normal_(model.node_head.weight,std=.1)
    a=model(**b,return_aux=True)[2]
    b['history'][:,5:]+=50
    z=model(**b,return_aux=True)[2]
    torch.testing.assert_close(a[:,:,:5],z[:,:,:5],atol=2e-5,rtol=2e-5)


def test_next_state_loss_has_physical_units_and_masks_both_endpoints():
    history=torch.zeros(1,8,2,15);history[:,:,:,3:6]=1
    t=torch.arange(8).float()
    history[0,:,:,0]=t[:,None]*.1;history[0,:,:,1]=t[:,None]*.1;history[0,:,:,2]=t[:,None]*.02
    prediction=torch.arange(1,7).float()[None,None,None,:,None].expand(1,2,8,6,3).clone()
    loss,count=causal_station_loss(prediction,history)
    assert loss<1e-10 and count>0
    history[:,:,:,3:6]=0
    loss,count=causal_station_loss(prediction,history)
    assert loss==0 and count==0


def test_innovation_value_path_transports_height_and_does_not_depend_on_other_queries():
    b=batch();b['history'][:,:,:,10]=.2
    model=NativePhysicsLM('gru',width=16,length=8).eval()
    with torch.no_grad():model.current_gain.bias.fill_(torch.atanh(torch.tensor(.1)))
    low=model(**b)[0]
    b['query_static'][:,2]=1 # +3000m
    high=model(**b)[0]
    physical=lambda x:torch.atanh(x[:,:,0]/5)*5
    torch.testing.assert_close(physical(high),physical(low)*torch.exp(torch.tensor(-1.)),atol=2e-5,rtol=2e-5)
    b['condition']=b['condition'][:,:3];b['query_static']=b['query_static'][:3]
    subset=model(**b)[0]
    torch.testing.assert_close(subset,high[:,:3],atol=2e-5,rtol=2e-5)
