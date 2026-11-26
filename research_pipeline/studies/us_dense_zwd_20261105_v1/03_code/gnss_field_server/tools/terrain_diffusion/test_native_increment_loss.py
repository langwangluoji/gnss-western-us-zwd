import importlib
import importlib.util
import torch
import pytest
from types import SimpleNamespace
from .native_weather_experiment import mean_loss


def module():
    name=__package__+'.native_increment_loss'
    assert importlib.util.find_spec(name) is not None, 'within-forecast increment loss is not implemented'
    return importlib.import_module(name)


def test_lead_axis_not_origin_axis_and_background_cancels():
    loss=module().within_forecast_increment_loss
    correction=torch.tensor([[[0.,10.,30.]],[[90.,100.,120.]]],requires_grad=True)
    truth=torch.zeros_like(correction);weights=torch.ones(2,1)
    # Lead errors /10 are [1,2] at both origins; base may vary by origin.
    torch.testing.assert_close(loss(correction,truth,torch.tensor([[12.],[-33.]]),weights),torch.tensor(2.5))
    value=loss(correction,truth,torch.zeros(2,1),weights);value.backward()
    assert torch.isfinite(correction.grad).all() and correction.grad.abs().sum()>0


def test_joint_mask_weights_and_missing_values_have_finite_gradient():
    loss=module().within_forecast_increment_loss
    correction=torch.tensor([[[0.,10.,30.],[0.,30.,float('nan')]]],requires_grad=True)
    truth=torch.tensor([[[0.,0.,0.],[0.,0.,float('nan')]]])
    # Per transition: (1*1^2 + 3*3^2)/4=7, then first query's 2^2=4.
    value=loss(correction,truth,torch.zeros(1,2),torch.tensor([[1.,3.]]))
    torch.testing.assert_close(value,torch.tensor(5.5));value.backward()
    assert torch.isfinite(correction.grad).all()
    assert correction.grad[0,1,2]==0


def test_empty_labels_yield_zero_connected_gradient():
    loss=module().within_forecast_increment_loss
    correction=torch.randn(2,3,7,requires_grad=True)
    value=loss(correction,torch.full_like(correction,float('nan')),torch.zeros(2,3),torch.ones(2,3))
    assert value.item()==0;value.backward()
    torch.testing.assert_close(correction.grad,torch.zeros_like(correction))


def test_zero_weight_reproduces_legacy_loss_exactly_and_negative_weight_rejected():
    joint=module().increment_mean_loss
    correction=torch.randn(12,4,7,requires_grad=True);truth=torch.randn_like(correction)
    base=torch.randn(12,4);weights=torch.rand(12,4)
    torch.testing.assert_close(joint(correction,truth,base,weights,0),mean_loss(correction,truth,base,weights),rtol=0,atol=0)
    with pytest.raises(ValueError,match='nonnegative'):
        joint(correction,truth,base,weights,-.5)


def test_formal_trainer_rejects_unregistered_budget_and_weight_before_loading_data():
    from . import native_increment_experiment as experiment
    args=SimpleNamespace(smoke_cache=False,steps=4,increment_weight=.5)
    with pytest.raises(ValueError,match='formal'):
        experiment.train(args)
    args.steps=3000;args.increment_weight=.1
    with pytest.raises(ValueError,match='weight'):
        experiment.train(args)
