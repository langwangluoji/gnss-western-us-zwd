import numpy as np
import torch
from . import model


def test_new_background_branch_initially_preserves_frozen_field_exactly():
    torch.manual_seed(1);m=model.BackgroundCorrection(True)
    base=torch.randn(2,3)*100;condition=torch.randn(2,3,111);history=torch.randn(2,16,3,4)
    prediction=m(base,condition,history)
    torch.testing.assert_close(prediction,base,rtol=0,atol=0)
    (prediction-20).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_control_cannot_use_any_vmf_values_even_after_training():
    torch.manual_seed(2);m=model.BackgroundCorrection(False)
    with torch.no_grad():m.head[-1].weight.fill_(.1)
    base=torch.zeros(2,3);condition=torch.randn(2,3,111)
    torch.testing.assert_close(m(base,condition,torch.zeros(2,16,3,4)),m(base,condition,torch.randn(2,16,3,4)*100),rtol=0,atol=0)


def test_training_batch_uses_no_future_background_or_station_label():
    rng=np.random.default_rng(1);cache=dict(field=rng.normal(size=(40,3)).astype('f4'),condition=rng.normal(size=(40,3,111)).astype('f4'),vmf=rng.normal(size=(40,3,4)).astype('f4'))
    before=model.batch(cache,np.array([20,21]),device='cpu')
    cache['vmf'][22:]=999;cache['condition'][22:]=888;cache['field'][22:]=777
    after=model.batch(cache,np.array([20,21]),device='cpu')
    for x,y in zip(before,after):torch.testing.assert_close(x,y,rtol=0,atol=0)
