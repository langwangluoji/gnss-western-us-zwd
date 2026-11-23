import pytest
import torch


def test_observation_adversarial_gradient_scale_uses_same_generator_parameters():
    from gnss_field.generative.training import adversarial_gradient_scale
    parameter=torch.tensor([1.,2.],requires_grad=True)
    values=adversarial_gradient_scale(parameter.square().sum(),.1*parameter.square().sum(),[parameter],1.)
    assert float(values['adversarial_gradient_ratio'])==pytest.approx(.1)
    with pytest.raises(FloatingPointError,match='dominates'):
        adversarial_gradient_scale(parameter.square().sum(),100*parameter.square().sum(),[parameter],1.)
