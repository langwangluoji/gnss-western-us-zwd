import torch
import pytest


@pytest.mark.parametrize('name', ['DeterministicHead', 'GaussianHead', 'GANGenerator'])
def test_paired_coefficient_heads_have_four_active_residual_blocks(name):
    from gnss_field.models import heads
    model = getattr(heads, name)(8, width=16)
    assert hasattr(model, 'backbone') and len(model.backbone.blocks) == 4
    h = torch.randn(3, 128)
    out = model(torch.randn(3, 8), h) if name == 'GANGenerator' else model(h)
    loss = sum(v.square().mean() for v in out) if isinstance(out, tuple) else out.square().mean()
    loss.backward()
    for block in model.backbone.blocks:
        assert sum(p.grad.abs().sum() for p in block.parameters()) > 0
