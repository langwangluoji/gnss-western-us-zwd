"""Condition-centred station residual critic; only training GNSS is real.

The native condition contract is surface[0:5], reference W0/100[5],
current geometry[6:10], station orthometric height/1000[10].
Subtracting this input-only background from both real and fake changes numerical
conditioning, not information availability. No dense HR image truth is assumed.
"""
import torch
from .sr_stage2_models import ConditionalObservationDiscriminator


class ConditionalResidualDiscriminator(ConditionalObservationDiscriminator):
    def __init__(self, condition_dim, *, width=32, value_scale_mm=10.):
        if condition_dim != 11:
            raise ValueError('The residual critic requires the audited 11-channel condition')
        super().__init__(condition_dim, width=width, value_scale_mm=value_scale_mm)

    def forward(self, zwd_mm, condition, query_xy, valid):
        if zwd_mm.ndim == 3 and zwd_mm.shape[-1] == 1:
            zwd_mm = zwd_mm.squeeze(-1)
        if condition.shape != (*zwd_mm.shape, 11) or valid.shape != zwd_mm.shape:
            raise ValueError('Incompatible residual critic tensors')
        if not torch.isfinite(condition[valid]).all() or not torch.isfinite(zwd_mm[valid]).all():
            raise ValueError('Nonfinite valid values')
        context = torch.where(valid[..., None], condition, torch.zeros_like(condition))
        values = torch.where(valid, zwd_mm, torch.zeros_like(zwd_mm))
        background_zwd_mm = 100.*context[..., 5]*torch.exp(-context[..., 10]/3.)
        return super().forward(values-background_zwd_mm, context, query_xy, valid)
