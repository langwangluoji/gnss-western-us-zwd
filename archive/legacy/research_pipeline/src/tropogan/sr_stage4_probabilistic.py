"""Optional stochastic-field product; its frozen point predictor cannot improve.

The odd residual has zero expectation under symmetric iid noise. This restricts
the distribution to central symmetry; it cannot learn conditional skewness or
repair bias in the fixed mean. Scores describe the GNSS-derived ZWD reference,
including its errors, rather than pure atmospheric uncertainty.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from tropogan.sr_stage2_models import SRStage2Generator


class FrozenMeanStochasticField(nn.Module):
    def __init__(self, mean_model, *, width=8, head_init_std=.05):
        super().__init__()
        if not math.isfinite(head_init_std) or head_init_std <= 0:
            raise ValueError("A positive nonzero residual-head initialization is required")
        self.mean_model = mean_model
        self.mean_model.requires_grad_(False)
        self.mean_model.zero_grad(set_to_none=True)
        self.mean_model.eval()
        self.residual_model = SRStage2Generator(
            in_channels=5, current_condition_channels=4, static_channels=4,
            width=width, scale_factor=4, backbone="cnn", temporal="none",
            noise_channels=2, residual_scale_mm=20.)
        nn.init.normal_(self.residual_model.residual_head.weight, std=head_init_std)
        nn.init.zeros_(self.residual_model.residual_head.bias)
        if set(map(id, self.mean_model.parameters())) & set(map(id, self.residual_model.parameters())):
            raise ValueError("Frozen mean and stochastic branch cannot share parameters")

    def train(self, mode=True):
        super().train(mode)
        self.mean_model.eval()
        return self

    @staticmethod
    def draw_noise(batch, *, generator):
        """One iid member: local Gaussian field plus independent global scalar.

        Spatial locations within a member are intentionally dependent through
        the broadcast channel. Different calls require independent RNG draws.
        """
        surface, background, *_ = batch
        shape = (len(surface), 1, *background.shape[-2:])
        local = torch.randn(shape, device=background.device, dtype=background.dtype, generator=generator)
        global_noise = torch.randn((len(surface), 1, 1, 1), device=background.device,
                                   dtype=background.dtype, generator=generator).expand_as(local)
        return torch.cat((local, global_noise), dim=1)

    @staticmethod
    def _call(model, batch, noise=None):
        surface, background, current, static, times = batch
        return model(surface, background, static, current_condition=current,
                     time_offsets_hours=times, noise=noise, return_components=True)

    @torch.no_grad()
    def point_field(self, batch):
        self.mean_model.eval()
        return self._call(self.mean_model, batch)["field_mm"]

    def residual_field(self, batch, noise):
        plus = self._call(self.residual_model, batch, noise)["residual_mm"]
        minus = self._call(self.residual_model, batch, -noise)["residual_mm"]
        return .5*(plus-minus)

    def ensemble(self, batch, *, members, generator, return_residuals=False):
        if not isinstance(members, int) or isinstance(members, bool) or members < 2:
            raise ValueError("At least two iid members required for fair scores")
        mean = self.point_field(batch)
        residuals = torch.stack([self.residual_field(batch, self.draw_noise(batch, generator=generator))
                                 for _ in range(members)])
        # Do NOT subtract the realized ensemble mean: that changes dependence
        # and invalidates the iid fair-score estimator. Point output is mean.
        output = {"point_field_mm": mean, "sample_fields_mm": mean[None]+residuals}
        if return_residuals:
            output["residual_fields_mm"] = residuals
        return output


def _score_inputs(samples, truth, valid):
    if samples.ndim != 3 or truth.shape != samples.shape[1:] or valid.shape != truth.shape:
        raise ValueError("Expected samples[K,B,N], truth[B,N], valid[B,N]")
    if samples.shape[0] < 2 or valid.dtype != torch.bool or not valid.any(dim=1).all():
        raise ValueError("K>=2 and at least one valid observation per frame required")
    if samples.dtype != truth.dtype or samples.device != truth.device or valid.device != truth.device:
        raise ValueError("Score tensors must share device and floating dtype")
    if not torch.isfinite(samples[:, valid]).all() or not torch.isfinite(truth[valid]).all():
        raise ValueError("Nonfinite valid predictive/reference values")
    return torch.where(valid[None], samples, torch.zeros_like(samples)), torch.where(valid, truth, torch.zeros_like(truth))


def ensemble_energy_score(samples, truth, valid, *, fair_weight=1.):
    """Per-frame energy score with RMS-normalized Euclidean distances.

    fair_weight=1 is the iid U-statistic; 0 is the ordinary finite-ensemble
    score. A mixture (e.g. .95) is an explicitly biased training stabilizer.
    Units match samples; no unit scaling or aggregation over frames is hidden.
    """
    if not math.isfinite(fair_weight) or not 0 <= fair_weight <= 1:
        raise ValueError("fair_weight must be in [0,1]")
    samples, truth = _score_inputs(samples, truth, valid)
    k = len(samples)
    divisor = valid.sum(-1).to(samples.dtype).sqrt()
    observation_term = (torch.linalg.vector_norm(samples-truth[None], dim=-1)/divisor).mean(0)
    pair_norm = torch.linalg.vector_norm(samples[:, None]-samples[None, :], dim=-1)/divisor
    coefficient = fair_weight/(k*(k-1))+(1-fair_weight)/(k*k)
    return observation_term-.5*coefficient*pair_norm.sum((0, 1))


def ensemble_crps(samples, truth, valid, *, fair_weight=1.):
    """Per-observation CRPS estimator. Invalid positions are zero, not scored."""
    if not math.isfinite(fair_weight) or not 0 <= fair_weight <= 1:
        raise ValueError("fair_weight must be in [0,1]")
    samples, truth = _score_inputs(samples, truth, valid)
    k = len(samples)
    coefficient = fair_weight/(k*(k-1))+(1-fair_weight)/(k*k)
    return (samples-truth[None]).abs().mean(0)-.5*coefficient*(samples[:, None]-samples[None, :]).abs().sum((0, 1))
