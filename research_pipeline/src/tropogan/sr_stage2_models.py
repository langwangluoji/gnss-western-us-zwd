"""Sparse-observation-constrained 2D delay fields, with no dense truth assumption.

Generators output a complete numerical field [B,1,sH,sW], never a centre-point
regression. The caller defines the physical reference plane (for example W0 at
a common altitude) and applies any explicit station-height observation operator.
Nothing here interprets IDW/bilinear backgrounds as observed high-resolution truth.

Temporal inputs are near-surface meteorology only. Current reference-GNSS maps
have a separate current_condition branch. Feature eligibility, physical-group
exclusion, train-only normalization and temporal split buffers remain mandatory
data contracts enforced by the caller. No module reads files or target histories.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def normalized_grid_coordinates(longitude, latitude, *, lon_bounds, lat_bounds,
                                latitude_descending=True):
    """Convert lon/lat to grid_sample x/y using OUTER PIXEL CELL bounds.

    For a 41x61 grid centred at 45..55N, 5..20E at 0.25 degrees, bounds are
    lat=(44.875,55.125), lon=(4.875,20.125). A x4 grid has 164x244 cells over
    exactly these bounds. Coordinates refer to align_corners=False. Default
    array row zero is the northernmost cell; set latitude_descending=False
    when rows run south to north. No CRS reprojection is performed.
    """
    longitude = torch.as_tensor(longitude)
    if not longitude.is_floating_point():
        longitude = longitude.to(torch.get_default_dtype())
    latitude = torch.as_tensor(latitude, dtype=longitude.dtype, device=longitude.device)
    longitude, latitude = torch.broadcast_tensors(longitude, latitude)
    west, east = map(float, lon_bounds)
    south, north = map(float, lat_bounds)
    if not all(math.isfinite(x) for x in (west, east, south, north)) or not west < east or not south < north:
        raise ValueError("Expected finite increasing outer cell bounds")
    x = 2*(longitude-west)/(east-west)-1
    y = 2*(latitude-south)/(north-south)-1
    if latitude_descending:
        y = -y
    return torch.stack((x, y), dim=-1)


def _query_mask(query_xy, valid, batch_size):
    if query_xy.ndim != 3 or query_xy.shape[0] != batch_size or query_xy.shape[-1] != 2:
        raise ValueError("query_xy must be [B,N,2], x/longitude then y/latitude")
    if valid is None:
        valid = torch.ones(query_xy.shape[:2], dtype=torch.bool, device=query_xy.device)
    if valid.dtype != torch.bool or tuple(valid.shape) != tuple(query_xy.shape[:2]):
        raise ValueError("valid must be a Boolean [B,N] mask")
    if valid.device != query_xy.device:
        raise ValueError("Mask and query coordinates must share a device")
    values = query_xy[valid]
    if not torch.isfinite(values).all() or (values.abs() > 1+1e-6).any():
        raise ValueError("Valid observation queries must be finite and within the declared grid bounds")
    return torch.where(valid[..., None], query_xy, torch.zeros_like(query_xy)).clamp(-1, 1), valid


def sparse_observe(field, query_xy, valid=None):
    """Differentiably sample a 2D field at continuous station locations.

    field [B,C,H,W], normalized query_xy [B,N,2], Boolean valid [B,N].
    Returns [B,N,C]; invalid padded stations are exactly zero with zero gradient.
    No nearest-cell rounding, station aggregation or target-derived filling.
    Border padding extends only the last half pixel inside the declared cell
    bounds; queries outside those bounds are rejected instead of extrapolated.
    """
    if field.ndim != 4 or not field.is_floating_point():
        raise ValueError("field must be floating [B,C,H,W]")
    if field.device != query_xy.device or field.dtype != query_xy.dtype:
        raise ValueError("Field and queries must share floating dtype/device")
    if not torch.isfinite(field).all():
        raise ValueError("Field contains nonfinite values")
    safe, valid = _query_mask(query_xy, valid, len(field))
    if query_xy.shape[1] == 0:
        return field.new_zeros((len(field), 0, field.shape[1]))
    values = F.grid_sample(field, safe[:, :, None, :], mode="bilinear",
                           padding_mode="border", align_corners=False)
    values = values.squeeze(-1).transpose(1, 2)
    return torch.where(valid[..., None], values, torch.zeros_like(values))


def masked_observation_loss(prediction, target, valid, *, kind="smooth_l1", beta_mm=10.):
    """Loss on observed scalar delays only; padding cannot create NaN gradients.

    Apply the caller's station-height operator before invoking this function.
    No-observation batches must be explicitly skipped by the training loop.
    """
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction.squeeze(-1)
    if prediction.shape != target.shape or target.shape != valid.shape or target.ndim != 2:
        raise ValueError("Prediction, target and mask must have shape [B,N]")
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("At least one valid observed target is required")
    pred, truth = prediction[valid], target[valid]
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError("Nonfinite valid observation or prediction")
    if kind == "mse":
        return F.mse_loss(pred, truth)
    if kind == "mae":
        return F.l1_loss(pred, truth)
    if kind == "smooth_l1" and math.isfinite(beta_mm) and beta_mm > 0:
        return F.smooth_l1_loss(pred, truth, beta=beta_mm)
    raise ValueError("Expected mse, mae or smooth_l1 with positive finite beta_mm")


def _group_norm(width):
    groups = max(group for group in (1, 2, 4, 8) if width % group == 0)
    return nn.GroupNorm(groups, width)


class _ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1),
                                    _group_norm(outgoing), nn.SiLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1),
                                    _group_norm(outgoing), nn.SiLU())

    def forward(self, value):
        return self.layers(value)


class _ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = _ConvBlock(width, width)

    def forward(self, value):
        return value+self.block(value)


class _SpatialBackbone(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.core = nn.Sequential(_ResidualBlock(width), _ResidualBlock(width))
        elif kind == "unet":
            self.skip = _ConvBlock(width, width)
            self.down = nn.Sequential(nn.Conv2d(width, width*2, 3, stride=2, padding=1),
                                      _ConvBlock(width*2, width*2))
            self.decode = _ConvBlock(width*3, width)
        else:
            raise ValueError("backbone must be cnn or unet")

    def forward(self, value):
        if self.kind == "cnn":
            return self.core(value)
        skip = self.skip(value)
        low = self.down(skip)
        restored = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.decode(torch.cat((skip, restored), dim=1))


class ConvGRUCell(nn.Module):
    """Causal spatial recurrent cell, independent of any fixed frame interval."""
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(input_channels+hidden_channels, 2*hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(input_channels+hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, current, previous=None):
        if previous is None:
            previous = current.new_zeros((len(current), self.hidden_channels, *current.shape[-2:]))
        reset, update = torch.sigmoid(self.gates(torch.cat((current, previous), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((current, reset*previous), dim=1)))
        return (1-update)*previous+update*candidate


class SRStage2Generator(nn.Module):
    """Residual CNN/U-Net x4 field generator with optional causal surface history.

    coarse_inputs: standardized legal surface fields [B,T,C,H,W] (or [B,C,H,W]).
    coarse_background_mm: current legal reference-only background [B,1,H,W].
    current_condition: optional current GNSS/mask/count/distance maps [B,K,H,W].
    static_hr: optional standardized static terrain/coordinates [B,S,sH,sW].
    For convgru, time_offsets_hours [B,T] must be ordered <=0, with current=0.
    Valid frames alone are used; current must be available. For temporal='none',
    only the final surface frame enters the model. No future frame is accepted.

    The residual head starts at zero, giving exact bilinear-background output.
    There is no positivity clamp: normalized GNSS-minus-ZHD references can have
    small negative estimates, which are not interpreted as negative water vapour.
    """
    def __init__(self, in_channels, *, current_condition_channels=0, static_channels=0,
                 width=24, scale_factor=4, backbone="cnn", temporal="none",
                 noise_channels=0, residual_scale_mm=20.):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1 or not isinstance(width, int) or width < 4:
            raise ValueError("Positive integer input channels and width>=4 are required")
        if any(not isinstance(x, int) or x < 0 for x in (current_condition_channels, static_channels, noise_channels)):
            raise ValueError("Condition/static/noise channel counts must be nonnegative integers")
        if scale_factor not in (1, 2, 4, 8) or temporal not in ("none", "convgru"):
            raise ValueError("Expected scale 1/2/4/8 and temporal none/convgru")
        if not math.isfinite(residual_scale_mm) or residual_scale_mm <= 0:
            raise ValueError("Residual scale must be positive and finite")
        self.in_channels = in_channels
        self.current_condition_channels = current_condition_channels
        self.static_channels = static_channels
        self.noise_channels = noise_channels
        self.scale_factor = int(scale_factor)
        self.temporal = temporal
        self.residual_scale_mm = float(residual_scale_mm)
        self.surface_encoder = _ConvBlock(in_channels+(temporal == "convgru"), width)
        self.recurrent = ConvGRUCell(width, width) if temporal == "convgru" else None
        # Current GNSS cannot accidentally be fed to the recurrent state via
        # this branch; normalized scalar background is also current-only.
        self.current_fusion = _ConvBlock(width+1+current_condition_channels+noise_channels, width)
        self.spatial = _SpatialBackbone(width, backbone)
        self.upsample_blocks = nn.ModuleList(_ConvBlock(width, width) for _ in range(int(math.log2(scale_factor))))
        self.hr_fusion = _ConvBlock(width+static_channels, width)
        self.residual_head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, coarse_inputs, coarse_background_mm, static_hr=None, *, current_condition=None,
                frame_valid=None, time_offsets_hours=None, noise=None, return_components=False):
        if coarse_inputs.ndim == 4:
            coarse_inputs = coarse_inputs[:, None]
        if coarse_inputs.ndim != 5 or coarse_inputs.shape[2] != self.in_channels:
            raise ValueError("coarse_inputs must be [B,T,C,H,W] with declared surface channel count")
        batch, frames, _, height, width = coarse_inputs.shape
        if frames < 1 or height < 2 or width < 2:
            raise ValueError("At least one frame and a spatial grid of at least 2x2 are required")
        expected_background = (batch, 1, height, width)
        if tuple(coarse_background_mm.shape) != expected_background or not torch.isfinite(coarse_background_mm).all():
            raise ValueError("Finite coarse_background_mm [B,1,H,W] is required")
        if frame_valid is None:
            frame_valid = torch.ones((batch, frames), dtype=torch.bool, device=coarse_inputs.device)
        if frame_valid.dtype != torch.bool or tuple(frame_valid.shape) != (batch, frames) or not frame_valid[:, -1].all():
            raise ValueError("frame_valid must be Boolean [B,T] and current frame must be valid")
        if not torch.isfinite(coarse_inputs[frame_valid]).all():
            raise ValueError("Valid surface frames must be finite; imputation and availability masks belong in the data pipeline")
        safe = torch.where(frame_valid[:, :, None, None, None], coarse_inputs, torch.zeros_like(coarse_inputs))
        if self.temporal == "none":
            encoded = self.surface_encoder(safe[:, -1])
        else:
            if time_offsets_hours is None or tuple(time_offsets_hours.shape) != (batch, frames):
                raise ValueError("ConvGRU requires explicit time_offsets_hours [B,T], including irregular raw intervals")
            times = torch.where(frame_valid, time_offsets_hours, torch.zeros_like(time_offsets_hours))
            if (not torch.isfinite(times).all() or (times[frame_valid] > 0).any()
                    or not torch.allclose(times[:, -1], torch.zeros_like(times[:, -1]), rtol=0, atol=1e-6)):
                raise ValueError("Time offsets must be finite, past-or-current, and current=0")
            for b in range(batch):
                if (torch.diff(times[b][frame_valid[b]]) < 0).any():
                    raise ValueError("Surface histories must be ordered oldest to current")
            state = None
            for index in range(frames):
                elapsed = (times[:, index]/24.)[:, None, None, None].expand(-1, 1, height, width)
                encoded_frame = self.surface_encoder(torch.cat((safe[:, index], elapsed), dim=1))
                updated = self.recurrent(encoded_frame, state)
                if state is None:
                    state = torch.zeros_like(updated)
                state = torch.where(frame_valid[:, index, None, None, None], updated, state)
            encoded = state
        conditions = [encoded, coarse_background_mm/100.]
        if self.current_condition_channels:
            expected = (batch, self.current_condition_channels, height, width)
            if current_condition is None or tuple(current_condition.shape) != expected or not torch.isfinite(current_condition).all():
                raise ValueError("Current-only condition channels must match [B,K,H,W] and be finite")
            conditions.append(current_condition)
        elif current_condition is not None:
            raise ValueError("current_condition supplied without declared current_condition_channels")
        if self.noise_channels:
            expected = (batch, self.noise_channels, height, width)
            if noise is None:
                noise = coarse_inputs.new_zeros(expected)
            if tuple(noise.shape) != expected or not torch.isfinite(noise).all():
                raise ValueError("Noise map must match [B,Z,H,W] and be finite")
            conditions.append(noise)
        elif noise is not None:
            raise ValueError("Noise supplied without declared noise_channels")
        decoded = self.spatial(self.current_fusion(torch.cat(conditions, dim=1)))
        for block in self.upsample_blocks:
            decoded = block(F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False))
        output_size = (height*self.scale_factor, width*self.scale_factor)
        if self.static_channels:
            expected = (batch, self.static_channels, *output_size)
            if static_hr is None or tuple(static_hr.shape) != expected or not torch.isfinite(static_hr).all():
                raise ValueError("Static high-resolution channels must match the full output grid and be finite")
            decoded = torch.cat((decoded, static_hr), dim=1)
        elif static_hr is not None:
            raise ValueError("static_hr supplied without declared static_channels")
        residual = self.residual_scale_mm*self.residual_head(self.hr_fusion(decoded))
        baseline = F.interpolate(coarse_background_mm, size=output_size, mode="bilinear", align_corners=False)
        field = baseline+residual
        if return_components:
            return {"field_mm": field, "baseline_mm": baseline, "residual_mm": residual}
        return field


class ConditionalObservationDiscriminator(nn.Module):
    """Mask-aware conditional set critic in the OBSERVED subspace only.

    real zwd_mm is measured training-target GNSS; fake zwd_mm is generated field
    passed through the identical station observation operator. Both calls MUST
    share coordinates, masks and legal condition vectors. This is not a full
    image critic and cannot establish correctness at unobserved HR pixels.

    condition [B,N,F] contains sampled legal surface/current-reference context,
    never the observed target value or any function of that value. Set pooling
    is permutation invariant and padding insensitive; each sample must contain
    at least one real observation. Output [B] is a logit/critic score, no sigmoid.
    """
    def __init__(self, condition_dim, *, width=64, value_scale_mm=100.):
        super().__init__()
        if not isinstance(condition_dim, int) or condition_dim < 0 or width < 4:
            raise ValueError("Nonnegative condition_dim and width>=4 are required")
        if not math.isfinite(value_scale_mm) or value_scale_mm <= 0:
            raise ValueError("value_scale_mm must be finite and positive")
        self.condition_dim = condition_dim
        self.value_scale_mm = float(value_scale_mm)
        self.point = nn.Sequential(nn.Linear(condition_dim+3, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.pool = nn.Sequential(nn.Linear(width*3, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, zwd_mm, condition, query_xy, valid):
        if zwd_mm.ndim == 3 and zwd_mm.shape[-1] == 1:
            zwd_mm = zwd_mm.squeeze(-1)
        if zwd_mm.ndim != 2 or tuple(condition.shape) != (*zwd_mm.shape, self.condition_dim):
            raise ValueError("Expected values [B,N] and condition [B,N,F]")
        safe_xy, valid = _query_mask(query_xy, valid, zwd_mm.shape[0])
        if tuple(valid.shape) != tuple(zwd_mm.shape) or not valid.any(dim=1).all():
            raise ValueError("Every critic sample requires at least one observed target; skip empty sets")
        if not torch.isfinite(zwd_mm[valid]).all() or not torch.isfinite(condition[valid]).all():
            raise ValueError("Valid critic values and conditions must be finite")
        value = torch.where(valid, zwd_mm, torch.zeros_like(zwd_mm))/self.value_scale_mm
        context = torch.where(valid[..., None], condition, torch.zeros_like(condition))
        feature = self.point(torch.cat((value[..., None], context, safe_xy), dim=-1))
        mask = valid[..., None]
        count = valid.sum(dim=1, keepdim=True).to(feature.dtype)
        mean = torch.where(mask, feature, torch.zeros_like(feature)).sum(dim=1)/count
        second = torch.where(mask, feature.square(), torch.zeros_like(feature)).sum(dim=1)/count
        variance = (second-mean.square()).clamp_min(0)
        maximum = feature.masked_fill(~mask, -torch.inf).amax(dim=1)
        return self.pool(torch.cat((mean, variance, maximum), dim=-1)).squeeze(-1)
