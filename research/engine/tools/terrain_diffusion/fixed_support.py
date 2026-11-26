"""Minimal fixed-support encoder prototype for the next foundation-field screen."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def height_transport(zwd: torch.Tensor, condition_height: torch.Tensor,
                     query_height: torch.Tensor, scale_km: float = 3.0) -> torch.Tensor:
    """Transport surface ZWD between station heights with an explicit scale."""
    return zwd * torch.exp((condition_height - query_height) / (scale_km * 1000.0))


def masked_multiscale_summary(query_xy: torch.Tensor, query_height: torch.Tensor,
                              condition_xy: torch.Tensor, condition_height: torch.Tensor,
                              values: torch.Tensor, mask: torch.Tensor,
                              radii_km: tuple[float, ...] = (50.0, 150.0, 300.0),
                              scale_km: float = 3.0) -> torch.Tensor:
    """Return finite per-query summaries for masked station observations.

    ``values`` and ``mask`` are [time, station]; the latest valid value per
    station is used. The output is 10 channels per radius: raw mean/std/count,
    height-transported mean/std, and five support diagnostics.
    """
    qxy = query_xy.float(); cxy = condition_xy.float()
    qh = query_height.float().reshape(-1, 1); ch = condition_height.float().reshape(1, -1)
    v = values.float(); m = mask.bool() & torch.isfinite(v)
    latest = torch.zeros(v.shape[1], device=v.device, dtype=v.dtype)
    available = torch.zeros(v.shape[1], device=v.device, dtype=torch.bool)
    for t in range(v.shape[0]):
        take = m[t]
        latest = torch.where(take, v[t], latest)
        available |= take
    d = torch.cdist(qxy, cxy)
    valid = available.reshape(1, -1)
    pieces = []
    for radius in radii_km:
        w = torch.exp(-0.5 * (d / radius).square()) * (d <= 3 * radius)
        w = w * valid.float()
        den = w.sum(-1, keepdim=True)
        safe = den.clamp_min(1e-8)
        mean = (w * latest.reshape(1, -1)).sum(-1, keepdim=True) / safe
        var = ((w * (latest.reshape(1, -1) - mean).square()).sum(-1, keepdim=True) / safe).clamp_min(0)
        transported = height_transport(latest.reshape(1, -1), ch, qh, scale_km)
        tmean = (w * transported).sum(-1, keepdim=True) / safe
        nearest = d.masked_fill(~valid, float('inf')).min(-1).values.clamp_max(600).reshape(-1, 1)
        support = (w > 0).sum(-1, keepdim=True).float()
        hdiff = (w * (ch - qh).abs()).sum(-1, keepdim=True) / safe
        hspread = ((w * ((ch - qh).abs() - hdiff).square()).sum(-1, keepdim=True) / safe).clamp_min(0).sqrt()
        pieces.append(torch.cat([mean, var.sqrt(), den.log1p(), tmean, nearest / 100.0,
                                 support.log1p(), hdiff / 1000.0, hspread / 1000.0,
                                 (den > 0).float(), (support > 0).float()], -1))
    return torch.cat(pieces, -1)


class _Conv(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FixedSupportModel(nn.Module):
    """Point readout from a fixed support map plus a zero-initialized U-Net correction."""
    def __init__(self, point_dim: int = 113, summary_dim: int = 30, width: int = 16):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(point_dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.proj = nn.Conv2d(summary_dim, width, 1)
        self.down = _Conv(width, width * 2)
        self.up = _Conv(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.point[-1].weight); nn.init.zeros_(self.point[-1].bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, point: torch.Tensor, support: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        x = self.proj(support)
        low = self.down(F.avg_pool2d(x, 2))
        low = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)
        correction = self.head(self.up(torch.cat([x, low], 1)))
        sampled = F.grid_sample(correction, uv, mode='bilinear', padding_mode='border', align_corners=True).flatten(1)
        if point.ndim == 2:
            sampled = sampled[0]
        return self.point(point).squeeze(-1) + sampled
