"""Whole-grid learned basis normalized on the fixed geographical land mask."""
import math

import torch
from torch import nn


class SpatialBasis(nn.Module):
    def __init__(self, rank=128, width=256):
        super().__init__()
        if width < 256 or not 1 <= rank <= width:
            raise ValueError('Basis rank must be positive and supported by hidden width >=256')
        self.rank = rank
        self.register_buffer('wavelengths', torch.tensor([50., 100., 200., 400., 800., 1600.]) * 1000)
        self.network = nn.Sequential(nn.Linear(27, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, rank))

    def forward(self, xyh, land_mask):
        if xyh.ndim != 2 or xyh.shape[1] != 3 or land_mask.shape != xyh.shape[:1]:
            raise ValueError('Expected physical xyh[G,3] and fixed land mask[G]')
        if land_mask.dtype != torch.bool or not land_mask.any():
            raise ValueError('A nonempty fixed boolean land mask is required')
        scaled = xyh / xyh.new_tensor([400000., 400000., 3000.])
        phase = 2 * math.pi * xyh[:, :2, None] / self.wavelengths
        features = torch.cat([scaled, phase.sin().flatten(1), phase.cos().flatten(1)], 1)
        values = self.network(features)
        rms = values[land_mask].square().mean(0).clamp_min(1e-16).sqrt()
        return values / rms
