"""A station-set and coarse-grid encoder for whole-domain coefficient heads."""
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ContextBatch
from .legacy_core import GeometryAttention


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim=128):
        super().__init__()
        self.station_encoder = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 64))
        self.amplitude = nn.Linear(5, 64, bias=False)
        self.null_station = nn.Parameter(torch.zeros(1, 1, 64))
        self.coarse_encoder = nn.Sequential(nn.Conv2d(60, 32, 3, padding=1), nn.SiLU(),
                                            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.query_features = nn.Conv2d(32, 64, 1)
        self.attention = GeometryAttention(64)
        self.fusion = nn.Sequential(nn.Conv2d(103, 64, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.projection = nn.Linear(1024, condition_dim)

    def forward(self, context: ContextBatch):
        stations = context.stations.masked_fill(context.padding[..., None], 0)
        batch = len(stations)
        tokens = self.station_encoder(stations) + self.amplitude(stations[..., :5])
        tokens = torch.cat([tokens, self.null_station.expand(batch, -1, -1)], 1)
        positions = torch.cat([stations[..., 10:13], stations.new_zeros(batch, 1, 3)], 1)
        padding = torch.cat([context.padding, ~context.padding.all(1, keepdim=True)], 1)
        coarse = self.coarse_encoder(context.coarse.flatten(1, 2))
        size = coarse.shape[-2:]
        maps = F.interpolate(context.maps, size=size, mode='bilinear', align_corners=False)
        geometry = maps[:, [1, 2, 0]]
        prepared = self.attention.prepare(tokens, positions, padding, geometry)
        attended = self.attention(self.query_features(coarse), prepared)
        mean = context.mean if context.mean is not None else context.baseline
        mean = F.interpolate(mean, size=size, mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([attended, coarse, mean, maps], 1))
        return self.projection(F.adaptive_avg_pool2d(fused, (4, 4)).flatten(1))
