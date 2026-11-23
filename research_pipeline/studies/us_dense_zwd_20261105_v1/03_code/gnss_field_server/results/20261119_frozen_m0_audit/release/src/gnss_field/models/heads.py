"""Distributions on whole-field coefficients, expressed in normalized units."""
import torch
from torch import nn
import math


class ResidualBackbone(nn.Module):
    """Four active blocks give all coefficient controls comparable depth."""
    def __init__(self, inputs, width):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                    nn.SiLU(), nn.Linear(width, width)) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU())

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = (x + block(x))/math.sqrt(2)
        return self.output(x)


class DeterministicHead(nn.Module):
    def __init__(self, rank, condition_dim=128, width=256):
        super().__init__()
        self.backbone = ResidualBackbone(condition_dim, width)
        self.output = nn.Linear(width, rank)

    def forward(self, h):
        return self.output(self.backbone(h))


class GaussianHead(nn.Module):
    def __init__(self, rank, condition_dim=128, width=256):
        super().__init__()
        self.backbone = ResidualBackbone(condition_dim, width)
        self.mean = nn.Linear(width, rank)
        self.log_std = nn.Linear(width, rank)

    def forward(self, h):
        hidden = self.backbone(h)
        return self.mean(hidden), self.log_std(hidden).clamp(-5, 3)


class GANGenerator(nn.Module):
    def __init__(self, rank, condition_dim=128, width=256):
        super().__init__()
        self.backbone = ResidualBackbone(rank + condition_dim, width)
        self.output = nn.Linear(width, rank)

    def forward(self, noise, h):
        return self.output(self.backbone(torch.cat([noise, h], -1)))


class ConditionalCritic(nn.Module):
    def __init__(self, rank, condition_dim=128, width=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(rank + condition_dim, width), nn.LeakyReLU(.2),
                                     nn.Linear(width, width), nn.LeakyReLU(.2),
                                     nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, c, h):
        return self.network(torch.cat([c, h], -1)).squeeze(-1)
