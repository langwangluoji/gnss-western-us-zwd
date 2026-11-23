"""Matched full-grid mean processors; probability factors live outside the heads."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.GroupNorm(8, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(8, outputs))
        self.skip = nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)

    def forward(self, x):
        return F.silu(self.layers(x)+self.skip(x))


class GridProcessor(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ('mlp69', 'resunet69'):
            raise ValueError('Unknown grid processor')
        self.arm = arm
        if arm == 'mlp69':
            self.processor = nn.Sequential(nn.Conv2d(69,64,1),nn.SiLU(),
                nn.Conv2d(64,32,1),nn.SiLU())
        else:
            widths = [32,64,128,128]
            self.encoders = nn.ModuleList([ResidualBlock(69,32)] +
                [ResidualBlock(widths[i],widths[i]) for i in range(1,4)])
            self.down = nn.ModuleList([nn.Conv2d(widths[i],widths[i+1],3,stride=2,padding=1)
                                      for i in range(3)])
            self.decoders = nn.ModuleList([ResidualBlock(widths[i+1]+widths[i],widths[i])
                                           for i in (2,1,0)])
        self.output = nn.Conv2d(32,1,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, available=None):
        if features.ndim != 4 or features.shape[1] != 69:
            raise ValueError('Expected [B,69,H,W] spatial conditions')
        if self.arm == 'mlp69':
            x = self.processor(features)
        else:
            x = self.encoders[0](features)
            levels = [x]
            for down, block in zip(self.down,self.encoders[1:]):
                x = block(down(x)); levels.append(x)
            for block, skip in zip(self.decoders,reversed(levels[:-1])):
                x = F.interpolate(x,size=skip.shape[-2:],mode='nearest')
                x = block(torch.cat([x,skip],1))
        field = self.output(x)
        if available is not None:
            field = field*available[:,None,None,None]
        return field


def query_field(field, indices, weights):
    if not torch.isfinite(field).all():
        raise ValueError('Nonfinite published grid')
    return (field.flatten(1)[:,indices]*weights).sum(-1)
