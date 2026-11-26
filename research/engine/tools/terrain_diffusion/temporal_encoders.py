"""Same fixed spatial carrier; replace only the causal temporal encoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalFoundation(nn.Module):
    def __init__(self, kind, length, channels=25, width=16, hidden=64, side=8):
        super().__init__()
        self.kind, self.length, self.side = kind, length, side
        self.spatial = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1),
                                     nn.GroupNorm(4, width), nn.SiLU())
        self.down = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU())
        self.fuse = nn.Conv2d(width*2, width, 1)
        if kind in ('gru', 'lstm'):
            self.temporal = (nn.GRU if kind == 'gru' else nn.LSTM)(width, hidden, 2, batch_first=True)
        elif kind == 'transformer':
            self.input_projection = nn.Linear(width, hidden)
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout=0, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        elif kind == 'mlp':
            self.temporal = nn.Sequential(nn.Linear(length*width, hidden), nn.SiLU())
        elif kind == 'none':
            self.temporal = nn.Linear(width, hidden)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden, 6)
        self.query_head = nn.Sequential(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 6))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.query_head[-1].weight); nn.init.zeros_(self.query_head[-1].bias)

    def encode_time(self, x):
        b, length, m, width = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b*m, length, width)
        if self.kind in ('gru', 'lstm'):
            encoded, _ = self.temporal(seq)
        elif self.kind == 'transformer':
            encoded = self.input_projection(seq)
            h = encoded.shape[-1]
            pos = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
            freq = torch.exp(torch.arange(0, h, 2, device=x.device, dtype=x.dtype)*(-math.log(10000)/h))
            pe = torch.zeros(length, h, device=x.device, dtype=x.dtype)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
            mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
            encoded = self.temporal(encoded+pe, mask=mask)
        else:
            raise ValueError('encode_time is reserved for sequential encoders')
        return encoded.reshape(b, m, length, -1).permute(0, 2, 1, 3)

    def field(self, history):
        b, length, m, channels = history.shape
        maps = history.permute(0, 1, 3, 2).reshape(b*length, channels, self.side, self.side)
        spatial = self.spatial(maps)
        low = F.interpolate(self.down(F.avg_pool2d(spatial, 2)), size=spatial.shape[-2:], mode='bilinear', align_corners=False)
        spatial = self.fuse(torch.cat([spatial, low], 1))
        x = spatial.reshape(b, length, -1, m).permute(0, 1, 3, 2)
        if self.kind == 'none':
            state = self.temporal(x[:, -1])
        elif self.kind == 'mlp':
            state = self.temporal(x.permute(0, 2, 1, 3).reshape(b, m, -1))
        else:
            state = self.encode_time(x)[:, -1]
        field = self.head(state).transpose(1, 2).reshape(b, 6, self.side, self.side)
        # First channel is current sea-level ZWD / 100; six-hour forecast adds
        # learned deltas. No future GNSS/weather is read.
        background = history[:, -1, :, 0].reshape(b, 1, self.side, self.side)
        return field + background

    def forward(self, history, uv, query_static, background=None):
        field = self.field(history)
        if background is not None:
            field = field - history[:, -1, :, 0].reshape(history.shape[0], 1, self.side, self.side)
        query = F.grid_sample(field, uv[:, :, None], align_corners=True, padding_mode='border').squeeze(-1)
        # Work in station-surface ZWD / 100, with fixed 3000 m transfer prior.
        height = query_static[:, :, 2] * 3000
        query = query * torch.exp(-height[:, None]/3000)
        if background is not None:
            query = query + background[:, None]
        return query + self.query_head(query_static).transpose(1, 2)
