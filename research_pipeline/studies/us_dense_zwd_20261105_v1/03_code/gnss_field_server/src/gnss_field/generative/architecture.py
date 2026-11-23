"""Fresh conditional full-field architectures with sparse station supervision.

Tensor coordinates are projected metres; field values are common-height mm.
Only pure keyed-noise and ODE integration utilities are shared with older models.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..models.diffusion import heun_sample, keyed_noise
from .observations import query_features


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


@dataclass
class EncodedCondition:
    pyramid: tuple[torch.Tensor, ...]
    global_features: torch.Tensor


class _GridResidual(nn.Module):
    """Preactivation GroupNorm residual block; no batch-dependent statistics."""
    def __init__(self, source, destination):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, source)
        self.conv1 = nn.Conv2d(source, destination, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, destination)
        self.conv2 = nn.Conv2d(destination, destination, 3, padding=1)
        self.shortcut = nn.Identity() if source == destination else nn.Conv2d(source, destination, 1)

    def forward(self, x):
        hidden = self.conv1(F.silu(self.norm1(x)))
        return (self.shortcut(x) + self.conv2(F.silu(self.norm2(hidden)))) / math.sqrt(2)


class ConditionEncoder(nn.Module):
    """37-channel grid pyramid with four-head geometric attention to A tokens."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(37, 32, 3, padding=1), _GridResidual(32, 32))
        widths = (32, 64, 96, 128, 192)
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(a, b, 3, stride=2, padding=1), _GridResidual(b, b))
                                   for a, b in zip(widths[:-1], widths[1:])])
        self.station_mlp = nn.Sequential(nn.Linear(31, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.query = nn.Linear(96, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.attention_output = nn.Linear(64, 96)
        self.geometry_bias = nn.Linear(2, 4, bias=False)
        self.null_token = nn.Parameter(torch.zeros(1, 1, 64))

    def _attend(self, grid, tokens, mask, grid_geometry, input_geometry):
        batch, channels, height, width = grid.shape
        clean = torch.where(mask[..., None], tokens, 0.)
        _finite(clean, 'Valid station tokens')
        station = self.station_mlp(clean)
        station = torch.cat((station, self.null_token.expand(batch, -1, -1)), 1)
        # The learned null token participates only for a genuinely empty set.
        mask = torch.cat((mask, ~mask.any(-1, keepdim=True)), 1)
        q = self.query(grid.flatten(2).transpose(1, 2)).reshape(batch, -1, 4, 16).transpose(1, 2)
        k = self.key(station).reshape(batch, -1, 4, 16).transpose(1, 2)
        v = self.value(station).reshape(batch, -1, 4, 16).transpose(1, 2)
        grid_xyz = F.interpolate(grid_geometry.permute(2, 0, 1)[None], size=(height, width), mode='bilinear', align_corners=False)
        grid_xyz = grid_xyz[0].flatten(1).transpose(0, 1)
        displacement = grid_xyz[:, None] - input_geometry[None]
        distance = torch.linalg.vector_norm(displacement[..., :2], dim=-1) / 150000.
        geometry = torch.stack((distance, displacement[..., 2] / 3000.), -1)
        bias = self.geometry_bias(geometry).permute(2, 0, 1)
        bias = F.pad(bias, (0, 1))  # Null geometry has zero distance/height bias.
        logits = (q @ k.transpose(-1, -2)) / 4. + bias[None]
        attention = logits.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, height * width, 64)
        return grid + self.attention_output(attended).transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, grid_condition, station_tokens, station_mask, grid_geometry, input_geometry):
        if grid_condition.ndim != 4 or grid_condition.shape[1] != 37:
            raise ValueError('Grid condition must be B37HW')
        batch, _, height, width = grid_condition.shape
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError('Grid height/width must be positive multiples of sixteen')
        if station_tokens.ndim != 3 or station_tokens.shape[0] != batch or station_tokens.shape[2] != 31:
            raise ValueError('Station tokens must be BA31')
        if station_mask.shape != station_tokens.shape[:2] or station_mask.dtype != torch.bool:
            raise ValueError('Station mask must be boolean BA')
        if grid_geometry.shape != (height, width, 3) or input_geometry.shape != (station_tokens.shape[1], 3):
            raise ValueError('Geometry must be HW3 and A3')
        for value, name in ((grid_condition, 'Grid condition'), (grid_geometry, 'Grid geometry'), (input_geometry, 'Input geometry')):
            _finite(value, name)
        pyramid = [self.stem(grid_condition)]
        for index, layer in enumerate(self.down):
            hidden = layer(pyramid[-1])
            if index == 1:
                hidden = self._attend(hidden, station_tokens, station_mask, grid_geometry, input_geometry)
            pyramid.append(hidden)
        global_features = pyramid[-1].mean((-2, -1))
        _finite(global_features, 'Encoded condition')
        return EncodedCondition(tuple(pyramid), global_features)


class FieldDecoder(nn.Module):
    """Nonlinear common-height residual decoder with latent FiLM at every scale."""
    def __init__(self, latent_dim=64, field_scale_mm=10., grid_shape=(144, 96)):
        super().__init__()
        if latent_dim < 1 or len(grid_shape) != 2 or min(grid_shape) < 16 or any(x % 16 for x in grid_shape):
            raise ValueError('Latent dimension and grid shape must be valid')
        if field_scale_mm <= 0 or not math.isfinite(float(field_scale_mm)):
            raise ValueError('Field scale must be positive finite')
        self.latent_dim = latent_dim
        self.grid_shape = tuple(grid_shape)
        self.field_scale_mm = field_scale_mm
        self.latent = nn.Sequential(nn.Linear(latent_dim, 256), nn.SiLU(), nn.Linear(256, 32 * (grid_shape[0] // 16) * (grid_shape[1] // 16)))
        self.blocks = nn.ModuleList([_GridResidual(a, b) for a, b in ((224, 192), (320, 128), (224, 96), (160, 64), (96, 32))])
        self.film = nn.ModuleList([nn.Linear(latent_dim, 2 * width) for width in (192, 128, 96, 64, 32)])
        self.output = nn.Conv2d(32, 1, 1)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, encoded, z, background):
        if z.ndim != 3 or z.shape[2] != self.latent_dim or z.shape[0] != encoded.global_features.shape[0] or z.shape[1] < 1:
            raise ValueError('Latents must be BKR and match condition batch')
        batch, members = z.shape[:2]
        if background.shape != (batch, 1, *self.grid_shape) or len(encoded.pyramid) != 5:
            raise ValueError('Background/condition must match configured published grid')
        _finite(z, 'Latents')
        _finite(background, 'Background')
        flat_z = z.flatten(0, 1)
        hidden = self.latent(flat_z).reshape(batch * members, 32, self.grid_shape[0] // 16, self.grid_shape[1] // 16)
        for block, film, condition in zip(self.blocks, self.film, reversed(encoded.pyramid)):
            if hidden.shape[-2:] != condition.shape[-2:]:
                hidden = F.interpolate(hidden, size=condition.shape[-2:], mode='bilinear', align_corners=False)
            skip = condition[:, None].expand(-1, members, -1, -1, -1).flatten(0, 1)
            hidden = block(torch.cat((hidden, skip), 1))
            scale, shift = film(flat_z).chunk(2, -1)
            hidden = hidden * (1 + scale[..., None, None]) + shift[..., None, None]
        fields = background[:, None] + self.field_scale_mm * self.output(hidden).reshape(batch, members, 1, *self.grid_shape)
        _finite(fields, 'Decoded fields')
        return fields


class _Posterior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.local = nn.Linear(32, 64)
        self.point = nn.Sequential(nn.Linear(69, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(448, 256), nn.SiLU(), nn.Linear(256, 2 * latent_dim))

    def forward(self, encoded, residual_surface, indices, weights, geometry, mask):
        if mask.ndim != 2 or mask.dtype != torch.bool or residual_surface.shape != mask.shape or geometry.shape != (mask.shape[1], 3):
            raise ValueError('Posterior expects BN residual/mask and N3 geometry')
        if not mask.any(-1).all():
            raise ValueError('Posterior training rejects empty valid-observation sets')
        residual = torch.where(mask, residual_surface, 0.)
        _finite(residual, 'Valid posterior residual')
        _finite(geometry, 'Target geometry')
        local = self.local(query_features(encoded.pyramid[0], indices, weights))
        normalized = geometry / geometry.new_tensor([150000., 150000., 3000.])
        features = torch.cat((local, residual[..., None] / 10., normalized[None].expand(len(mask), -1, -1), mask[..., None].to(local.dtype)), -1)
        points = self.point(features)
        mean = torch.where(mask[..., None], points, 0.).sum(1) / mask.sum(1, keepdim=True)
        maximum = points.masked_fill(~mask[..., None], -torch.inf).amax(1)
        return self.output(torch.cat((mean, maximum, encoded.global_features), -1))


class ConditionalFieldVAE(nn.Module):
    def __init__(self, latent_dim=64, grid_shape=(144, 96)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConditionEncoder()
        self.decoder = FieldDecoder(latent_dim=latent_dim, grid_shape=grid_shape)
        self.prior = nn.Sequential(nn.Linear(192, 256), nn.SiLU(), nn.Linear(256, 2 * latent_dim))
        self.posterior = _Posterior(latent_dim)

    def encode(self, batch):
        return self.encoder(*(batch[key] for key in ('grid_condition', 'station_tokens', 'station_mask', 'grid_geometry', 'input_geometry')))

    @staticmethod
    def _gaussian_parameters(raw):
        mu, log_std = raw.chunk(2, -1)
        _finite(raw, 'Gaussian parameters')
        return mu, log_std.clamp(-6., 2.)

    def prior_parameters(self, encoded):
        return self._gaussian_parameters(self.prior(encoded.global_features))

    def posterior_parameters(self, encoded, residual_surface, query_indices, query_weights, target_geometry, target_mask):
        return self._gaussian_parameters(self.posterior(encoded, residual_surface, query_indices, query_weights, target_geometry, target_mask))

    def sample(self, encoded, background, noise):
        mu, log_std = self.prior_parameters(encoded)
        return self.decoder(encoded, mu[:, None] + log_std[:, None].exp() * noise, background)

    def posterior_sample(self, encoded, background, residual_surface, query_indices, query_weights, target_geometry, target_mask, noise):
        mu_q, logstd_q = self.posterior_parameters(encoded, residual_surface, query_indices, query_weights, target_geometry, target_mask)
        mu_p, logstd_p = self.prior_parameters(encoded)
        fields = self.decoder(encoded, mu_q[:, None] + logstd_q[:, None].exp() * noise, background)
        return fields, mu_q, logstd_q, mu_p, logstd_p


class _FiLMResidual(nn.Module):
    def __init__(self, width, embedding_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.film = nn.Linear(embedding_dim, 2 * width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x, embedding):
        hidden = self.linear1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, -1)
        hidden = self.norm2(hidden) * (1 + scale) + shift
        return (x + self.linear2(F.silu(hidden))) / math.sqrt(2)


class LatentGenerator(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=192, width=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.input = nn.Linear(latent_dim + condition_dim, width)
        self.blocks = nn.ModuleList([_FiLMResidual(width, latent_dim + condition_dim) for _ in range(4)])
        self.output = nn.Linear(width, latent_dim)

    def forward(self, noise, condition):
        if noise.ndim != 3 or noise.shape[-1] != self.latent_dim or condition.shape != (len(noise), self.condition_dim):
            raise ValueError('Generator expects BKR noise and BC condition')
        _finite(noise, 'Generator noise')
        _finite(condition, 'Generator condition')
        embedding = torch.cat((noise, condition[:, None].expand(-1, noise.shape[1], -1)), -1)
        hidden = self.input(embedding)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        result = self.output(hidden)
        _finite(result, 'Generated latents')
        return result


class LatentCritic(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=192):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.net = nn.Sequential(nn.Linear(latent_dim + condition_dim, 256), nn.LeakyReLU(.2),
                                 nn.Linear(256, 256), nn.LeakyReLU(.2), nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

    def forward(self, latent, condition):
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim or condition.shape != (len(latent), self.condition_dim):
            raise ValueError('Critic expects BR latents and BC condition')
        _finite(latent, 'Critic latent')
        _finite(condition, 'Critic condition')
        return self.net(torch.cat((latent, condition), -1)).squeeze(-1)


class LatentDenoiser(nn.Module):
    """EDM sigma_data=1 with 18 positive levels and a terminal zero (35 NFE)."""
    def __init__(self, latent_dim=64, condition_dim=192, width=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sigma_data = 1.
        self.register_buffer('time_frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 31))
        self.input = nn.Linear(latent_dim, width)
        self.embedding = nn.Sequential(nn.Linear(condition_dim + 64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([_FiLMResidual(width, width) for _ in range(4)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, latent_dim)

    def denoise(self, noisy, sigma, condition):
        if noisy.ndim != 2 or noisy.shape[-1] != self.latent_dim or condition.shape != (len(noisy), self.condition_dim):
            raise ValueError('Denoiser expects BR latent and BC condition')
        sigma = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(len(noisy))
        if sigma.shape != (len(noisy),) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError('Sigma must be positive finite scalar or B vector')
        _finite(noisy, 'Noisy latent')
        _finite(condition, 'Denoising condition')
        denominator = sigma.square() + 1.
        phase = sigma.log()[:, None] / 4 * self.time_frequencies[None]
        embedding = self.embedding(torch.cat((condition, phase.sin(), phase.cos()), -1))
        hidden = self.input(noisy * denominator.rsqrt()[:, None])
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.output(F.silu(self.output_norm(hidden)))
        result = noisy / denominator[:, None] + sigma[:, None] * denominator.rsqrt()[:, None] * residual
        _finite(result, 'Denoised latent')
        return result

    forward = denoise

    def sample(self, condition, hour_ids, members, seed, steps=18):
        hours = torch.as_tensor(hour_ids).reshape(-1)
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim or len(hours) != len(condition):
            raise ValueError('Sampling hours and conditions must match')
        noise = keyed_noise(hours, members, seed, shape=(self.latent_dim,), device=condition.device, dtype=condition.dtype)
        repeated = condition[:, None].expand(-1, members, -1).flatten(0, 1)
        result = heun_sample(lambda x, sigma: self.denoise(x, sigma, repeated), noise.flatten(0, 1), steps=steps)
        return result.reshape(len(condition), members, self.latent_dim)


class ObservationCritic(nn.Module):
    """Masked station-set critic with deterministic geometry-only distance bins."""
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(36, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.pair = nn.Sequential(nn.Linear(36, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 1))

    @staticmethod
    def _pairs(geometry):
        # Lexicographic physical order makes truncation invariant to input station order.
        order = torch.arange(len(geometry), device=geometry.device)
        for axis in (2, 1, 0):
            order = order[torch.argsort(geometry[order, axis], stable=True)]
        pairs = torch.triu_indices(len(order), len(order), 1, device=geometry.device)
        left, right = order[pairs[0]], order[pairs[1]]
        distance = torch.linalg.vector_norm(geometry[left, :2] - geometry[right, :2], dim=-1)
        return [(left[(distance >= low) & (distance < high)], right[(distance >= low) & (distance < high)])
                for low, high in ((0., 25000.), (25000., 50000.), (50000., 100000.), (100000., 200000.))]

    def forward(self, values, local_condition, geometry, mask):
        if values.ndim != 2 or local_condition.shape != (*values.shape, 32) or geometry.shape != (values.shape[1], 3) or mask.shape != values.shape or mask.dtype != torch.bool:
            raise ValueError('Observation critic expects BN, BN32, N3, boolean BN')
        _finite(geometry, 'Observation geometry')
        clean = torch.where(mask, values, 0.)
        local = torch.where(mask[..., None], local_condition, 0.)
        _finite(clean, 'Valid observed residuals')
        _finite(local, 'Valid local conditions')
        xyz = geometry / geometry.new_tensor([150000., 150000., 3000.])
        point = self.point(torch.cat((local, clean[..., None], xyz[None].expand(len(values), -1, -1)), -1))
        counts = mask.sum(1, keepdim=True).clamp_min(1)
        mean = torch.where(mask[..., None], point, 0.).sum(1) / counts
        variance = torch.where(mask[..., None], (point - mean[:, None]).square(), 0.).sum(1) / counts
        pair_bins = self._pairs(geometry)
        features, pair_masks = [], []
        rows = torch.arange(len(values), device=values.device)[:, None]
        for left, right in pair_bins:
            if not len(left):
                features.append(values.new_zeros((len(values), 256, 36)))
                pair_masks.append(mask.new_zeros((len(values), 256)))
                continue
            valid = mask[:, left] & mask[:, right]
            # Original geometry positions are unique, so sorted top-k selects
            # exactly the first 256 valid pairs, independently for every row.
            positions = torch.arange(len(left), device=values.device)[None].expand(len(values), -1)
            selected = positions.masked_fill(~valid, len(left)).topk(min(256, len(left)), dim=1, largest=False, sorted=True).indices
            selected_mask = valid.gather(1, selected)
            left, right = left[selected], right[selected]
            difference = clean[rows, left] - clean[rows, right]
            distance = torch.linalg.vector_norm(geometry[left, :2] - geometry[right, :2], dim=-1) / 150000.
            height = (geometry[left, 2] - geometry[right, 2]) / 3000.
            pair_features = torch.cat((local[rows, left] - local[rows, right], difference[..., None], difference.abs()[..., None], distance[..., None], height[..., None]), -1)
            padding = 256 - selected.shape[1]
            features.append(F.pad(pair_features, (0, 0, 0, padding)))
            pair_masks.append(F.pad(selected_mask, (0, padding)))
        pair_mask = torch.stack(pair_masks, 1)
        pair_points = self.pair(torch.stack(features, 1))
        pooled = torch.where(pair_mask[..., None], pair_points, 0.).sum(2) / pair_mask.sum(2, keepdim=True).clamp_min(1)
        result = self.output(torch.cat((mean, variance, pooled.flatten(1)), -1)).squeeze(-1)
        _finite(result, 'Observation critic score')
        return result
