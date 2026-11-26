"""Small geometry-conditioned residual models and EDM sampling in normalized Z0."""
import hashlib
import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


def edm_coefficients(sigma, sigma_data=.5):
    """Return c_skip, c_out, c_in, c_noise and the weighted-MSE coefficient."""
    denominator = sigma.square() + sigma_data**2
    return (sigma_data**2 / denominator,
            sigma*sigma_data/denominator.sqrt(), denominator.rsqrt(),
            sigma.log()/4, denominator/(sigma*sigma_data).square())


def keyed_noise(hour_ids, members, seed, shape=(1, 24, 24), device='cpu', dtype=torch.float32):
    """CPU-generated keyed streams give stable prefixes and batch-independent draws."""
    hours = torch.as_tensor(hour_ids).detach().cpu().reshape(-1).tolist()
    if members < 1 or not hours:
        raise ValueError('At least one hour and ensemble member is required')
    rows = []
    for hour in hours:
        ensemble = []
        for member in range(members):
            identity = f'{int(seed)}:{int(hour)}:{member}'.encode('ascii')
            value = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') & ((1 << 63)-1)
            generator = torch.Generator(device='cpu').manual_seed(value)
            ensemble.append(torch.randn(shape, generator=generator, dtype=torch.float32))
        rows.append(torch.stack(ensemble))
    return torch.stack(rows).to(device=device, dtype=dtype)


def heun_sample(denoise: Callable, initial_noise, domain, steps=18,
                sigma_min=.002, sigma_max=80., rho=7.):
    """Deterministic EDM ODE solver; denoise accepts (state, scalar sigma)."""
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError('Invalid EDM noise schedule')
    dtype, device = initial_noise.dtype, initial_noise.device
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    schedule = (sigma_max**(1/rho) + ramp*(sigma_min**(1/rho)-sigma_max**(1/rho)))**rho
    schedule = torch.cat([schedule, schedule.new_zeros(1)])
    mask = domain.to(device=device, dtype=dtype)
    current = initial_noise*schedule[0]*mask
    for index in range(steps):
        sigma, following = schedule[index], schedule[index+1]
        clean = denoise(current, sigma)*mask
        derivative = (current-clean)/sigma
        predicted = (current+(following-sigma)*derivative)*mask
        if index < steps-1:
            corrected = denoise(predicted, following)*mask
            next_derivative = (predicted-corrected)/following
            predicted = (current+(following-sigma)*(derivative+next_derivative)/2)*mask
        current = predicted
    return current


class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.film = nn.Linear(128, channels_out*2)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, value, embedding):
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        hidden = self.norm2(hidden)*(1+scale[..., None, None])+shift[..., None, None]
        return (self.skip(value)+self.conv2(F.silu(hidden)))/math.sqrt(2)


class GeometryAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, 64, 1)
        self.query_position = nn.Linear(3, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.geometry = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 4))
        self.output = nn.Conv2d(64, channels, 1)

    def prepare(self, tokens, positions, padding, query_geometry):
        b, n = tokens.shape[:2]
        key = self.key(tokens).reshape(b, n, 4, 16).transpose(1, 2)
        value = self.value(tokens).reshape(b, n, 4, 16).transpose(1, 2)
        qpos = query_geometry.flatten(2).transpose(1, 2)
        relative = qpos[:, :, None, :]-positions[:, None, :, :]
        distance = relative[..., :2].square().sum(-1, keepdim=True).sqrt()
        bias = self.geometry(torch.cat([relative, distance], -1)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(padding[:, None, None, :], float('-inf'))
        return dict(key=key, value=value, bias=bias, query_position=self.query_position(qpos))

    def forward(self, hidden, prepared):
        b, _, h, w = hidden.shape
        query = self.query(hidden).flatten(2).transpose(1, 2)+prepared['query_position']
        query = query.reshape(b, h*w, 4, 16).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, prepared['key'], prepared['value'],
                                                  attn_mask=prepared['bias'], dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(b, h*w, 64).transpose(1, 2).reshape(b, 64, h, w)
        return (hidden+self.output(attended))/math.sqrt(2)


class ResidualModel(nn.Module):
    """Matched deterministic, independent-Gaussian and EDM residual branches."""
    sigma_data = .5

    def __init__(self, arm='edm'):
        super().__init__()
        if arm not in ('det', 'gaussian', 'edm'):
            raise ValueError(f'Unknown arm: {arm}')
        self.arm = arm
        self.station_encoder = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 64))
        self.amplitude = nn.Linear(5, 64, bias=False)
        self.null_station = nn.Parameter(torch.zeros(1, 1, 64))
        self.coarse_encoder = nn.Sequential(nn.Conv2d(60, 32, 3, padding=1), nn.SiLU(),
                                            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.noise_encoder = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.register_buffer('noise_frequencies', torch.exp(-math.log(10000)*torch.arange(32)/31))
        self.stem = nn.Conv2d(42, 32, 3, padding=1)
        self.enc24 = nn.ModuleList([ResidualBlock(32, 32), ResidualBlock(32, 32)])
        self.down12 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc12 = nn.ModuleList([ResidualBlock(64, 64), ResidualBlock(64, 64)])
        self.attn12 = GeometryAttention(64)
        self.down6 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc6 = nn.ModuleList([ResidualBlock(128, 128), ResidualBlock(128, 128)])
        self.attn6 = GeometryAttention(128)
        self.up12 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec12 = nn.ModuleList([ResidualBlock(128, 64), ResidualBlock(64, 64)])
        self.up24 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec24 = nn.ModuleList([ResidualBlock(64, 32), ResidualBlock(32, 32)])
        self.output = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 2 if arm=='gaussian' else 1, 3, padding=1))

    def encode_condition(self, batch):
        coarse = batch['coarse']
        b = coarse.shape[0]
        if coarse.shape[1:] != (4, 15, 6, 6) or batch['stations'].shape[1:] != (39, 17):
            raise ValueError('Expected coarse Bx4x15x6x6 and stations Bx39x17')
        domain = batch['domain'].to(device=coarse.device, dtype=coarse.dtype).expand(b, -1, -1, -1)
        padding = batch['padding'].bool()
        stations = batch['stations'].masked_fill(padding[..., None], 0)
        tokens = self.station_encoder(stations)+self.amplitude(stations[..., :5])
        tokens = torch.cat([tokens, self.null_station.expand(b, -1, -1)], 1)
        positions = torch.cat([stations[..., 10:13], stations.new_zeros(b, 1, 3)], 1)
        padding = torch.cat([padding, ~padding.all(1, keepdim=True)], 1)
        features = F.interpolate(self.coarse_encoder(coarse.flatten(1, 2)), size=(24, 24), mode='bilinear', align_corners=False)
        # 41 deterministic channels; noisy residual is prepended by the backbone.
        maps = batch['maps']
        condition_maps = torch.cat([batch['mu_condition'], batch['baseline_condition'], maps, features, domain], 1)
        query_geometry = maps[:, [1, 2, 0]]
        return dict(maps=condition_maps, domain=domain,
                    attn12=self.attn12.prepare(tokens, positions, padding, F.interpolate(query_geometry, size=(12,12), mode='bilinear', align_corners=False)),
                    attn6=self.attn6.prepare(tokens, positions, padding, F.interpolate(query_geometry, size=(6,6), mode='bilinear', align_corners=False)))

    def _backbone(self, residual, noise_condition, condition):
        b = residual.shape[0]
        phase = noise_condition.reshape(b, 1)*self.noise_frequencies[None]
        embedding = self.noise_encoder(torch.cat([phase.cos(), phase.sin()], 1))
        hidden = self.stem(torch.cat([residual, condition['maps']], 1))
        for block in self.enc24:
            hidden = block(hidden, embedding)
        skip24 = hidden
        hidden = self.down12(F.avg_pool2d(hidden, 2))
        for block in self.enc12:
            hidden = block(hidden, embedding)
        hidden = self.attn12(hidden, condition['attn12'])
        skip12 = hidden
        hidden = self.down6(F.avg_pool2d(hidden, 2))
        for block in self.enc6:
            hidden = block(hidden, embedding)
        hidden = self.attn6(hidden, condition['attn6'])
        hidden = self.up12(F.interpolate(hidden, size=(12,12), mode='bilinear', align_corners=False))
        hidden = torch.cat([hidden, skip12], 1)
        for block in self.dec12:
            hidden = block(hidden, embedding)
        hidden = self.up24(F.interpolate(hidden, size=(24,24), mode='bilinear', align_corners=False))
        hidden = torch.cat([hidden, skip24], 1)
        for block in self.dec24:
            hidden = block(hidden, embedding)
        return self.output(hidden)*condition['domain']

    def _prediction(self, condition):
        zero = torch.zeros_like(condition['domain'])
        prediction = self._backbone(zero, zero.new_zeros(len(zero)), condition)
        if self.arm == 'gaussian':
            mean, log_std = prediction.chunk(2, 1)
            return mean, log_std.clamp(-5, 3)
        return prediction

    def forward(self, batch):
        if self.arm == 'edm':
            raise ValueError('EDM requires denoise or sample, not an unspecified noise state')
        return self._prediction(self.encode_condition(batch))

    def denoise(self, r_sigma, sigma, condition):
        sigma = torch.as_tensor(sigma, device=r_sigma.device, dtype=r_sigma.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(len(r_sigma))
        if sigma.numel() != len(r_sigma):
            raise ValueError('sigma must be scalar or have one value per sample')
        skip, out, inp, noise, _ = edm_coefficients(sigma, self.sigma_data)
        noisy = r_sigma*condition['domain']
        prediction = self._backbone(inp[:,None,None,None]*noisy, noise, condition)
        return (skip[:,None,None,None]*noisy+out[:,None,None,None]*prediction)*condition['domain']

    def loss(self, batch, p_mean=-1.2, p_std=1.2):
        condition = self.encode_condition(batch)
        target = batch['residual']
        domain = condition['domain'].bool()
        if not domain.any():
            raise ValueError('Cannot train an empty domain')
        # Replace excluded target values before arithmetic, including stored NaNs.
        target = target.masked_fill(~domain, 0)
        if self.arm == 'edm':
            sigma = (torch.randn(len(target), device=target.device)*p_std+p_mean).exp()
            noisy = target+torch.randn_like(target)*sigma[:,None,None,None]*domain
            clean = self.denoise(noisy, sigma, condition)
            weight = edm_coefficients(sigma, self.sigma_data)[-1]
            losses = (clean-target).square()*weight[:,None,None,None]
        elif self.arm == 'gaussian':
            mean, log_std = self._prediction(condition)
            losses = .5*(target-mean).square()*torch.exp(-2*log_std)+log_std+.5*math.log(2*math.pi)
        else:
            losses = (self._prediction(condition)-target).square()
        return losses.masked_select(domain).mean()

    @torch.no_grad()
    def sample(self, condition, hour_ids, members, seed, steps=18):
        if len(hour_ids) != condition['maps'].shape[0]:
            raise ValueError('hour_ids must match the condition batch')
        domain = condition['domain']
        if self.arm == 'det':
            return self._prediction(condition)[:,None].expand(-1, members, -1, -1, -1).clone()
        noise = keyed_noise(hour_ids, members, seed, device=domain.device, dtype=domain.dtype)
        if self.arm == 'gaussian':
            mean, log_std = self._prediction(condition)
            return (mean[:,None]+log_std.exp()[:,None]*noise)*domain[:,None]
        outputs = []
        for member in range(members):
            outputs.append(heun_sample(lambda x, sigma: self.denoise(x, sigma, condition),
                                       noise[:,member], domain, steps=steps))
        return torch.stack(outputs, 1)
