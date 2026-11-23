"""Conditional EDM on coefficient vectors with differentiable Heun sampling."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def edm_coefficients(sigma, sigma_data=.5):
    """Return c_skip, c_out, c_in, c_noise and the denoising-loss weight."""
    sigma = torch.as_tensor(sigma)
    if sigma_data <= 0 or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('EDM requires positive finite sigma and sigma_data')
    denominator = sigma.square() + sigma_data**2
    return (sigma_data**2 / denominator,
            sigma * sigma_data / denominator.sqrt(), denominator.rsqrt(),
            sigma.log() / 4, denominator / (sigma * sigma_data).square())


def keyed_noise(hour_ids, members, seed, shape=(128,), device='cpu', dtype=torch.float32):
    """CPU keyed streams preserve exact prefixes and batch-independent draws."""
    hours = torch.as_tensor(hour_ids).detach().cpu().reshape(-1).tolist()
    if members < 1 or not hours:
        raise ValueError('At least one hour and ensemble member is required')
    rows = []
    for hour in hours:
        ensemble = []
        for member in range(members):
            identity = f'{int(seed)}:{int(hour)}:{member}'.encode('ascii')
            value = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') & ((1 << 63) - 1)
            generator = torch.Generator(device='cpu').manual_seed(value)
            ensemble.append(torch.randn(shape, generator=generator, dtype=torch.float32))
        rows.append(torch.stack(ensemble))
    return torch.stack(rows).to(device=device, dtype=dtype)


def heun_sample(denoise, initial_noise, domain=None, steps=18,
                sigma_min=.002, sigma_max=80., rho=7.):
    """Solve the EDM ODE on tensor states; retain gradients through 2*steps-1 calls."""
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError('Invalid EDM noise schedule')
    ramp = torch.linspace(0, 1, steps, device=initial_noise.device, dtype=initial_noise.dtype)
    schedule = (sigma_max**(1 / rho) + ramp *
                (sigma_min**(1 / rho) - sigma_max**(1 / rho)))**rho
    schedule = torch.cat([schedule, schedule.new_zeros(1)])
    mask = 1 if domain is None else domain.to(device=initial_noise.device, dtype=initial_noise.dtype)
    current = initial_noise * schedule[0] * mask
    for index in range(steps):
        sigma, following = schedule[index], schedule[index + 1]
        clean = denoise(current, sigma) * mask
        derivative = (current - clean) / sigma
        predicted = (current + (following - sigma) * derivative) * mask
        if index < steps - 1:
            corrected = denoise(predicted, following) * mask
            next_derivative = (predicted - corrected) / following
            predicted = (current + (following - sigma) * (derivative + next_derivative) / 2) * mask
        current = predicted
    return current


class CoefficientBlock(nn.Module):
    """Fully connected residual block; coefficient indices carry no spatial ordering."""
    def __init__(self, width, embedding_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.film = nn.Linear(embedding_dim, width * 2)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x, embedding):
        hidden = self.linear1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, -1)
        hidden = self.norm2(hidden) * (1 + scale) + shift
        return (x + self.linear2(F.silu(hidden))) / math.sqrt(2)


class EDMHead(nn.Module):
    def __init__(self, rank, condition_dim=128, width=256, sigma_data=.5):
        super().__init__()
        self.rank = rank
        self.sigma_data = sigma_data
        self.register_buffer('noise_frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 31))
        self.input = nn.Linear(rank, width)
        self.blocks = nn.ModuleList([CoefficientBlock(width, condition_dim + 64) for _ in range(4)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, rank)

    def denoise(self, x, sigma, h):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(len(x))
        if sigma.numel() != len(x):
            raise ValueError('sigma must be scalar or have one value per sample')
        skip, out, inp, noise, _ = edm_coefficients(sigma, self.sigma_data)
        phase = noise[:, None] * self.noise_frequencies[None]
        embedding = torch.cat([h, phase.cos(), phase.sin()], -1)
        hidden = self.input(x * inp[:, None])
        for block in self.blocks:
            hidden = block(hidden, embedding)
        prediction = self.output(F.silu(self.output_norm(hidden)))
        return skip[:, None] * x + out[:, None] * prediction

    forward = denoise

    def sample(self, h, hour_ids, members, seed, steps=18,
               sigma_min=.002, sigma_max=80., rho=7.):
        hours = torch.as_tensor(hour_ids).reshape(-1)
        if len(hours) != len(h):
            raise ValueError('hour_ids must match the condition batch')
        noise = keyed_noise(hours, members, seed, shape=(self.rank,), device=h.device, dtype=h.dtype)
        expanded = h[:, None].expand(-1, members, -1).reshape(-1, h.shape[-1])
        samples = heun_sample(lambda x, sigma: self.denoise(x, sigma, expanded),
                              noise.flatten(0, 1), steps=steps, sigma_min=sigma_min,
                              sigma_max=sigma_max, rho=rho)
        return samples.reshape(len(h), members, self.rank)
