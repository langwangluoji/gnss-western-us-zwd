"""EDM on a fixed observation-sensitive subspace, with frozen Gaussian remainder."""
import torch
from torch import nn
from torch.nn import functional as F
from .diffusion import edm_coefficients, heun_sample


class SubspaceEDM(nn.Module):
    def __init__(self, rank=64, condition_dim=32, width=128):
        super().__init__()
        self.rank, self.condition_dim = rank, condition_dim
        self.register_buffer('frequencies', torch.exp(-torch.arange(16, dtype=torch.float32)*.5))
        self.net = nn.Sequential(nn.Linear(rank+condition_dim+32, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, rank))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # One schedule-only correction. It is fixed before training and C scoring.
        with torch.no_grad():
            one = torch.ones(1, rank, dtype=torch.float64)
            gain = heun_sample(lambda x, s: x/(1+s*s), one, steps=18)[0, 0]
        self.register_buffer('gaussian_gain', gain)

    def denoise(self, x, sigma, condition):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).reshape(-1)
        if len(sigma) == 1:
            sigma = sigma.expand(len(x))
        skip, out, inp, noise, _ = edm_coefficients(sigma, sigma_data=1.)
        phase = noise[:, None]*self.frequencies[None]
        features = torch.cat([x*inp[:, None], condition, phase.cos(), phase.sin()], -1)
        return skip[:, None]*x+out[:, None]*self.net(features)

    def sample_noise(self, condition, initial_noise):
        b, k, r = initial_noise.shape
        if r != self.rank or condition.shape != (b, self.condition_dim):
            raise ValueError('Subspace condition/noise dimensions differ')
        expanded = condition[:, None].expand(-1, k, -1).reshape(b*k, -1)
        output = heun_sample(lambda x, s: self.denoise(x, s, expanded), initial_noise.reshape(b*k, r), steps=18)
        return output.reshape(b, k, r)/self.gaussian_gain.to(dtype=output.dtype)


def station_members(mean, latent, coefficient_noise, station_noise, projection, remainder, scale, tau):
    """Fixed matrices have surface-mm units; only latent receives gradients."""
    change = latent@projection.T+coefficient_noise@remainder.T+float(tau)*station_noise
    return mean[:, None]+scale[:, None, None]*change
