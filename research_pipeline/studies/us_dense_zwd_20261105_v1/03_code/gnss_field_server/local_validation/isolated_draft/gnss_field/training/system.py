"""The same frozen spatial decoder serves training, publication and evaluation."""
from dataclasses import replace
import torch
from torch import nn

from gnss_field.contracts import FieldProduct
from gnss_field.geometry import basis_observation_matrix, surface_query
from gnss_field.models.condition import ConditionEncoder
from gnss_field.models.heads import DeterministicHead, GaussianHead, GANGenerator, ConditionalCritic
from gnss_field.models.diffusion import EDMHead, keyed_noise
from gnss_field.objectives import CoefficientTransform


class FrozenDecoder:
    def __init__(self, phi, shape, transform, scale):
        self.phi, self.shape, self.transform, self.scale = phi, tuple(shape), transform, float(scale)

    def field(self, normalized, mean):
        c = self.transform.decode(normalized)
        r = torch.einsum('gr,bkr->bkg', self.phi.to(c), c).reshape(*c.shape[:2], 1, *self.shape)
        return mean[:, None] + self.scale*r

    def matrix(self, xy, heights):
        return basis_observation_matrix(self.phi, self.shape, xy, heights)

    def station(self, normalized, mean_at_stations, xy=None, heights=None, *, matrix=None):
        h = matrix if matrix is not None else self.matrix(xy, heights)
        c = self.transform.decode(normalized)
        return mean_at_stations[:, None] + self.scale*torch.einsum('nr,bkr->bkn', h.to(c), c)


class FunctionalModel(nn.Module):
    def __init__(self, arm, rank=128, condition_dim=128, *, frozen_encoder=False):
        super().__init__()
        self.arm, self.rank = arm, rank
        self.encoder = ConditionEncoder(condition_dim)
        if arm == 'det':
            self.head = DeterministicHead(rank, condition_dim)
        elif arm == 'gaussian':
            self.head = GaussianHead(rank, condition_dim)
        elif arm == 'edm':
            self.head = EDMHead(rank, condition_dim)
        elif arm == 'gan':
            self.head = GANGenerator(rank, condition_dim)
            self.critic = ConditionalCritic(rank, condition_dim)
        else:
            raise ValueError(f'Unknown functional arm {arm}')
        self.encoder.requires_grad_(not frozen_encoder)
        self.supports_joint = arm in ('gaussian', 'edm', 'gan')

    def condition(self, context, scale):
        scaled = replace(context, mean=context.mean/scale if context.mean is not None else None,
                         baseline=context.baseline/scale)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(scaled)
        return self.encoder(scaled)

    def coefficients(self, h, hour_ids, members=64, seed=20261117, *, science=None):
        if self.arm == 'det':
            return self.head(h)[:, None]
        if self.arm == 'edm':
            kwargs = {} if science is None else dict(steps=science.sampler_steps, sigma_min=science.sigma_min,
                    sigma_max=science.sigma_max, rho=science.rho)
            return self.head.sample(h, hour_ids, members, seed, **kwargs)
        noise = keyed_noise(hour_ids, members, seed, (self.rank,), device=h.device, dtype=h.dtype)
        if self.arm == 'gaussian':
            mean, log_std = self.head(h)
            return mean[:, None]+noise*log_std.exp()[:, None]
        conditions = h[:, None].expand(-1, members, -1).flatten(0, 1)
        return self.head(noise.flatten(0, 1), conditions).reshape(len(h), members, self.rank)

    def predict_field(self, context, decoder, *, members=64, seed=20261117, science=None):
        if context.mean is None:
            raise ValueError('A frozen published M0 field is required')
        h = self.condition(context, decoder.scale)
        c = self.coefficients(h, context.hours, members, seed, science=science)
        fields = decoder.field(c, context.mean)
        return FieldProduct(mean=fields.mean(1), samples=fields if self.supports_joint else None,
                            support=context.maps[:, 3].bool(), grid_id=context.grid_id)
