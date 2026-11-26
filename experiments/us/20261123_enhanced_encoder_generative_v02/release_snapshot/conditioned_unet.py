"""Independent conditional field GAN, VAE and observation-score diffusion.

No class imports the legacy VAE/latent pipeline. Noise is keyed by hour and
member. Diffusion has a complete spatial ODE sampler; its local training
objective is observational proper scoring, not dense supervised DSM.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .diffusion import keyed_noise, heun_sample, edm_coefficients


def finite(value):
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite field, condition or scored observation')


class DoubleConv(nn.Sequential):
    def __init__(self, source, destination):
        super().__init__(nn.Conv2d(source, destination, 3, padding=1), nn.GroupNorm(4, destination), nn.SiLU(),
                         nn.Conv2d(destination, destination, 3, padding=1), nn.GroupNorm(4, destination), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, channels=42, widths=(32, 64, 128)):
        super().__init__()
        self.widths = tuple(widths)
        self.first = DoubleConv(channels, widths[0])
        self.second = DoubleConv(widths[0], widths[1])
        self.bottom = DoubleConv(widths[1], widths[2])

    def forward(self, x):
        finite(x)
        if x.ndim != 4 or x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError('Conditions must be BCHW with H/W divisible by4')
        a = self.first(x)
        b = self.second(F.avg_pool2d(a, 2))
        c = self.bottom(F.avg_pool2d(b, 2))
        return a, b, c


class Decoder(nn.Module):
    def __init__(self, widths=(32, 64, 128), zero_head=True):
        super().__init__()
        self.up2 = DoubleConv(widths[2]+widths[1], 32)
        self.up1 = DoubleConv(32+widths[0], 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.bias)
        if zero_head:
            nn.init.zeros_(self.head.weight)
        else:
            nn.init.normal_(self.head.weight, std=.001)

    def forward(self, encoded):
        a, b, c = encoded
        d = self.up2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        return self.head(self.up1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1)))


class ConditionedUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128)):
        super().__init__()
        self.encoder = Encoder(widths=widths)
        self.decoder = Decoder(widths=widths)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def expand_members(tensor, members):
    return tensor[:, None].expand(-1, members, *tensor.shape[1:]).flatten(0, 1)


class _SpatialGenerator(nn.Module):
    def __init__(self, widths=(32, 64, 128)):
        super().__init__()
        self.core = ConditionedUNet(widths)
        self.latent_projection = nn.Conv2d(4, widths[2], 1)
        nn.init.normal_(self.latent_projection.weight, std=.02)
        nn.init.zeros_(self.latent_projection.bias)

    def encode(self, x):
        return self.core.encoder(x)

    def decode(self, encoded, base, latent):
        batch, members = latent.shape[:2]
        a, b, c = [expand_members(v, members) for v in encoded]
        c = c+self.latent_projection(latent.flatten(0, 1))
        result = expand_members(base, members)+10*self.core.decoder((a, b, c))
        finite(result)
        return result.reshape(batch, members, 1, *base.shape[-2:])

    def sample_encoded(self, encoded, base, hours, members, seed):
        noise = keyed_noise(hours, members, seed, shape=(4, *encoded[-1].shape[-2:]), device=base.device)
        return self.decode(encoded, base, noise)

    def sample(self, x, base, hours, members, seed):
        return self.sample_encoded(self.encode(x), base, hours, members, seed)


class FieldGAN(_SpatialGenerator):
    pass


class FieldVAE(_SpatialGenerator):
    def __init__(self, widths=(32, 64, 128)):
        super().__init__(widths)
        self.prior = nn.Conv2d(widths[2], 8, 1)
        self.posterior = nn.Sequential(nn.Conv2d(widths[2]+2, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 8, 1))
        for layer in (self.prior, self.posterior[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def parameters_from(raw):
        mu, logstd = raw.chunk(2, 1)
        finite(raw)
        return mu, logstd.clamp(-4, 1)

    def sample_encoded(self, encoded, base, hours, members, seed):
        mu, logstd = self.parameters_from(self.prior(encoded[-1]))
        white = keyed_noise(hours, members, seed, shape=tuple(mu.shape[1:]), device=base.device)
        return self.decode(encoded, base, mu[:, None]+logstd[:, None].exp()*white)

    def posterior_sample(self, encoded, base, posterior_grid, hours, members, seed):
        if posterior_grid.shape != (len(base), 2, *base.shape[-2:]):
            raise ValueError('Posterior receives only explicit two-channel training-B scatter')
        context = F.avg_pool2d(posterior_grid, 4)
        mu_q, log_q = self.parameters_from(self.posterior(torch.cat([encoded[-1], context], 1)))
        mu_p, log_p = self.parameters_from(self.prior(encoded[-1]))
        white = keyed_noise(hours, members, seed, shape=tuple(mu_q.shape[1:]), device=base.device)
        return self.decode(encoded, base, mu_q[:, None]+log_q[:, None].exp()*white), (mu_q, log_q, mu_p, log_p)


class FieldDiffusion(nn.Module):
    def __init__(self, widths=(32, 64, 128), steps=6):
        super().__init__()
        self.core = ConditionedUNet(widths)
        self.noise_encoder = Encoder(1, (16, 32, 64))
        self.condition_projection = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in zip(widths, (16, 32, 64))])
        self.noise_decoder = Decoder((16, 32, 64), zero_head=False)
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 16+32+64))
        self.steps = steps
        self.last_calls = 0

    def encode(self, x):
        return self.core.encoder(x)

    def sample_encoded(self, encoded, base, hours, members, seed):
        batch = len(base)
        mean = self.core.decoder(encoded)
        projected = [expand_members(p(c), members) for p, c in zip(self.condition_projection, encoded)]
        noise = keyed_noise(hours, members, seed, shape=(1, *base.shape[-2:]), device=base.device)
        initial = noise.flatten(0, 1)
        self.last_calls = 0

        def denoise(state, sigma):
            self.last_calls += 1
            skip, out, inp, time, _ = edm_coefficients(sigma, sigma_data=.5)
            features = self.noise_encoder(state*inp)
            embedding = self.time(time.reshape(1, 1)).split((16, 32, 64), -1)
            mixed = tuple(n+c+t[..., None, None] for n, c, t in zip(features, projected, embedding))
            return skip*state+out*self.noise_decoder(mixed)

        residual = heun_sample(denoise, initial, steps=self.steps, sigma_min=.02, sigma_max=2.)
        result = expand_members(base+10*mean, members)+10*residual
        finite(result)
        return result.reshape(batch, members, 1, *base.shape[-2:])

    def sample(self, x, base, hours, members, seed):
        return self.sample_encoded(self.encode(x), base, hours, members, seed)


class StationCritic(nn.Module):
    """Conditional station-set critic with explicit pairwise interaction."""
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 32))
        self.attention = nn.MultiheadAttention(32, 2, batch_first=True)
        self.output = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, values, condition, geometry, mask):
        batch, members, stations = values.shape
        if not mask.any(-1).all():
            raise ValueError('Critic rejects empty station sets')
        safe = torch.where(mask[:, None], values, 0.)
        features = torch.cat([safe[..., None], expand_members(condition, members).reshape(batch, members, stations, 4),
                              geometry[None, None].expand(batch, members, -1, -1)], -1)
        hidden = self.point(features.flatten(0, 1))
        valid = mask[:, None].expand(-1, members, -1).flatten(0, 1)
        mixed, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~valid, need_weights=False)
        pooled = ((hidden+mixed)*valid[..., None]).sum(1)/valid.sum(1, keepdim=True)
        result = self.output(pooled).reshape(batch, members)
        finite(result)
        return result


def fair_scores(samples, truth, mask):
    if samples.ndim != 3 or samples.shape[1] < 2 or samples.shape[::2] != truth.shape:
        raise ValueError('Proper scores require BKN with K>=2 and BN targets')
    finite(samples)
    finite(truth[mask])
    count = mask.sum(-1)
    if (count == 0).any():
        raise ValueError('Scoring population contains empty hours')
    clean = torch.where(mask, truth, 0.)
    x = samples*mask[:, None]
    first = torch.linalg.vector_norm(x-clean[:, None], dim=-1).mean(1)
    k = samples.shape[1]
    distances = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    es = ((first-distances.sum((1, 2))/(2*k*(k-1)))/count.sqrt()).mean()
    sorted_x = samples.sort(1).values
    coefficient = (2*torch.arange(1, k+1, device=samples.device)-k-1).to(samples.dtype)
    crps = (abs(samples-clean[:, None]).mean(1)-(sorted_x*coefficient[None, :, None]).sum(1)/(k*(k-1)))
    crps = (crps*mask).sum()/mask.sum()
    return es, crps
