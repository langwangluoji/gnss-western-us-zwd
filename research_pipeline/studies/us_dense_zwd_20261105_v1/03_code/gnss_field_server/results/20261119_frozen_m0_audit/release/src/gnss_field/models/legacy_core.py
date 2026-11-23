"""Pure neural definitions extracted for checkpoint parity. No historical imports.

Sources: Europe model/network.py; US us_gan.py, cvae_pilot.py, station_mae.py,
mae_gan_cvae.py, moment_superresolution.py, regional_temporal_mae.py,
architecture_search.py and residual_edm_v1/model.py. Constructors and operations
retain original ordering so historical state dictionaries load without remapping.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):

    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())

    def forward(self, x):
        return self.net(x)

class ResidualUNet(nn.Module):

    def __init__(self, channels, width=32):
        super().__init__()
        if width % 4:
            raise ValueError('width must be divisible by 4')
        self.enc = Block(4 * channels, width)
        self.down = Block(width, width * 2)
        self.bottleneck = Block(width * 2, width * 2)
        self.dec = Block(width * 3, width)
        self.coarse = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.coarse.weight)
        nn.init.zeros_(self.coarse.bias)
        self.fine = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    def forward(self, conditions, baseline_mm, reference_mm=None):
        if conditions.ndim != 5 or conditions.shape[1] != 4 or conditions.shape[-2:] != (32, 32):
            raise ValueError('expected [B,4,C,32,32]')
        if baseline_mm.shape != (conditions.shape[0], 1, 32, 32):
            raise ValueError('baseline shape')
        x = conditions.flatten(1, 2)
        e = self.enc(x)
        d = self.down(F.avg_pool2d(e, 2))
        d = self.bottleneck(d)
        h = self.dec(torch.cat((e, F.interpolate(d, size=e.shape[-2:], mode='bilinear', align_corners=False)), 1))
        coarse = self.coarse(h)
        fine = self.fine(h)
        fine = fine - F.avg_pool2d(fine, 4, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        bg = F.interpolate(baseline_mm, scale_factor=4, mode='bilinear', align_corners=False)
        reference_mm = torch.zeros_like(baseline_mm) if reference_mm is None else reference_mm
        reference = F.interpolate(reference_mm, scale_factor=4, mode='bilinear', align_corners=False)
        field = bg + reference + F.interpolate(coarse, scale_factor=4, mode='nearest') + fine
        return {'field_mm': field, 'baseline_mm': bg, 'coarse_mm': coarse, 'fine_mm': fine, 'reference_mm': reference}

class USResidualUNet(ResidualUNet):

    def forward(self, conditions, baseline_mm, reference_mm=None):
        if conditions.ndim != 5 or conditions.shape[1] != 4 or conditions.shape[-2:] != (6, 6):
            raise ValueError('expected [B,4,C,6,6]')
        if baseline_mm.shape != (conditions.shape[0], 1, 6, 6):
            raise ValueError('baseline shape')
        e = self.enc(conditions.flatten(1, 2))
        d = self.bottleneck(self.down(F.avg_pool2d(e, 2)))
        hidden = self.dec(torch.cat((e, F.interpolate(d, size=e.shape[-2:], mode='bilinear', align_corners=False)), 1))
        coarse = self.coarse(hidden)
        fine = self.fine(hidden)
        fine = fine - F.avg_pool2d(fine, 4, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        baseline = F.interpolate(baseline_mm, scale_factor=4, mode='bilinear', align_corners=False)
        field = baseline + F.interpolate(coarse, scale_factor=4, mode='nearest') + fine
        return dict(field_mm=field, baseline_mm=baseline, coarse_mm=coarse, fine_mm=fine)

def gaussian_kl(mean_q, logvar_q, mean_p, logvar_p):
    return 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mean_q - mean_p).square()) / logvar_p.exp() - 1)

class ConditionalVAE(USResidualUNet):

    def __init__(self):
        super().__init__(15, 16)
        self.prior = nn.Conv2d(32, 16, 1)
        self.target_encoder = nn.Sequential(nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU())
        self.posterior = nn.Conv2d(48, 16, 1)
        self.latent_projection = nn.Conv2d(8, 32, 1)

    def forward(self, conditions, baseline_mm, target=None, mask=None, sample=False, prior_only=False, encoder_addition=None, use_latent=True, residual_latent=False, fine_addition=None, film=None):
        if conditions.shape[1:] != (4, 15, 6, 6):
            raise ValueError('Expected four15-channel6x6 history frames')
        if not self.training and (target is not None or mask is not None):
            raise ValueError('Evaluation uses the conditional prior and cannot receive targets')
        encoded = self.enc(conditions.flatten(1, 2))
        if encoder_addition is not None:
            if encoder_addition.shape != encoded.shape:
                raise ValueError('Encoder addition must match the6x6 encoded feature map')
            encoded = encoded + encoder_addition
        conditional = self.bottleneck(self.down(F.avg_pool2d(encoded, 2)))
        mean_p, logvar_p = self.prior(conditional).chunk(2, dim=1)
        logvar_p = logvar_p.clamp(-8, 8)
        baseline = F.interpolate(baseline_mm, scale_factor=4, mode='bilinear', align_corners=False)
        if self.training and (not prior_only) and use_latent:
            if target is None or mask is None:
                raise ValueError('Training posterior requires target and mask')
            residual = torch.where(mask, target - baseline, 0)
            target_features = self.target_encoder(torch.cat([residual, mask.float()], dim=1))
            mean_q, logvar_q = self.posterior(torch.cat([conditional, target_features], dim=1)).chunk(2, dim=1)
            logvar_q = logvar_q.clamp(-8, 8)
            latent = mean_q + torch.exp(0.5 * logvar_q) * torch.randn_like(mean_q)
            kl = gaussian_kl(mean_q, logvar_q, mean_p, logvar_p).mean()
        else:
            latent = mean_p
            if sample:
                latent = latent + torch.exp(0.5 * logvar_p) * torch.randn_like(latent)
            kl = conditional.new_zeros(())
        if residual_latent:
            result = self.decode_features(encoded, conditional, baseline, fine_addition, film)
            if use_latent and (self.training and (not prior_only) or sample):
                stochastic = self.decode_features(encoded, conditional + self.latent_projection(latent), baseline, fine_addition, film)
                centered = self.decode_features(encoded, conditional + self.latent_projection(mean_p), baseline, fine_addition, film)
                for key in ['field_mm', 'coarse_mm', 'fine_mm']:
                    result[key] = result[key] + stochastic[key] - centered[key]
        else:
            decoded = conditional + self.latent_projection(latent) if use_latent else conditional
            result = self.decode_features(encoded, decoded, baseline, fine_addition, film)
        result.update(kl=kl, prior_mean=mean_p, prior_logvar=logvar_p)
        return result

    def decode_features(self, encoded, decoded, baseline, fine_addition=None, film=None):
        hidden = self.dec(torch.cat([encoded, F.interpolate(decoded, size=(6, 6), mode='bilinear', align_corners=False)], dim=1))
        if film is not None:
            hidden = hidden * (1 + film[:, 0, :, None, None]) + film[:, 1, :, None, None]
        coarse = self.coarse(hidden)
        if film is None:
            fine = self.fine(hidden)
        else:
            fine = self.fine[:3](hidden)
            fine = fine * (1 + film[:, 2, :, None, None]) + film[:, 3, :, None, None]
            fine = self.fine[3:](fine)
        if fine_addition is not None:
            fine = fine + fine_addition
        fine = fine - F.avg_pool2d(fine, 4, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        field = baseline + F.interpolate(coarse, scale_factor=4, mode='nearest') + fine
        return dict(field_mm=field, baseline_mm=baseline, coarse_mm=coarse, fine_mm=fine)

class StationMAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(13, 48)
        layer = nn.TransformerEncoderLayer(48, 4, 96, dropout=0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.query_embedding = nn.Linear(7, 48)
        self.cross_attention = nn.MultiheadAttention(48, 4, dropout=0, batch_first=True)
        self.query_norm = nn.LayerNorm(48)
        self.query_mlp = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Linear(96, 48))
        self.readout = nn.Linear(48, 5)

    def representations(self, tokens, padding, queries):
        if padding.all(dim=1).any():
            raise ValueError('A station window must contain at least one observed token')
        clean = tokens.masked_fill(padding[:, :, None], 0)
        encoded = self.encoder(self.embedding(clean), src_key_padding_mask=padding)
        query = self.query_embedding(queries)
        context, _ = self.cross_attention(query, encoded, encoded, key_padding_mask=padding, need_weights=False)
        context = self.query_norm(context + query)
        return context + self.query_mlp(context)

    def forward(self, tokens, padding, queries):
        return self.readout(self.representations(tokens, padding, queries))

class FusionGenerator(nn.Module):

    def __init__(self, use_latent=True, residual_latent=False):
        super().__init__()
        self.core = ConditionalVAE()
        self.mae = StationMAE()
        self.projection = nn.Conv2d(48, 16, 1)
        self.use_latent = use_latent
        self.residual_latent = residual_latent

    def token_features(self, tokens, padding, queries):
        context = self.mae.representations(tokens, padding, queries)
        return self.projection(context.transpose(1, 2).reshape(-1, 48, 6, 6))

    def forward(self, conditions, baseline, tokens=None, padding=None, queries=None, addition=None, target=None, mask=None, prior_only=True, sample=False):
        if addition is None:
            addition = self.token_features(tokens, padding, queries)
        return self.core(conditions, baseline, target=target, mask=mask, prior_only=prior_only, sample=sample, encoder_addition=addition, use_latent=self.use_latent, residual_latent=self.residual_latent)

class SpatialAdapter(nn.Module):

    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 48))
        self.position = nn.Linear(3, 48)
        self.query = nn.Linear(7, 48)
        self.attention = nn.MultiheadAttention(48, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(48)

    def representations(self, tokens, padding, queries):
        tokens = tokens.masked_fill(padding[:, :, None], 0)
        encoded = self.embedding(tokens[:, :, :self.feature_dim]) + self.position(tokens[:, :, self.feature_dim:])
        query = self.query(queries)
        result, _ = self.attention(query, encoded, encoded, key_padding_mask=padding, need_weights=False)
        return self.norm(query + result)

class RegionalGenerator(FusionGenerator):

    def __init__(self, use_latent=True, residual_latent=False):
        super().__init__(use_latent, residual_latent)
        self.mae = SpatialAdapter(feature_dim=64)

class SearchGenerator(RegionalGenerator):

    def __init__(self, variant='base', use_latent=False, fine_height=None):
        super().__init__(use_latent)
        self.variant = variant
        if 'film' in variant:
            self.film_head = nn.Linear(64, 64)
            nn.init.zeros_(self.film_head.weight)
            nn.init.zeros_(self.film_head.bias)
        if 'query' in variant:
            self.query_head = nn.Linear(48, 1)
            nn.init.zeros_(self.query_head.weight)
            nn.init.zeros_(self.query_head.bias)
            self.query_gate = nn.Parameter(torch.tensor(0.05))
            axis = torch.arange(24, dtype=torch.float32) * 25 - 287.5
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            height = torch.zeros(576) if fine_height is None else fine_height.reshape(-1).float() / 3000
            self.register_buffer('fine_positions', torch.stack([xx.reshape(-1) / 400, yy.reshape(-1) / 400, height], dim=1), persistent=False)

    def forward(self, conditions, baseline, tokens, padding, queries, target=None, mask=None, prior_only=True, sample=False):
        if self.variant == 'no_history':
            return self.core(conditions, baseline, target=target, mask=mask, prior_only=prior_only, sample=sample, use_latent=self.use_latent)
        tokens = tokens.masked_fill(padding[:, :, None], 0)
        addition = self.token_features(tokens, padding, queries)
        film = None
        fine_addition = None
        if 'film' in self.variant:
            valid = (~padding).float()
            pooled = (tokens[:, :, :64] * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
            film = 0.1 * torch.tanh(self.film_head(pooled).reshape(-1, 4, 16))
        if 'query' in self.variant:
            q = torch.cat([self.fine_positions[None].expand(len(tokens), -1, -1), queries[:, 0:1, 3:].expand(-1, 576, -1)], dim=-1)
            encoded = self.mae.embedding(tokens[:, :, :64]) + self.mae.position(tokens[:, :, 64:])
            query = self.mae.query(q)
            distance = torch.cdist(q[:, :, :2], tokens[:, :, 64:66]) / 0.25
            attention_mask = (-distance).masked_fill(padding[:, None, :], float('-inf')).repeat_interleave(4, dim=0)
            context, _ = self.mae.attention(query, encoded, encoded, attn_mask=attention_mask, need_weights=False)
            context = self.mae.norm(query + context)
            fine_addition = torch.tanh(self.query_gate) * self.query_head(context).transpose(1, 2).reshape(-1, 1, 24, 24)
        return self.core(conditions, baseline, target=target, mask=mask, prior_only=prior_only, sample=sample, encoder_addition=addition, use_latent=self.use_latent, fine_addition=fine_addition, film=film)

class ResidualBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels_in)
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.film = nn.Linear(128, channels_out * 2)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, value, embedding):
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[..., None, None]) + shift[..., None, None]
        return (self.skip(value) + self.conv2(F.silu(hidden))) / math.sqrt(2)

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
        relative = qpos[:, :, None, :] - positions[:, None, :, :]
        distance = relative[..., :2].square().sum(-1, keepdim=True).sqrt()
        bias = self.geometry(torch.cat([relative, distance], -1)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(padding[:, None, None, :], float('-inf'))
        return dict(key=key, value=value, bias=bias, query_position=self.query_position(qpos))

    def forward(self, hidden, prepared):
        b, _, h, w = hidden.shape
        query = self.query(hidden).flatten(2).transpose(1, 2) + prepared['query_position']
        query = query.reshape(b, h * w, 4, 16).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, prepared['key'], prepared['value'], attn_mask=prepared['bias'], dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(b, h * w, 64).transpose(1, 2).reshape(b, 64, h, w)
        return (hidden + self.output(attended)) / math.sqrt(2)

class ResidualModel(nn.Module):
    """Matched deterministic, independent-Gaussian and EDM residual branches."""
    sigma_data = 0.5

    def __init__(self, arm='edm'):
        super().__init__()
        if arm not in ('det', 'gaussian', 'edm'):
            raise ValueError(f'Unknown arm: {arm}')
        self.arm = arm
        self.station_encoder = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 64))
        self.amplitude = nn.Linear(5, 64, bias=False)
        self.null_station = nn.Parameter(torch.zeros(1, 1, 64))
        self.coarse_encoder = nn.Sequential(nn.Conv2d(60, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.noise_encoder = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.register_buffer('noise_frequencies', torch.exp(-math.log(10000) * torch.arange(32) / 31))
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
        self.output = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 2 if arm == 'gaussian' else 1, 3, padding=1))

    def _backbone(self, residual, noise_condition, condition):
        b = residual.shape[0]
        phase = noise_condition.reshape(b, 1) * self.noise_frequencies[None]
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
        hidden = self.up12(F.interpolate(hidden, size=(12, 12), mode='bilinear', align_corners=False))
        hidden = torch.cat([hidden, skip12], 1)
        for block in self.dec12:
            hidden = block(hidden, embedding)
        hidden = self.up24(F.interpolate(hidden, size=(24, 24), mode='bilinear', align_corners=False))
        hidden = torch.cat([hidden, skip24], 1)
        for block in self.dec24:
            hidden = block(hidden, embedding)
        return self.output(hidden) * condition['domain']
