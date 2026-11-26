"""Physical surface queries and explicit sparse-observation Gaussian ELBO."""
import math

import torch


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def _query_contract(features, indices, weights):
    if features.ndim != 4 or indices.ndim != 2 or indices.shape[-1] != 4 or weights.shape != indices.shape:
        raise ValueError('Expected BCHW features and N-by-4 query indices/weights')
    if indices.dtype != torch.long:
        raise ValueError('Query indices must be int64')
    if indices.numel() and ((indices < 0).any() or (indices >= features.shape[-2] * features.shape[-1]).any()):
        raise ValueError('Query indices outside published grid')
    _finite(weights, 'Query weights')
    if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('Query weights must be pure nonnegative bilinear weights summing to one')


def query_features(features, indices, weights):
    """Interpolate BCHW features to BNC without applying an altitude factor."""
    _query_contract(features, indices, weights)
    _finite(features, 'Features')
    selected = features.flatten(-2)[:, :, indices]
    return (selected * weights[None, None]).sum(-1).transpose(1, 2)


def query_surface(fields, query_indices, query_weights, heights):
    """Interpolate the common-height field, then restore surface ZWD in mm."""
    if fields.ndim == 4:
        fields = fields[:, None]
    if fields.ndim != 5 or fields.shape[2] != 1:
        raise ValueError('Fields must be BK1HW or B1HW')
    if heights.shape != (len(query_indices),):
        raise ValueError('Heights must contain one value per query')
    _finite(heights, 'Heights')
    batch, members = fields.shape[:2]
    values = query_features(fields.flatten(0, 1), query_indices, query_weights)[..., 0]
    result = values.reshape(batch, members, -1) * torch.exp(-heights / 3000.)
    _finite(result, 'Surface predictions')
    return result


def diagonal_gaussian_kl(mu_q, logstd_q, mu_p, logstd_p):
    """KL(q||p), summed over latent coordinates; inputs use log standard deviation."""
    if mu_q.ndim != 2 or any(x.shape != mu_q.shape for x in (logstd_q, mu_p, logstd_p)):
        raise ValueError('Gaussian parameters must share shape B-by-R')
    for value in (mu_q, logstd_q, mu_p, logstd_p):
        _finite(value, 'Gaussian parameters')
    result = (logstd_p - logstd_q + .5 * (
        torch.exp(2 * (logstd_q - logstd_p)) + (mu_q - mu_p).square() * torch.exp(-2 * logstd_p) - 1)).sum(-1)
    _finite(result, 'Gaussian KL')
    return result


def variational_loss(predictions, target, mask, mu_q, logstd_q, mu_p, logstd_p,
                     sigma_eff_mm, beta=1., reference_count=None):
    """Average hourly ELBO using one fixed count for both NLL sum and KL sum.

    The default is the fixed query count, never a variable per-hour denominator.
    Production runs supply their registered training-only median eligible count.
    """
    if predictions.ndim != 3 or target.shape != predictions.shape[::2] or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('Expected BKN predictions with BN target and boolean mask')
    if not mask.any(-1).all():
        raise ValueError('Every training hour must have valid observations; empty hours are ineligible')
    if predictions.shape[1] < 1 or sigma_eff_mm <= 0 or not math.isfinite(float(sigma_eff_mm)):
        raise ValueError('Members and effective discrepancy scale must be positive finite')
    reference = float(target.shape[-1] if reference_count is None else reference_count)
    if not math.isfinite(reference) or reference <= 0 or not math.isfinite(float(beta)) or beta < 0:
        raise ValueError('Reference count must be one positive finite scalar and beta nonnegative')
    # Sanitize BEFORE any nonlinear arithmetic so invalid NaNs have zero gradients.
    clean_prediction = torch.where(mask[:, None], predictions, 0.)
    clean_target = torch.where(mask, target, 0.)
    _finite(clean_prediction, 'Valid predictions')
    _finite(clean_target, 'Valid targets')
    gaussian_terms = .5 * ((clean_prediction - clean_target[:, None]) / sigma_eff_mm).square()
    gaussian_terms = gaussian_terms + math.log(float(sigma_eff_mm)) + .5 * math.log(2 * math.pi)
    nll = torch.where(mask[:, None], gaussian_terms, 0.).sum(-1).mean() / reference
    hourly_kl = diagonal_gaussian_kl(mu_q, logstd_q, mu_p, logstd_p)
    if len(hourly_kl) != len(target):
        raise ValueError('Gaussian parameters must match observation batch')
    kl = hourly_kl.mean() / reference
    return dict(loss=nll + beta * kl, nll=nll, kl=kl, kl_per_dim=hourly_kl.mean() / mu_q.shape[-1])


def sample_surface(fields, batch, noise, sigma_eff_mm):
    """Draw observation products; field products never contain station discrepancy."""
    means = query_surface(fields, batch['query_indices'], batch['query_weights'], batch['target_geometry'][:, 2])
    if noise.shape != means.shape or sigma_eff_mm <= 0 or not math.isfinite(float(sigma_eff_mm)):
        raise ValueError('Observation noise must match BKN and have positive finite scale')
    _finite(noise, 'Observation noise')
    result = means + sigma_eff_mm * noise
    _finite(result, 'Observation samples')
    return result
