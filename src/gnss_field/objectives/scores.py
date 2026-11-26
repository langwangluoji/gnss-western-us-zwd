import torch


def energy_score(samples, truth, mask=None, scale=1.):
    """Unbiased ES, samples[B,K,N], normalized by sqrt(valid N), then hours."""
    if samples.ndim != 3 or samples.shape[1] < 2 or truth.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError('ES requires [B,K>=2,N] samples and [B,N] truth')
    if mask is None:
        mask = torch.ones_like(truth, dtype=torch.bool)
    if not torch.isfinite(samples).all() or not torch.isfinite(truth[mask]).all():
        raise ValueError('Nonfinite forecast or scored observation')
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError('No valid ES targets')
    x = samples / scale
    y = torch.where(mask, truth, 0) / scale
    x = x * mask[:, None]
    first = torch.linalg.vector_norm(x-y[:, None], dim=-1).mean(-1)
    k = x.shape[1]
    # Explicit off-diagonal U-statistic; torch.cdist derivatives at zero are zero-safe.
    distances = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
    distances = distances.masked_fill(torch.eye(k, dtype=torch.bool, device=x.device), 0.)
    pair = distances.sum((-1, -2)) / (2*k*(k-1))
    return ((first-pair)/counts.sqrt()).mean()


def masked_mse(prediction, target, valid, scale=1., weights=None):
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite mean prediction or target')
    error = (prediction-torch.where(valid, target, 0))/scale
    if weights is None:
        return .5*error[valid].square().mean()
    return .5*(error.square()*valid*weights).sum(-1).mean()


def gaussian_nll(mean, log_std, target):
    return .5*((target-mean)*torch.exp(-log_std)).square().mean()+log_std.mean()


def edm_loss(model, clean, condition, *, p_mean=-1.2, p_std=1.2, sigma_data=.5):
    sigma = (torch.randn(len(clean), device=clean.device)*p_std+p_mean).exp()
    noisy = clean + torch.randn_like(clean)*sigma[:, None]
    prediction = model.denoise(noisy, sigma, condition)
    weight = (sigma.square()+sigma_data**2)/(sigma*sigma_data).square()
    return ((prediction-clean).square().mean(-1)*weight).mean()


def gradient_penalty(critic, real, fake, condition):
    alpha = torch.rand(len(real), 1, device=real.device)
    x = (real*alpha+fake.detach()*(1-alpha)).requires_grad_(True)
    score = critic(x, condition)
    grad = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
    return (torch.linalg.vector_norm(grad, dim=-1)-1).square().mean()
