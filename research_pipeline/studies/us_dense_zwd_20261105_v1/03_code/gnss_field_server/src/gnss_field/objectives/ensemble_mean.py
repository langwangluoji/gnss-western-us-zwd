"""Independent-member U estimator of squared predictive-mean error."""
import torch


def unbiased_mean_mse(samples, truth, mask):
    """BKN samples and BN labels: station mean per hour, then hour mean.

    Inputs must already share units/normalization. The estimator can be negative;
    neither its value nor its variance-correction gradient may be clipped away.
    """
    if samples.ndim != 3 or samples.shape[1] < 2 or samples.shape[::2] != truth.shape or mask.shape != truth.shape:
        raise ValueError('Expected BKN with K>=2 and BN labels/mask')
    if not torch.isfinite(samples).all() or not torch.isfinite(truth[mask]).all():
        raise ValueError('Nonfinite prediction or scored label')
    count = mask.sum(-1)
    if (count == 0).any():
        raise ValueError('Empty scoring hour')
    errors = torch.where(mask[:, None], samples-torch.where(mask, truth, 0.)[:, None], 0.)
    k = samples.shape[1]
    per_station = (errors.sum(1).square()-errors.square().sum(1))/(k*(k-1))
    return (per_station.sum(-1)/count).mean()
