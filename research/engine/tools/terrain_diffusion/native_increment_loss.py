"""Within-origin lead differences; the legacy cross-origin h0 term is unchanged."""
import torch
from .native_weather_experiment import mean_loss


def within_forecast_increment_loss(correction,truth,base,weights):
    valid=torch.isfinite(truth)
    # The same base is added at every lead of one origin and cancels exactly.
    # Sanitize both operands before arithmetic, including invalid model cells.
    error=(torch.where(valid,correction,0)-torch.where(valid,truth,0))/10
    adjacent=valid[...,1:]&valid[...,:-1]
    weight=weights[...,None]*adjacent
    difference=torch.diff(error,dim=-1)
    per_lead=(difference.square()*weight).sum((0,1))/weight.sum((0,1)).clamp_min(1)
    return per_lead.mean()


def increment_mean_loss(correction,truth,base,weights,increment_weight=.5):
    if increment_weight<0:raise ValueError('increment weight must be nonnegative')
    old=mean_loss(correction,truth,base,weights)
    if increment_weight==0:return old
    return old+increment_weight*within_forecast_increment_loss(correction,truth,base,weights)
