"""Causal condition-only spatial anomalies at the residual's physical scale."""
import numpy as np
import torch

from .station_query import StationData


def leave_group_out_anomaly(values, height, xy, groups, scale_mm=10.):
    """Surface-mm LOO innovations, excluding every co-located physical group.

    Each output hour uses only that hour's visible conditions. The fixed
    150 km Gaussian / inverse-square kernel uses no labels or fitted scales.
    """
    xy = xy.float(); height = height.float(); values = values.float()
    distance = torch.cdist(xy, xy, compute_mode='donot_use_mm_for_euclid_dist')
    weight = torch.exp(-.5*(distance/150).square()) / distance.clamp_min(10).square()
    weight = weight * (distance <= 450) * (groups[:, None] != groups[None])
    observed = torch.isfinite(values)
    common = torch.where(observed, values*torch.exp(height[None]/3000), 0)
    denominator = observed.float() @ weight.T
    estimate = (common @ weight.T) / denominator.clamp_min(1e-8)
    estimate = estimate * torch.exp(-height[None]/3000)
    valid = observed & (denominator > 1e-8)
    anomaly = torch.where(valid, (values-estimate)/scale_mm, 0)
    return anomaly, valid


class AnomalyData(StationData):
    @torch.no_grad()
    def batch(self, anchors, context, development=False, temporal=True, features=True):
        b = super().batch(anchors, context, development, temporal, features)
        start, end = int(anchors[0]), int(anchors[-1])
        c = torch.as_tensor(context, device=self.values.device)
        raw = self.values[start-self.length+1:end+1, c, 0]
        groups = torch.as_tensor(np.asarray(self.events.geo['groups'][context]), device=raw.device)
        anomaly, valid = leave_group_out_anomaly(raw, b['cheight'], b['cxy'], groups)
        b['station_history'] = b['station_history'].clone()
        b['station_history'][..., 0] = anomaly.unfold(0, self.length, 1).permute(0, 2, 1)
        b['station_history'][..., 3] = valid.unfold(0, self.length, 1).permute(0, 2, 1)
        return b
