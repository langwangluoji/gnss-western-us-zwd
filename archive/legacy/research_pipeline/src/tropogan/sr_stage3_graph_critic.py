"""Local node and geometric-pair critics for sparse GNSS residuals.

This scores observed nodes and nearby station pairs, not unseen dense pixels.
Real and fake calls must use identical legal conditions, coordinates and masks.
No target value participates in graph construction or context normalization.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _sn_linear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features, out_features), n_power_iterations=1)


def _nonnegative_weight(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def masked_graph_mean(value, mask):
    """Mean over valid entries; empty support returns a gradient-connected zero."""
    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Scores and Boolean mask must have identical shapes")
    if not torch.isfinite(value[mask]).all():
        raise ValueError("Valid graph scores must be finite")
    safe = torch.where(mask, value, torch.zeros_like(value))
    return safe.sum()/mask.sum().clamp_min(1).to(value.dtype)


def graph_discriminator_hinge(real_output, fake_output, *, node_weight=1., pair_weight=1.):
    """D minimizes mean(relu(1-real))+mean(relu(1+fake)), per support type.

    This helper does not detach fake values: the D caller must pass detached
    generator outputs. Node and pair means are separate, globally pooled over
    valid entries in the batch; a frame with more entries receives more weight.
    """
    node_weight = _nonnegative_weight(node_weight, "node_weight")
    pair_weight = _nonnegative_weight(pair_weight, "pair_weight")
    losses = {}
    for kind in ("node", "pair"):
        mask = real_output[kind+"_mask"]
        if not torch.equal(mask, fake_output[kind+"_mask"]):
            raise ValueError("Real/fake graph masks must be identical")
        losses[kind+"_loss"] = masked_graph_mean(F.relu(1-real_output[kind+"_scores"]), mask)
        losses[kind+"_loss"] += masked_graph_mean(F.relu(1+fake_output[kind+"_scores"]), mask)
    losses["total_loss"] = node_weight*losses["node_loss"]+pair_weight*losses["pair_loss"]
    return losses


def graph_generator_hinge(fake_output, *, node_weight=1., pair_weight=1.):
    """G minimizes negative local critic scores; use critic.frozen_for_generator()."""
    node_weight = _nonnegative_weight(node_weight, "node_weight")
    pair_weight = _nonnegative_weight(pair_weight, "pair_weight")
    node = -masked_graph_mean(fake_output["node_scores"], fake_output["node_mask"])
    pair = -masked_graph_mean(fake_output["pair_scores"], fake_output["pair_mask"])
    return {"node_loss": node, "pair_loss": pair, "total_loss": node_weight*node+pair_weight*pair}


class GraphResidualCritic(nn.Module):
    """Spectral-normalized node and local directed-pair residual critics.

    values_mm: [B,N] or [B,N,1], station ZWD in mm.
    condition: [B,N,11], surface5 + coarse W0/100 + current4 + Horth/1000.
    xy: [B,N,2], normalized cell-bound coordinates, default south-to-north.
    valid: Boolean [B,N], one available observation per physical group.

    Station background = condition[...,5]*100*exp(-condition[...,10]/3).
    Critic residual = (values_mm - station_background_mm)/10. The 3000 m
    scale-height representation is inherited from the field observation model,
    not a claim that the input background or normalized W0 is physical truth.

    Nodes use [legal condition, xy, residual]. Edges use endpoint contexts,
    relative xy (destination-source), haversine distance/100 km, residual
    difference (destination-source), and both endpoint residuals. Each source
    keeps <=k_neighbors directed edges within max_distance_km, excluding self.
    Equal distances resolve by destination x then y, then slot for identical
    coordinates. With unique coordinates the operation is permutation equivariant.
    Conditions and values never define adjacency; graph geometry is detached.

    Returns node_scores [B,N], node_mask [B,N], pair_scores [B,N,N],
    pair_mask [B,N,N], residuals_dimensionless [B,N], background_site_mm [B,N]
    and distance_km [B,N,N]. Invalid entries are zero. Distance is available for
    all distinct valid endpoint pairs, including pairs outside the selected graph.
    Optional node_features [B,N,width] and pair_features [B,N,N,width] permit
    later feature matching; no feature-matching loss is enabled implicitly.
    """
    def __init__(self, condition_dim=11, *, width=64, max_distance_km=300., k_neighbors=4,
                 lon_bounds=(4.875, 20.125), lat_bounds=(44.875, 55.125),
                 latitude_descending=False, residual_scale_mm=10.):
        super().__init__()
        if condition_dim != 11:
            raise ValueError("The legal condition contract has exactly 11 channels")
        if not isinstance(width, int) or isinstance(width, bool) or width < 4:
            raise ValueError("width must be an integer >=4")
        if not isinstance(k_neighbors, int) or isinstance(k_neighbors, bool) or k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")
        if not math.isfinite(max_distance_km) or max_distance_km <= 0:
            raise ValueError("max_distance_km must be finite and positive")
        if not math.isfinite(residual_scale_mm) or residual_scale_mm <= 0:
            raise ValueError("residual_scale_mm must be finite and positive")
        for name, bounds, low, high in (("longitude", lon_bounds, -180, 180), ("latitude", lat_bounds, -90, 90)):
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not low <= bounds[0] < bounds[1] <= high:
                raise ValueError(f"Invalid {name} cell bounds")
        self.condition_dim, self.width = condition_dim, width
        self.max_distance_km, self.k_neighbors = float(max_distance_km), k_neighbors
        self.lon_bounds, self.lat_bounds = tuple(lon_bounds), tuple(lat_bounds)
        self.latitude_descending = bool(latitude_descending)
        self.residual_scale_mm = float(residual_scale_mm)
        context_dim = condition_dim+2
        self.node_feature = nn.Sequential(_sn_linear(context_dim+1, width), nn.LeakyReLU(.2),
                                          _sn_linear(width, width), nn.LeakyReLU(.2))
        self.node_head = _sn_linear(width, 1)
        # Endpoint contexts + relative xy + distance + residual difference + two residuals.
        self.pair_feature = nn.Sequential(_sn_linear(2*context_dim+6, width), nn.LeakyReLU(.2),
                                          _sn_linear(width, width), nn.LeakyReLU(.2))
        self.pair_head = _sn_linear(width, 1)

    def _safe_coordinates(self, xy, valid):
        if xy.ndim != 3 or xy.shape[-1] != 2 or valid.shape != xy.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected xy [B,N,2] and Boolean valid [B,N]")
        if xy.shape[0] < 1 or xy.shape[1] < 1 or not xy.is_floating_point():
            raise ValueError("Nonempty batch/node slots and floating coordinates are required")
        if xy.device != valid.device:
            raise ValueError("Coordinates and mask must be on the same device")
        if not torch.isfinite(xy[valid]).all() or (xy[valid].abs() > 1+1e-6).any():
            raise ValueError("Valid coordinates must be finite and within normalized cell bounds")
        return torch.where(valid[..., None], xy, torch.zeros_like(xy))

    @torch.no_grad()
    def build_graph(self, xy, valid):
        """Return (directed_pair_mask, distance_km) using only coordinates/mask."""
        xy = self._safe_coordinates(xy, valid).detach()
        batch, count = valid.shape
        lon = self.lon_bounds[0]+(xy[..., 0]+1)*.5*(self.lon_bounds[1]-self.lon_bounds[0])
        fraction = (1-xy[..., 1])*.5 if self.latitude_descending else (xy[..., 1]+1)*.5
        lat = self.lat_bounds[0]+fraction*(self.lat_bounds[1]-self.lat_bounds[0])
        lon, lat = torch.deg2rad(lon), torch.deg2rad(lat)
        dlat, dlon = lat[:, :, None]-lat[:, None, :], lon[:, :, None]-lon[:, None, :]
        a = torch.sin(dlat*.5).square()+torch.cos(lat[:, :, None])*torch.cos(lat[:, None, :])*torch.sin(dlon*.5).square()
        distance = 6371.0088*2*torch.asin(torch.sqrt(a.clamp(0, 1)))
        endpoints = valid[:, :, None] & valid[:, None, :] & ~torch.eye(count, dtype=torch.bool, device=xy.device)[None]
        candidate = endpoints & (distance <= self.max_distance_km)
        ranked_distance = distance.masked_fill(~candidate, torch.inf)
        # Stable sort x(primary), y(secondary) before the stable distance sort.
        by_y = torch.argsort(xy[..., 1], dim=1, stable=True)
        sorted_x = xy[..., 0].gather(1, by_y)
        coordinate_order = by_y.gather(1, torch.argsort(sorted_x, dim=1, stable=True))
        coordinate_order = coordinate_order[:, None, :].expand(batch, count, count)
        arranged = ranked_distance.gather(2, coordinate_order)
        nearest_order = torch.argsort(arranged, dim=2, stable=True)[..., :min(self.k_neighbors, count)]
        destination = coordinate_order.gather(2, nearest_order)
        accepted = candidate.gather(2, destination)
        pair_mask = torch.zeros_like(candidate).scatter_(2, destination, accepted)
        return pair_mask, torch.where(endpoints, distance, torch.zeros_like(distance))

    @contextmanager
    def frozen_for_generator(self):
        """Freeze D weights and spectral-norm updates, retaining gradients to G.

        Use ``with critic.frozen_for_generator(): loss = ...; loss.backward()``.
        This is not torch.no_grad(). Exact module training flags and individual
        parameter requires_grad flags are restored, including after exceptions.
        """
        modes = [(module, module.training) for module in self.modules()]
        parameters = [(parameter, parameter.requires_grad) for parameter in self.parameters()]
        self.eval()
        self.requires_grad_(False)
        try:
            yield self
        finally:
            for module, mode in modes:
                module.training = mode
            for parameter, flag in parameters:
                parameter.requires_grad_(flag)

    def forward(self, values_mm, condition, xy, valid, *, return_features=False):
        if values_mm.ndim == 3 and values_mm.shape[-1] == 1:
            values_mm = values_mm.squeeze(-1)
        safe_xy = self._safe_coordinates(xy, valid)
        if values_mm.shape != valid.shape or condition.shape != (*valid.shape, self.condition_dim):
            raise ValueError("Expected station values [B,N] and condition [B,N,11]")
        if not values_mm.is_floating_point() or condition.dtype != values_mm.dtype or xy.dtype != values_mm.dtype:
            raise ValueError("Values, conditions and coordinates must share a floating dtype")
        if values_mm.device != condition.device or values_mm.device != xy.device:
            raise ValueError("Values, conditions and coordinates must share a device")
        if not torch.isfinite(values_mm[valid]).all() or not torch.isfinite(condition[valid]).all():
            raise ValueError("Valid values and legal conditions must be finite")
        value = torch.where(valid, values_mm, torch.zeros_like(values_mm))
        legal = torch.where(valid[..., None], condition, torch.zeros_like(condition))
        background = legal[..., 5]*100*torch.exp(-legal[..., 10]/3.)
        residual = (value-background)/self.residual_scale_mm
        if not torch.isfinite(residual[valid]).all():
            raise ValueError("Condition-derived background and residual must be finite")
        context = torch.cat((legal, safe_xy), dim=-1)
        node_features = self.node_feature(torch.cat((context, residual[..., None]), dim=-1))
        node_scores = self.node_head(node_features).squeeze(-1)
        node_scores = torch.where(valid, node_scores, torch.zeros_like(node_scores))
        pair_mask, distance = self.build_graph(xy, valid)
        batch_index, source, destination = pair_mask.nonzero(as_tuple=True)
        ri, rj = residual[batch_index, source], residual[batch_index, destination]
        relative_xy = safe_xy[batch_index, destination]-safe_xy[batch_index, source]
        pair_input = torch.cat((context[batch_index, source], context[batch_index, destination], relative_xy,
            (distance[batch_index, source, destination]/100.)[:, None], (rj-ri)[:, None], ri[:, None], rj[:, None]), dim=-1)
        pair_features = self.pair_feature(pair_input)
        selected_scores = self.pair_head(pair_features).squeeze(-1)
        pair_scores = values_mm.new_zeros(pair_mask.shape).index_put((batch_index, source, destination), selected_scores)
        result = {"node_scores": node_scores, "node_mask": valid.clone(), "pair_scores": pair_scores,
                  "pair_mask": pair_mask, "residuals_dimensionless": residual,
                  "background_site_mm": background, "distance_km": distance}
        if return_features:
            result["node_features"] = torch.where(valid[..., None], node_features, torch.zeros_like(node_features))
            result["pair_features"] = values_mm.new_zeros((*pair_mask.shape, self.width)).index_put(
                (batch_index, source, destination), pair_features)
        return result
