"""Verified frozen fusion background, separate from the new generative networks.

The readout reproduces the registered input-only geometry correction. Its old
training normalization remains fixed; new generative input statistics are separate.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gnss_field.artifacts import fingerprint, read_json, sha256


def frozen_ridge_correction(features, coefficients, scale):
    values = np.asarray(features, 'f8')
    center = np.asarray(coefficients['center'])
    std = np.asarray(coefficients['std'])
    beta = np.asarray(coefficients['beta'])
    if (values.shape[-1] != len(center) or len(beta) != len(center)+1
            or (std <= 0).any() or not np.isfinite(values).all()):
        raise ValueError('Invalid frozen ridge feature contract')
    return scale*((values-center)/std @ beta[:-1]+beta[-1])


class FrozenInnovationMoments:
    def __init__(self, xy, heights, input_xy, input_heights):
        delta = np.asarray(input_xy)[None]-np.asarray(xy)[:, None]
        distance = np.linalg.norm(delta, axis=-1)
        dh = np.asarray(input_heights)[None]-np.asarray(heights)[:, None]
        self.kernels = [];self.matrices = []
        for radius in (50000., 100000., 200000.):
            for vertical in (None, 750.):
                kernel = np.exp(-.5*(distance/radius)**2)*(distance <= 3*radius)
                if vertical is not None:
                    kernel *= np.exp(-.5*(dh/vertical)**2)
                kernel = kernel.astype('f4')
                self.kernels.append(kernel)
                self.matrices.append([kernel*factor.astype('f4') for factor in
                    (np.ones_like(distance), delta[:, :, 0]/radius,
                     delta[:, :, 1]/radius, dh/1000)])

    def __call__(self, residual, valid):
        valid = np.asarray(valid, 'f4')
        residual = np.where(valid, np.asarray(residual, 'f4'), 0.)
        columns = []
        for kernel, matrices in zip(self.kernels, self.matrices):
            denominator = np.maximum(valid @ kernel.T, 1e-30)
            columns.extend([(residual @ matrix.T)/denominator for matrix in matrices])
        return np.stack(columns, -1).astype('f4')


@dataclass(frozen=True)
class FusionBackgroundAssets:
    network_checkpoint: str
    geometry_coefficients: str
    fusion_settings: str
    expected_sha256: dict[str, str]
    geometry_readout_index: int = 0

    def verify(self):
        identity = {}
        for name in ('network_checkpoint', 'geometry_coefficients', 'fusion_settings'):
            path = Path(getattr(self, name))
            expected = self.expected_sha256.get(str(path))
            if expected is None or sha256(path) != expected:
                raise ValueError('Frozen background asset hash mismatch: '+name)
            identity[name] = expected
        if self.geometry_readout_index not in (0, 1):
            raise ValueError('Unregistered geometry readout index')
        return {**identity, 'geometry_readout_index': self.geometry_readout_index}


class FrozenFusionBackground:
    """Callable input-only provider for FullFieldData's verified adapter."""
    def __init__(self, store, assets: FusionBackgroundAssets, device='cuda', patch_chunk=8):
        self.identity = assets.verify()
        # These are pure frozen-inference/feature modules, not old experiment trainers.
        from local_context_features import LocalFeatureBuilder, VertexQuery
        from gnss_field.data import ContextBuilder
        from gnss_field.models.mean import MeanField
        from gnss_field.training.prediction import safe_query_geometry

        self.store = store;self.device = device;self.patch_chunk = patch_chunk
        self.builder = ContextBuilder(store)
        self.model = MeanField().to(device).eval().requires_grad_(False)
        checkpoint = torch.load(assets.network_checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['ema'])
        ids = store.ids('A');self.input_ids = ids
        nodes = np.arange(np.prod(store.shape))
        qxy, _ = safe_query_geometry(store, ids)
        query = VertexQuery(store.shape, qxy, np.zeros(len(ids)), nodes)
        self.input_indices = query.indices.numpy();self.input_weights = query.weights.numpy()
        xy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
        heights = store.stations.height.to_numpy()[ids]
        self.features = LocalFeatureBuilder(xy, store.xy[ids], heights, store.grid['height'],
            store.grid['region'], nodes, store.stats, qxy, store.query_valid[ids])
        self.moments = FrozenInnovationMoments(xy, store.grid['height'].ravel(), store.xy[ids], heights)
        settings = read_json(assets.fusion_settings)
        self.scale = float(settings['grid_scale']);self.weight = float(settings['baseline_weight'])
        if abs(self.scale-store.stats['z0_std']) > 1e-8:
            raise ValueError('Frozen fusion background normalization changed')
        selection = read_json(assets.geometry_coefficients)
        self.coefficients = selection['coefficients'][str(assets.geometry_readout_index)]['directional']
        self.identity.update(statistics=fingerprint(store.stats), grid=store.grid_id,
                             source='registered frozen input-only fusion field')

    def correct(self, frames, network_grid, hour):
        from gnss_field.data.context import cycles
        value = self.features.build(frames, network_grid, np.zeros(32, 'f4'), cycles([hour])[0])[:, :67].copy()
        density = [5, 11, 17, 23, 29, 35, 41, 47, 49, 51]
        value[:, density] = np.log1p(value[:, density])/np.log1p(174.)
        base = np.asarray(network_grid).ravel()+self.weight*(value[:, 59]*self.scale-np.asarray(network_grid).ravel())
        base_at_input = (base[self.input_indices]*self.input_weights).sum(1)
        valid = np.isfinite(frames[0]).all(1) & self.store.query_valid[self.input_ids]
        residual = np.where(valid, frames[0, :, 0]-base_at_input, 0.)
        innovations = []
        for kernel in self.features.kernels:
            weights = np.asarray(kernel, 'f8')*valid[None]
            innovations.append((weights @ residual)/np.maximum(weights.sum(1), 1e-30))
        design = np.concatenate([value, np.stack(innovations, -1).astype('f4')/self.scale,
                                 self.moments(residual, valid)/self.scale], -1)
        field = (base+frozen_ridge_correction(design, self.coefficients, self.scale)).astype('f4')
        if not np.isfinite(field).all():
            raise ValueError('Nonfinite frozen fusion background')
        return field.reshape(self.store.shape)

    def __call__(self, hour_ids, frames):
        from gnss_field.training.fields import predict_patches
        hours = np.asarray(hour_ids, 'i8')
        frames = np.asarray(frames)
        if frames.shape != (len(hours), 4, len(self.input_ids), 5):
            raise ValueError('Frozen background requires raw four-frame input observations')
        if str(self.device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            context = self.builder.build_context(hours).to(self.device)
            field = predict_patches(self.model, context, self.store, chunk=self.patch_chunk)[0][:, 0].cpu().numpy()
        return np.stack([self.correct(row, grid, int(hour))
                         for hour, row, grid in zip(hours, frames, field)])[:, None]
