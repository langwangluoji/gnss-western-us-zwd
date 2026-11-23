"""Population-locked inference; target labels are read only by this evaluator."""
from dataclasses import replace
from pathlib import Path
import io
import time

import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import atomic_bytes, fingerprint, read_json, sha256, write_json
from gnss_field.data import PackedStore, ContextBuilder, InputScenario
from gnss_field.geometry import (coarsen_100km, query_grid, surface_query,
                                  basis_observation_matrix, safe_coarsening_mask)
from .fields import predict_patches


def safe_query_geometry(store, ids):
    xy = store.qxy[ids].copy()
    xy[:, 0] = np.clip(xy[:, 0], 0, store.shape[1]-1)
    xy[:, 1] = np.clip(xy[:, 1], 0, store.shape[0]-1)
    return xy, store.stations.height.to_numpy()[ids]


def safe_fine_sites(store, ids):
    safe = safe_coarsening_mask(store.grid['region']).numpy()
    xy, _ = safe_query_geometry(store, ids)
    x = np.floor(xy[:, 0]).astype(int).clip(0, store.shape[1]-2)
    y = np.floor(xy[:, 1]).astype(int).clip(0, store.shape[0]-2)
    return safe[y, x] & safe[y+1, x] & safe[y, x+1] & safe[y+1, x+1] & store.query_valid[ids]


@torch.no_grad()
def evaluate_predictor(store, hours, predictor, output, *, members=64, role='C', scenario=InputScenario(),
                       pause_check=lambda: None, full_year=False, checkpoint_hash=None, use_history=True):
    """predictor(hours, context) returns mean, coarse, samples?,std? in surface mm."""
    from gnss_field.evaluation import evaluate_arrays, selection_values
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = store.ids(role)
    builder = ContextBuilder(store, use_history=use_history)
    context_policy = dict(use_history=bool(use_history), frame_offsets=[0, 6, 12, 18])
    hours = np.asarray(hours, dtype='i8')
    contract = fingerprint(dict(data=store.manifest['fingerprint'], grid=store.grid_id, hours=hours,
                                ids=ids, members=members, scenario=scenario.identity,
                                checkpoint=checkpoint_hash, context_policy=context_policy))
    result_path = output/'metrics.json'
    if result_path.exists():
        previous = read_json(result_path)
        if previous['contract'] != contract:
            raise ValueError('Evaluation artifact contract differs')
        return previous
    months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
    records = []
    for month in sorted(set(months)):
        selected = hours[months == month]
        path = output/f'forecast_{month}.npz'
        meta_path = path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            meta = read_json(meta_path)
            if meta['contract'] != contract or meta['sha256'] != sha256(path):
                raise ValueError('Partial evaluation shard mismatch')
        else:
            truth = store.targets(role, selected)
            f = np.isfinite(truth) & store.query_valid[ids]
            prediction, coarse, support, degraded, samples, std = [], [], [], [], [], []
            for hour in selected:
                pause_check()
                context = builder.build_context([hour], scenario)
                row = predictor(np.array([hour]), context, ids, members)
                prediction.append(np.asarray(row['mean'])[0])
                coarse.append(np.asarray(row['coarsened'])[0])
                support.append(builder.support(hour, InputScenario())[1][ids])
                degraded.append(builder.support(hour, scenario)[1][ids])
                if row.get('samples') is not None:
                    samples.append(np.asarray(row['samples'])[0])
                if row.get('std') is not None:
                    std.append(np.asarray(row['std'])[0])
            fields = dict(hours=selected, station_ids=ids, truth=truth, prediction=np.asarray(prediction),
                          coarsened_prediction=np.asarray(coarse), population_F=f, normal_support_S0=np.asarray(support),
                          degraded_support=np.asarray(degraded))
            if samples:
                fields['samples'] = np.asarray(samples)
            if std:
                fields['std'] = np.asarray(std)
            buf = io.BytesIO()
            np.savez_compressed(buf, **fields)
            atomic_bytes(path, buf.getvalue())
            write_json(meta_path, dict(contract=contract, sha256=sha256(path), count=len(selected)))
        with np.load(path) as z:
            records.append({key: z[key] for key in z.files})
    arrays = {key: np.concatenate([z[key] for z in records]) for key in records[0] if key != 'station_ids'}
    ids = records[0]['station_ids']
    metrics = evaluate_arrays(arrays['hours'], ids, arrays['truth'], arrays['prediction'], arrays['population_F'],
                               arrays['normal_support_S0'] & arrays['population_F'],
                               samples=arrays.get('samples'), std=arrays.get('std'),
                               metadata=store.stations.iloc[ids].reset_index(drop=True),
                               coarsened_prediction=arrays['coarsened_prediction'], safe_T=safe_fine_sites(store, ids))
    result = dict(contract=contract, metrics=metrics, values=selection_values(metrics),
                  members=members, full_year=full_year, role=role, hours=len(hours),
                  checkpoint_hash=checkpoint_hash, context_policy=context_policy, updated=time.time())
    write_json(result_path, result)
    return result


def mean_predictor(model, store, runtime):
    def predict(hours, context, ids, members):
        context = context.to(runtime.device)
        field, _ = predict_patches(model, context, store, chunk=runtime.patch_chunk)
        xy, height = safe_query_geometry(store, ids)
        mean = surface_query(field, xy, height).squeeze(1)
        coarse = surface_query(coarsen_100km(field), xy, height).squeeze(1)
        return dict(mean=mean.cpu().numpy(), coarsened=coarse.cpu().numpy())
    return predict


def functional_predictor(model, store, runtime, mean_cache, decoder, science, *, condition_provider=None):
    def predict(hours, context, ids, members):
        context = context.to(runtime.device)
        if not science.use_history and torch.count_nonzero(context.coarse[:, 1:, :6]):
            raise ValueError('Current-only inference requires masked historical values and availability')
        if context.scenario_hash != mean_cache.scenario_hash:
            raise ValueError('Rebuild frozen M0 under this outage before residual inference')
        context.mean = mean_cache.get(hours, device=runtime.device)
        h = model.condition(context, decoder.scale) if condition_provider is None else condition_provider(hours, context)
        h = h.to(runtime.device)
        if h.shape != (len(hours), science.condition_dim):
            raise ValueError('Frozen condition cache has a wrong shape')
        z = model.coefficients(h, context.hours, members, science.sampling_seed, science=science)
        xy, heights = safe_query_geometry(store, ids)
        xy = torch.as_tensor(xy, device=runtime.device, dtype=torch.float32)
        heights = torch.as_tensor(heights, device=runtime.device, dtype=torch.float32)
        base = surface_query(context.mean, xy, heights).squeeze(1)
        sample = decoder.station(z, base, xy, heights)
        coarse_phi = coarsen_100km(decoder.phi.T.reshape(science.rank, *store.shape)).reshape(science.rank, -1).T
        coarse_h = basis_observation_matrix(coarse_phi, store.shape, xy, heights)
        coarse_base = surface_query(coarsen_100km(context.mean), xy, heights).squeeze(1)
        coarse_samples = decoder.station(z, coarse_base, matrix=coarse_h)
        return dict(mean=sample.mean(1).cpu().numpy(), coarsened=coarse_samples.mean(1).cpu().numpy(),
                    samples=sample.cpu().numpy() if model.supports_joint else None)
    return predict


def pixel_predictor(model, store, runtime, mean_cache, science, builder):
    def predict(hours, context, ids, members):
        context = context.to(runtime.device)
        if not science.use_history and torch.count_nonzero(context.coarse[:, 1:, :6]):
            raise ValueError('Current-only inference requires masked historical values and availability')
        if context.scenario_hash != mean_cache.scenario_hash:
            raise ValueError('Rebuild frozen M0 under this outage before residual inference')
        context.mean = mean_cache.get(hours, device=runtime.device)
        field, features = predict_patches(model, context, store, chunk=runtime.patch_chunk,
                                         residual=True, scale=store.stats['z0_std'])
        xy, heights = safe_query_geometry(store, ids)
        f = query_grid(features, xy).transpose(1, 2)
        geometries = []
        for hour in hours:
            valid = np.isfinite(store.context_values([hour])[0]).all(-1)
            _, geo = builder._geometry(valid.tobytes(), 'stations')
            geometries.append(np.c_[heights/3000, np.minimum(geo['nearest'][ids]/400000, 2.5), geo['count'][ids]/16])
        geometry = torch.as_tensor(np.asarray(geometries), device=f.device, dtype=f.dtype)
        std = model.standard_deviation(f, geometry, store.stats['surface_std'], faithful=False)
        return dict(mean=surface_query(field, xy, heights).squeeze(1).cpu().numpy(),
                    coarsened=surface_query(coarsen_100km(field), xy, heights).squeeze(1).cpu().numpy(),
                    std=std.cpu().numpy())
    return predict
