"""Materialize only the registered 768-hour local study, with resumable features."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from local_context_data import PROJECT, RUN, DESIGN, open_cache
from local_context_features import LocalFeatureBuilder, VertexQuery, fit_normalizer, project_operator
from local_probe_diagnostics import ProbeData, save_npz
from local_subspace_data import transform_condition
from local_conditional_gaussian import wet_feature
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256, fingerprint
from gnss_field.data import ContextBuilder
from gnss_field.data.context import cycles
from gnss_field.geometry import surface_query
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.training.observable import load_bundle
from gnss_field.runtime import configure

PILOT = PROJECT/'results/20261119_local_edm_subspace_v01'


def make_dataset():
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    data = ProbeData(device='cpu')
    cache = open_cache(data)
    if not cache.done.all():
        raise ValueError('Complete the sparse M0 cache first')
    design = read_json(DESIGN)
    bundle, manifest = load_bundle(PILOT/'server_bundle')
    if sha256(PILOT/'server_bundle/spatial.npz') != manifest['spatial_sha256']:
        raise ValueError('Frozen spatial factors checksum mismatch')
    spatial = dict(np.load(PILOT/'server_bundle/spatial.npz'))
    pca_path = PILOT/'condition_pca.npz'
    if sha256(pca_path) != manifest['frozen_contract']['pca']:
        raise ValueError('Frozen condition PCA differs')
    store, grid = data.store, data.store.grid
    ids, aid = store.ids('B'), store.ids('A')
    np.testing.assert_array_equal(ids, bundle['station_ids'])
    xy, heights = safe_query_geometry(store, ids)
    axy, _ = safe_query_geometry(store, aid)
    x, y = np.floor(xy).astype(int).T
    corners = np.stack([y*store.shape[1]+x, y*store.shape[1]+x+1,
                        (y+1)*store.shape[1]+x, (y+1)*store.shape[1]+x+1], -1)
    nodes = np.union1d(np.flatnonzero(grid['region']), corners.ravel())
    land = grid['region'].ravel()[nodes].astype(bool)
    query = VertexQuery(store.shape, xy, heights, nodes)
    s = spatial['phi'].astype('f8')@spatial['v'].astype('f8').T
    projection = surface_query(torch.tensor(s.T.reshape(64, *store.shape), dtype=torch.float64), xy, heights).T.numpy()
    np.testing.assert_allclose(projection, bundle['projection'], atol=.001, rtol=0)
    operator = project_operator(s[nodes], land)
    directory = RUN/'dataset'
    directory.mkdir(parents=True, exist_ok=True)
    dependencies = dict(design=sha256(DESIGN), M0_cache=sha256(cache.root/'mean.npy'),
        B_manifest=sha256(PILOT/'server_bundle/manifest.json'),
        feature_source=sha256(Path(__file__).with_name('local_context_features.py')),
        prepare_source=sha256(__file__), raw67_plus_frozen_PCA32=True,
        normalization='raw67 core hours only, across retained vertices; PCA32 frozen unchanged',
        kernels=dict(scales_km=[75, 150], cutoff_in_scales=3, gaussian_exponent=-.5),
        targets='B2018 only; historical frozen base and prior have B exposure', D2020=False)
    immutable_json(directory/'contract.json', dependencies)
    if (directory/'complete.json').exists():
        for name, digest in read_json(directory/'complete.json')['files'].items():
            if sha256(directory/name) != digest:
                raise ValueError(f'Dataset checksum mismatch: {name}')
        return
    path = directory/'features67.npy'
    features = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
        dtype='f4', shape=(len(cache.hours), len(nodes), 67))
    state_path = directory/'progress.json'
    done = read_json(state_path)['done'] if state_path.exists() else 0
    small_path = directory/'small.building.npz'
    small = dict(np.load(small_path)) if small_path.exists() else dict(
        condition=np.zeros((len(cache.hours), 32), 'f4'), mean=np.zeros((len(cache.hours), len(ids)), 'f4'),
        amplitude=np.zeros(len(cache.hours), 'f4'), gate=np.zeros(len(cache.hours), bool))
    builder = ContextBuilder(store)
    maker = LocalFeatureBuilder(np.c_[grid['x'].ravel(), grid['y'].ravel()], store.xy[aid],
        store.stations.height.to_numpy()[aid], grid['height'], grid['region'], nodes,
        store.stats, axy, store.query_valid[aid])
    pca = dict(np.load(pca_path))
    amplitude_state = read_json(PROJECT/'results/20261119_local_conditional_probability_v01/B_only_parameters.lock.json')
    mixture = read_json(PROJECT/'results/20261119_local_fused_probability_v02/B_only_parameters.lock.json')['weight']
    started = time.monotonic()
    for i in range(done, len(cache.hours)):
        if (RUN/'PAUSE').exists():
            raise RuntimeError('Requested pause before feature batch; resume is safe')
        hour = cache.hours[i]
        frames = np.array([store.context_values([hour-offset])[0] for offset in (0, 6, 12, 18)])
        m0 = cache.get(hour)
        ctx = builder.build_context([hour])
        global_feature = torch.cat([ctx.stations[0, :, :10].flatten(),
            F.adaptive_avg_pool2d(ctx.coarse.flatten(1, 2), (4, 4)).flatten(),
            torch.tensor(cycles([hour])[0])]).numpy()
        condition = transform_condition(global_feature[None], pca)[0]
        value = maker.build(frames, m0, condition, cycles([hour])[0])
        features[i] = value[:, :67]
        small['condition'][i] = condition
        published = m0.ravel()[nodes]+mixture*(value[:, 59]*data.scale-m0.ravel()[nodes])
        with torch.no_grad():
            small['mean'][i] = query(torch.tensor(published, dtype=torch.float32)).numpy()
        wet, present = wet_feature(frames[0:1])
        z = (wet[0]-amplitude_state['A_wetness_center'])/amplitude_state['A_wetness_std'] if present[0] else 0.
        small['amplitude'][i] = np.exp(amplitude_state['a']+amplitude_state['b']*z)
        small['gate'][i] = np.isfinite(frames).all(-1).any()
        if (i+1) % 24 == 0 or i+1 == len(cache.hours):
            features.flush()
            save_npz(small_path, **small)
            write_json(state_path, dict(done=i+1, elapsed_seconds=time.monotonic()-started))
            print('FEATURES', i+1, len(cache.hours), round(time.monotonic()-started, 1), flush=True)
    # Labels are loaded after input-only construction, solely for loss/evaluation.
    truth = store.targets('B', cache.hours)
    valid = data.manifest.masks(cache.hours)
    core = np.searchsorted(cache.hours, design['core_hours'])
    inner = np.searchsorted(cache.hours, design['inner_hours'])
    outer = np.searchsorted(cache.hours, design['outer_hours'])
    center, scale = fit_normalizer(features, core)
    arrays = dict(hours=cache.hours, station_ids=ids, truth=truth, valid=valid, nodes=nodes,
        land=land, query_indices=query.indices.numpy(), query_weights=query.weights.numpy(),
        projection=bundle['projection'], remainder=bundle['remainder'],
        operator=operator.astype('f4'), grid_basis=s[nodes].astype('f4'),
        center=center, scale=scale, core=core, inner=inner, outer=outer,
        folds=np.array(design['B_fold_assignment']), **small)
    save_npz(directory/'arrays.npz', **arrays)
    store.stations.iloc[ids].to_csv(directory/'stations.csv', index=False, encoding='utf-8-sig')
    write_json(directory/'settings.json', dict(tau=manifest['tau_surface_mm'], grid_scale=data.scale,
        shape=store.shape, baseline_weight=mixture, nodes=len(nodes), land_nodes=int(land.sum())))
    features.flush()
    write_json(directory/'complete.json', dict(files={p:sha256(directory/p) for p in
        ('features67.npy', 'arrays.npz', 'stations.csv', 'settings.json', 'contract.json')}))
    print('DATASET_COMPLETE', len(nodes), flush=True)


if __name__ == '__main__':
    make_dataset()
