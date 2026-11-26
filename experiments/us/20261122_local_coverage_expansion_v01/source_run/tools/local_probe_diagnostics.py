"""Frozen-data diagnostics and small local condition caches for the GNSS probe."""
from pathlib import Path
import io
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))

from gnss_field.artifacts import atomic_bytes, fingerprint, immutable_json, read_json, sha256, write_json
from gnss_field.data import PackedStore, ContextBuilder
from gnss_field.data.context import cycles
from gnss_field.data.manifests import build_training_manifest, spatial_folds, stratified_hours
from gnss_field.evaluation import evaluate_arrays
from gnss_field.evaluation.metrics import increment_metrics
from gnss_field.geometry import surface_query, coarsen_100km, basis_observation_matrix, curvature_matrix
from gnss_field.models.basis import SpatialBasis
from gnss_field.models.mean import MeanField
from gnss_field.objectives import fit_latent
from gnss_field.training.baselines import get_baseline_predictor
from gnss_field.training.fields import predict_patches
from gnss_field.training.prediction import safe_query_geometry, safe_fine_sites

FOUNDATION = PROJECT/'results/20261119_local_basis_diagnosis'
OUTPUT = PROJECT/'results/20261119_local_condition_probe_v01'
MEAN = PROJECT/'results/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/seed_42/mean_base/candidate_005000.pt'


def emit(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def save_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def split_head_hours(hours):
    hours = np.asarray(hours, dtype='i8')
    blocks = (hours-hours[0])//168
    unique = np.unique(blocks)
    development_blocks = np.random.default_rng(42).permutation(unique)[::4]
    development = np.isin(blocks, development_blocks)
    return hours[~development], hours[development]


def score_arrays(store, role, hours, truth, prediction, valid, coarse=None):
    ids = store.ids(role)
    return evaluate_arrays(hours, ids, truth, prediction, valid, valid,
                           coarsened_prediction=coarse,
                           metadata=store.stations.iloc[ids].reset_index(drop=True),
                           safe_T=safe_fine_sites(store, ids))


class ProbeData:
    def __init__(self, device='cuda'):
        self.device = device
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.store = PackedStore(PROJECT/'../../02_data/processed/gnss_field_server_v1', purpose='train')
        self.development = PackedStore(self.store.root, purpose='development')
        self.manifest = build_training_manifest(self.store, PROJECT/'local_validation/training_manifest.npz')
        self.assignment = spatial_folds(self.store, 20261117)
        self.scale = self.store.stats['z0_std']
        self.old_hours = np.array(read_json(FOUNDATION/'run_config.json')['contract']['hours'])
        self.extra_hours = np.unique(np.concatenate(read_json(FOUNDATION/'confirmation/candidate.lock.json')['fold_hours']))
        self.hours = np.sort(np.r_[self.old_hours, self.extra_hours])
        assert len(np.unique(self.hours)) == len(self.hours) == 864
        self.old_mean = np.load(FOUNDATION/'mean_subset.npy', mmap_mode='r')
        self.extra_mean = np.load(FOUNDATION/'confirmation/mean.npy', mmap_mode='r')
        self.old_support = np.load(FOUNDATION/'support_subset.npy', mmap_mode='r')
        self.extra_support = np.load(FOUNDATION/'confirmation/support.npy', mmap_mode='r')
        self.mean_hash = sha256(MEAN)
        assert self.mean_hash == '605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0'
        self.contract = dict(mean=self.mean_hash, training=self.manifest.contract,
                             data=self.store.manifest['fingerprint'], grid=self.store.grid_id,
                             input='joint_five_valid_A_four_frames_v1', feature_pool=[4, 4])
        immutable_json(OUTPUT/'foundation.lock.json', self.contract)
        self._arrays = {}

    def frozen_field(self, hour):
        index = np.searchsorted(self.old_hours, hour)
        if index < len(self.old_hours) and self.old_hours[index] == hour:
            return np.array(self.old_mean[index]), np.array(self.old_support[index])
        index = np.searchsorted(self.extra_hours, hour)
        if index >= len(self.extra_hours) or self.extra_hours[index] != hour:
            raise ValueError('Hour missing from frozen local field cache')
        return np.array(self.extra_mean[index]), np.array(self.extra_support[index])

    def arrays(self, year):
        if year in self._arrays:
            return self._arrays[year]
        path = OUTPUT/f'conditions_{year}.npz'
        if path.exists():
            meta = read_json(path.with_suffix('.json'))
            assert meta['contract'] == fingerprint(self.contract) and meta['sha256'] == sha256(path)
            with np.load(path) as z:
                self._arrays[year] = {key: z[key] for key in z.files}
            return self._arrays[year]
        store, role = (self.store, 'B') if year == 2018 else (self.development, 'C')
        hours = self.hours if year == 2018 else stratified_hours(np.array(store.hours), 2019, per_cell=1)
        builder = ContextBuilder(store)
        ids = store.ids(role)
        xy, heights = safe_query_geometry(store, ids)
        a_ids = store.ids('A')
        a_xy, a_height = safe_query_geometry(store, a_ids)
        idw = get_baseline_predictor('A_IDW', store)
        net = None
        if year == 2019:
            net = MeanField().to(self.device).eval().requires_grad_(False)
            net.load_state_dict(torch.load(MEAN, map_location=self.device, weights_only=False)['ema'])
        rows = {key: [] for key in ('features', 'baseline', 'coarse', 'idw', 'mean_A', 'observed_A', 'valid_A', 'support', 'normal_support')}
        means = []
        with torch.inference_mode():
            for j, hour in enumerate(hours):
                context = builder.build_context([hour])
                if year == 2018:
                    value, support = self.frozen_field(hour)
                    mean = torch.tensor(value[None])
                else:
                    mean = predict_patches(net, context.to(self.device), store, chunk=4)[0].cpu()
                    support = context.maps[0, 3].bool().numpy()
                assert torch.isfinite(mean).all()
                feature = torch.cat([context.stations[0, :, :10].flatten(),
                                     F.adaptive_avg_pool2d(context.coarse.flatten(1, 2), (4, 4)).flatten(),
                                     torch.tensor(cycles([hour])[0]),
                                     F.adaptive_avg_pool2d(mean/self.scale, (4, 4)).flatten()])
                raw = store.context_values([hour])[0]
                valid_a = np.isfinite(raw).all(-1) & store.query_valid[a_ids]
                observed_a = np.nan_to_num(raw[:, 0])*np.exp(-a_height/3000)
                rows['features'].append(feature.numpy())
                rows['baseline'].append(surface_query(mean, xy, heights).numpy()[0, 0])
                rows['coarse'].append(surface_query(coarsen_100km(mean), xy, heights).numpy()[0, 0])
                rows['idw'].append(idw([hour], context, ids, 1)['mean'][0])
                rows['mean_A'].append(surface_query(mean, a_xy, a_height).numpy()[0, 0])
                rows['observed_A'].append(observed_a)
                rows['valid_A'].append(valid_a)
                rows['support'].append(support)
                rows['normal_support'].append(builder.support(hour)[1][ids])
                if year == 2019:
                    means.append(mean.numpy()[0])
                if (j+1) % 48 == 0:
                    emit(f'condition cache {year}: {j+1}/{len(hours)}')
        truth = store.targets(role, hours)
        valid = (self.manifest.masks(hours) if year == 2018 else
                 np.isfinite(truth) & store.query_valid[ids] & np.array(rows['normal_support']))
        arrays = dict(hours=hours, station_ids=ids, truth=truth, valid=valid,
                      **{key: np.array(value) for key, value in rows.items()})
        if means:
            arrays['mean_fields'] = np.array(means)
        save_npz(path, **arrays)
        write_json(path.with_suffix('.json'), dict(contract=fingerprint(self.contract), sha256=sha256(path)))
        self._arrays[year] = arrays
        del net
        return arrays

    def basis(self, fold):
        path = FOUNDATION/f'server/seed_42/basis_cv_{fold}/last.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['spec']['dependencies']['mean'] == self.mean_hash
        np.testing.assert_array_equal(state['spec']['dependencies']['fit_station_mask'], self.assignment != fold)
        model = SpatialBasis(128, 256).to(self.device).eval()
        model.load_state_dict(state['ema'])
        grid = self.store.grid
        xyh = torch.tensor(np.c_[grid['x'].ravel(), grid['y'].ravel(), grid['height'].ravel()], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            phi = model(xyh, torch.tensor(grid['region'].ravel(), device=self.device, dtype=torch.bool)).detach()
        return phi, sha256(path)


def audit_population(data):
    folder = PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'
    records = [dict(np.load(path)) for path in sorted(folder.glob('forecast_*.npz'))]
    assert len(records) == 12
    ids = records[0]['station_ids']
    truth = np.concatenate([r['truth'] for r in records])
    f0 = np.concatenate([r['population_F'] for r in records])
    support = np.concatenate([r['normal_support_S0'] for r in records]) & f0
    safe = safe_fine_sites(data.development, ids)
    masks = {'valid_observation': np.isfinite(truth), 'query_F0': f0,
             'normal_support_S0': support, 'common_coarsening_T': support & safe[None]}
    metadata = data.development.stations.iloc[ids].reset_index(drop=True)
    rows = []
    for name, population in masks.items():
        pairs = increment_metrics(truth, np.zeros_like(truth), population, metadata)['0-50']
        rows.append(dict(stage=name, stations=int(population.any(0).sum()),
                         station_hours=int(population.sum()), groups_0_50=pairs['n_groups'],
                         pairs_0_50=pairs['n_pairs'], pair_hours_0_50=pairs['n_pair_hours']))
    assert rows[2]['stations'] == 76 and rows[2]['station_hours'] == 642258
    assert rows[3]['stations'] == 23 and rows[3]['pairs_0_50'] == 1
    write_json(OUTPUT/'population_audit.json', rows)
    table = metadata.copy()
    for name, population in masks.items():
        table[name+'_hours'] = population.sum(0)
    table['safe_T'] = safe
    table.to_csv(OUTPUT/'station_retention.csv', index=False, encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    for ax, (name, color) in zip(axes, [('query_F0', '#777777'), ('normal_support_S0', '#c2721c'), ('common_coarsening_T', '#246c9a')]):
        keep = masks[name].any(0)
        ax.scatter(metadata.x/1000, metadata.y/1000, s=8, color='#dddddd')
        ax.scatter(metadata.x[keep]/1000, metadata.y[keep]/1000, s=20, color=color)
        ax.set(title=f'{name}: {keep.sum()} stations', xlabel='Lambert x (km)', ylabel='Lambert y (km)')
        ax.set_aspect('equal')
    fig.savefig(OUTPUT/'evaluation_coverage.png', dpi=170)
    plt.close(fig)
    emit(f'population audit: {rows}')


def inverse_diagnostics(data):
    arrays = data.arrays(2018)
    store, device = data.store, data.device
    bxy, bh = safe_query_geometry(store, store.ids('B'))
    axy, ah = safe_query_geometry(store, store.ids('A'))
    results = []
    for fold in range(3):
        phi, digest = data.basis(fold)
        hb = basis_observation_matrix(phi, store.shape, bxy, bh)
        ha = basis_observation_matrix(phi, store.shape, axy, ah)
        coarse_phi = coarsen_100km(phi.T.reshape(128, 1, *store.shape))
        hc = surface_query(coarse_phi, bxy, bh)[:, 0].T
        fit = data.assignment != fold
        predictions = {key: [] for key in ('M0_fine_residual_fine', 'M0_fine_residual_coarse',
                                          'M0_coarse_residual_fine', 'M0_coarse_residual_coarse', 'A_inverse')}
        grams = []
        degrees = []
        counts = []
        with torch.no_grad():
            for j, hour in enumerate(arrays['hours']):
                support = torch.tensor(arrays['support'][j], device=device)
                curvature = curvature_matrix(phi, store.shape, support)
                cd = curvature.double()
                gram = cd.T@cd/max(1, len(cd))
                grams.append(gram.float().cpu().numpy())
                mask = arrays['valid'][j] & fit
                base = torch.tensor(arrays['baseline'][j], device=device)
                base_coarse = torch.tensor(arrays['coarse'][j], device=device)
                target = torch.tensor(arrays['truth'][j, mask], device=device)
                coefficient, _ = fit_latent(hb[mask], (target-base[mask])/data.scale, curvature,
                                           penalty=1., curvature_weight=.001)
                coefficient = coefficient.float()*.25
                fine_delta = data.scale*(hb@coefficient)
                coarse_delta = data.scale*(hc@coefficient)
                for name, value in [('M0_fine_residual_fine', base+fine_delta),
                                    ('M0_fine_residual_coarse', base+coarse_delta),
                                    ('M0_coarse_residual_fine', base_coarse+fine_delta),
                                    ('M0_coarse_residual_coarse', base_coarse+coarse_delta)]:
                    predictions[name].append(value.cpu().numpy())
                available = arrays['valid_A'][j]
                counts.append(int(available.sum()))
                if available.any():
                    residual = (arrays['observed_A'][j, available]-arrays['mean_A'][j, available])/data.scale
                    ac, _ = fit_latent(ha[available], torch.tensor(residual, device=device), curvature,
                                       penalty=1., curvature_weight=.001)
                    a_prediction = base+.25*data.scale*(hb@ac.float())
                else:
                    a_prediction = base
                predictions['A_inverse'].append(a_prediction.cpu().numpy())
                if j % 72 == 0:
                    design = hb[mask].double()
                    normal = design.T@design/len(design)
                    penalty = torch.eye(128, device=device, dtype=torch.float64)/128+.001*gram
                    degrees.append(float(torch.trace(torch.linalg.solve(normal+penalty, normal))))
                if (j+1) % 144 == 0:
                    emit(f'inverse diagnostics fold {fold}: {j+1}/{len(arrays["hours"])}')
            values = phi[torch.tensor(store.grid['region'].ravel(), device=device, dtype=torch.bool)].double()
            gram = values.T@values/len(values)
            eigenvalues = torch.linalg.eigvalsh(gram).cpu().numpy()
        np.save(OUTPUT/f'curvature_gram_fold_{fold}.npy', np.array(grams))
        valid = arrays['valid'] & ~fit
        metrics = {}
        for name, prediction in {**predictions, 'M0': arrays['baseline'], 'A_IDW': arrays['idw']}.items():
            prediction = np.asarray(prediction)
            metrics[name] = score_arrays(store, 'B', arrays['hours'], arrays['truth'], prediction, valid)
            common = valid & safe_fine_sites(store, store.ids('B'))[None]
            metrics[name]['common_T'] = score_arrays(store, 'B', arrays['hours'], arrays['truth'], prediction, common)
        save_npz(OUTPUT/f'inverse_fold_{fold}.npz', hours=arrays['hours'], **{k: np.array(v) for k, v in predictions.items()})
        row = dict(fold=fold, basis_sha256=digest, metrics=metrics,
                   basis_gram_eigenvalues=eigenvalues.tolist(),
                   gram_participation_rank=float(eigenvalues.sum()**2/(eigenvalues**2).sum()),
                   sampled_regularized_degrees=degrees,
                   available_A_min=min(counts), available_A_median=float(np.median(counts)),
                   zero_A_hours=int(np.sum(np.array(counts) == 0)),
                   interpretation='Exploratory 864-hour B group diagnostics; current B used only by B-assisted arms')
        results.append(row)
        write_json(OUTPUT/f'inverse_fold_{fold}.json', row)
    write_json(OUTPUT/'inverse_diagnostics.json', results)


if __name__ == '__main__':
    from gnss_field.runtime import configure
    from types import SimpleNamespace
    configure(SimpleNamespace(cpus=2), 42)
    data = ProbeData()
    audit_population(data)
    inverse_diagnostics(data)
