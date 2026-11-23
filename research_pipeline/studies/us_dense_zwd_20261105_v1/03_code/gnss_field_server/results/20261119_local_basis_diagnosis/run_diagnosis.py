"""Reproduce frozen three-fold diagnostics and inspect inverse regularization locally."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'results/20261119_frozen_m0_audit/release/src'))

from gnss_field.artifacts import fingerprint, sha256, write_json, atomic_numpy
from gnss_field.data import PackedStore, ContextBuilder
from gnss_field.data.manifests import build_training_manifest, spatial_folds, stratified_hours
from gnss_field.evaluation import evaluate_arrays
from gnss_field.geometry import (surface_query, basis_observation_matrix,
                                 coarsen_100km, curvature_matrix)
from gnss_field.models.basis import SpatialBasis
from gnss_field.models.mean import MeanField
from gnss_field.objectives import fit_latent, observability
from gnss_field.runtime import configure
from gnss_field.training.fields import predict_patches
from gnss_field.training.prediction import safe_query_geometry, safe_fine_sites
from types import SimpleNamespace


def emit(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def summarize(bundle):
    fine = bundle['fine']
    return dict(
        rmse_mm=fine['model']['station_macro_rmse_mm'],
        mae_mm=fine['model']['station_macro_mae_mm'],
        bias_mm=fine['model']['station_macro_bias_mm'],
        coarse_rmse_mm=fine['coarsened']['station_macro_rmse_mm'],
        increment_rmse_mm=fine['model_increments']['0-50']['group_macro_rmse_mm'],
        coarse_increment_rmse_mm=fine['coarsened_increments']['0-50']['group_macro_rmse_mm'],
        n_stations=fine['model']['n_stations'],
        n_station_hours=fine['model']['n_station_hours'],
        n_pairs=fine['model_increments']['0-50']['n_pairs'])


def metric_delta(actual, expected):
    return max(abs(actual[key] - expected[key]) for key in (
        'rmse_mm', 'coarse_rmse_mm', 'increment_rmse_mm', 'coarse_increment_rmse_mm'))


def main():
    started = time.monotonic()
    configure(SimpleNamespace(cpus=2), 42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    store = PackedStore(PROJECT / '../../02_data/processed/gnss_field_server_v1', purpose='train')
    manifest = build_training_manifest(store, PROJECT / 'local_validation/training_manifest.npz')
    config = json.loads((ROOT / 'server/resolved.json').read_text())['science']
    assignments = spatial_folds(store, config['study_seed'])
    checkpoint = PROJECT / 'results/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/seed_42/mean_base/candidate_005000.pt'
    mean_hash = sha256(checkpoint)
    ids = store.ids('B')
    xy, heights = safe_query_geometry(store, ids)
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    fine_sites = safe_fine_sites(store, ids)
    scale = store.stats['z0_std']
    folds = []
    for fold in range(3):
        path = ROOT / f'server/seed_42/basis_cv_{fold}/last.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        dependencies = state['spec']['dependencies']
        fit = assignments != fold
        assert state['step'] == 3000
        assert dependencies['mean'] == mean_hash
        assert dependencies['training_manifest'] == manifest.contract
        assert dependencies['training_data'] == store.manifest['training_hash']
        assert dependencies['grid'] == store.grid_id
        np.testing.assert_array_equal(fit, dependencies['fit_station_mask'])
        for line in (ROOT / 'download.log').read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1] == f'seed_42/basis_cv_{fold}/last.pt':
                assert sha256(path) == parts[0]
                break
        else:
            raise ValueError('No downloaded checkpoint checksum')
        hours = stratified_hours(manifest.select([2018], config['rank']+8, fit), 2018)
        expected = json.loads((path.parent / 'diagnostics.json').read_text())
        assert len(hours) == expected['hours'] == 576
        folds.append((state, fit, hours, expected))
    hours_all = np.unique(np.concatenate([row[2] for row in folds]))
    contract = dict(mean_sha256=mean_hash, training=manifest.contract,
                    data=store.manifest['training_hash'], grid=store.grid_id,
                    hours=hours_all.tolist(), schema='local_sparse_mean_v1')
    cache_contract = fingerprint(contract)
    write_json(ROOT / 'run_config.json', dict(contract=contract, device=device,
               gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
               torch=torch.__version__, penalties=[.01, .1, 1.], amplitudes=[0., .25, .5, 1.],
               interpretation='2018 outer held B groups; diagnostic sweep, no product selection'))
    emit(f'device={device}; {len(hours_all)} unique hours; three frozen 3000-step bases')
    builder = ContextBuilder(store)
    mean = MeanField().to(device).eval().requires_grad_(False)
    mean.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['ema'])
    cache = ROOT / 'mean_subset.npy'
    support_cache = ROOT / 'support_subset.npy'
    cursor_file = ROOT / 'mean_cursor.json'
    if cache.exists():
        cursor = json.loads(cursor_file.read_text())
        assert cursor['contract'] == cache_contract
        means = np.lib.format.open_memmap(cache, mode='r+')
        supports = np.lib.format.open_memmap(support_cache, mode='r+')
        done = cursor['done']
    else:
        means = np.lib.format.open_memmap(cache, mode='w+', dtype='f4', shape=(len(hours_all), 1, *store.shape))
        supports = np.lib.format.open_memmap(support_cache, mode='w+', dtype='bool', shape=(len(hours_all), *store.shape))
        done = 0
        write_json(cursor_file, dict(contract=cache_contract, done=done))
    with torch.inference_mode():
        for i in range(done, len(hours_all)):
            context = builder.build_context(hours_all[i:i+1], device=device)
            value = predict_patches(mean, context, store, chunk=4)[0]
            assert torch.isfinite(value).all()
            means[i] = value[0].cpu().numpy()
            supports[i] = context.maps[0, 3].bool().cpu().numpy()
            if (i+1) % 12 == 0 or i+1 == len(hours_all):
                means.flush()
                supports.flush()
                write_json(cursor_file, dict(contract=cache_contract, done=i+1))
                emit(f'M0 cache {i+1}/{len(hours_all)}; elapsed {time.monotonic()-started:.1f}s')
    del mean, builder
    if device == 'cuda':
        torch.cuda.empty_cache()

    xyh = torch.tensor(np.c_[store.grid['x'].ravel(), store.grid['y'].ravel(),
                            store.grid['height'].ravel()], device=device, dtype=torch.float32)
    land = torch.tensor(store.grid['region'].ravel(), device=device, dtype=torch.bool)
    combinations = [(penalty, alpha) for penalty in (.01, .1, 1.) for alpha in (0., .25, .5, 1.)]
    summaries = []
    for fold, (state, fit, hours, expected) in enumerate(folds):
        emit(f'fold {fold}: inverse sweep over {len(hours)} hours')
        model = SpatialBasis(config['rank'], config['basis_width']).to(device).eval()
        model.load_state_dict(state['ema'])
        with torch.no_grad():
            phi = model(xyh, land).detach()
            h = basis_observation_matrix(phi, store.shape, xy, heights)
            coarse_phi = coarsen_100km(phi.T.reshape(config['rank'], 1, *store.shape))
            h_coarse = surface_query(coarse_phi, xy, heights)[:, 0].T
        truth = store.targets('B', hours)
        qualified = manifest.masks(hours)
        valid = qualified & ~fit
        predictions = np.empty((len(combinations), len(hours), len(ids)), dtype='f4')
        coarse_predictions = np.empty_like(predictions)
        baseline = np.empty_like(truth)
        baseline_coarse = np.empty_like(truth)
        fit_error, held_error, ranks, sensitivity = [], [], [], []
        singular_rows = []
        max_equivalence_error = 0.
        with torch.no_grad():
            for j, hour in enumerate(hours):
                index = int(np.searchsorted(hours_all, hour))
                base_field = torch.tensor(np.array(means[index:index+1]), device=device)
                support = torch.tensor(np.array(supports[index]), device=device)
                base = surface_query(base_field, xy, heights).flatten()
                base_coarse = surface_query(coarsen_100km(base_field), xy, heights).flatten()
                baseline[j] = base.cpu().numpy()
                baseline_coarse[j] = base_coarse.cpu().numpy()
                mask = qualified[j] & fit
                design = h[mask]
                residual = (torch.tensor(truth[j, mask], device=device)-base[mask])/scale
                curvature = curvature_matrix(phi, store.shape, support)
                for penalty in (.01, .1, 1.):
                    coefficient, _ = fit_latent(design, residual, curvature, penalty=penalty,
                                               curvature_weight=config['curvature_weight'])
                    coefficient = coefficient.float()
                    correction = scale*(h@coefficient)
                    coarse_correction = scale*(h_coarse@coefficient)
                    for alpha in (0., .25, .5, 1.):
                        k = combinations.index((penalty, alpha))
                        predictions[k, j] = (base+alpha*correction).cpu().numpy()
                        coarse_predictions[k, j] = (base_coarse+alpha*coarse_correction).cpu().numpy()
                    if penalty == .01:
                        if j < 8:
                            field = base_field+scale*(phi@coefficient).reshape(1, 1, *store.shape)
                            direct = surface_query(field, xy, heights).flatten()
                            direct_coarse = surface_query(coarsen_100km(field), xy, heights).flatten()
                            error = max((direct-base-correction).abs().max().item(),
                                        (direct_coarse-base_coarse-coarse_correction).abs().max().item())
                            max_equivalence_error = max(max_equivalence_error, error)
                        fit_error.append(float(((base+correction)[mask]-torch.tensor(truth[j, mask], device=device)).square().mean()))
                        held = valid[j]
                        held_error.append(float(((base+correction)[held]-torch.tensor(truth[j, held], device=device)).square().mean()))
                if j % 48 == 0:
                    singular = torch.linalg.svdvals(design.double())
                    ranks.append(observability(design)['rank'])
                    singular_rows.append(singular.cpu().numpy())
                    generator = torch.Generator(device=device).manual_seed(421000+fold*1000+j)
                    perturbation = .01*torch.randn(len(design), generator=generator, device=device, dtype=torch.float64)/scale
                    perturbation_c, _ = fit_latent(design, perturbation, curvature, penalty=.01,
                                                  curvature_weight=config['curvature_weight'])
                    amplification = scale*(h.double()@perturbation_c)
                    sensitivity.append(float(amplification[valid[j]].square().mean().sqrt()/.01))
                if (j+1) % 48 == 0:
                    emit(f'fold {fold}: {j+1}/{len(hours)} inversions')
        assert max_equivalence_error <= .001, max_equivalence_error
        kwargs = dict(metadata=metadata, safe_T=fine_sites)
        base_metrics = evaluate_arrays(hours, ids, truth, baseline, valid, valid,
                                       coarsened_prediction=baseline_coarse, **kwargs)
        rows = []
        for k, (penalty, alpha) in enumerate(combinations):
            metrics = evaluate_arrays(hours, ids, truth, predictions[k], valid, valid,
                                      coarsened_prediction=coarse_predictions[k], **kwargs)
            rows.append(dict(penalty=penalty, alpha=alpha, **summarize(metrics)))
        reference = rows[combinations.index((.01, 1.))]
        expected_summary = summarize(expected['metrics'])
        reproduction_error = metric_delta(reference, expected_summary)
        assert reproduction_error <= .001, (fold, reference, expected_summary)
        base_diff = abs(summarize(base_metrics)['rmse_mm']-expected['baseline']['fine']['model']['station_macro_rmse_mm'])
        assert base_diff <= .001, base_diff
        result = dict(fold=fold, rows=rows, baseline=summarize(base_metrics),
                      server_reproduction_max_metric_error_mm=reproduction_error,
                      baseline_reproduction_error_mm=base_diff,
                      query_equivalence_max_error_mm=max_equivalence_error,
                      fitted_hour_macro_rmse_mm=float(np.sqrt(np.mean(fit_error))),
                      held_hour_macro_rmse_mm=float(np.sqrt(np.mean(held_error))),
                      sampled_ranks=ranks, sensitivity_amplification=sensitivity,
                      mean_singular_values=np.mean(singular_rows, axis=0).tolist())
        write_json(ROOT/f'fold_{fold}_summary.json', result)
        np.savez_compressed(ROOT/f'fold_{fold}_predictions.npz', hours=hours, station_ids=ids,
                            truth=truth, valid=valid, baseline=baseline, predictions=predictions,
                            combinations=np.array(combinations))
        summaries.append(result)
        emit(f'fold {fold}: reproduced within {reproduction_error:.6f} mm; M0={result["baseline"]["rmse_mm"]:.3f}')
    write_json(ROOT/'summary.json', dict(folds=summaries, elapsed_seconds=time.monotonic()-started,
                                        status='COMPLETE', selection='Diagnostic sweep; outer scores are not tuning targets'))
    plot(summaries)
    emit(f'COMPLETE, elapsed={time.monotonic()-started:.1f}s')


def plot(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for fold, (summary, ax) in enumerate(zip(summaries, axes)):
        for penalty in (.01, .1, 1.):
            rows = [r for r in summary['rows'] if r['penalty'] == penalty]
            ax.plot([r['alpha'] for r in rows], [r['rmse_mm'] for r in rows], marker='o', label=f'Ridge {penalty:g}')
        ax.axhline(summary['baseline']['rmse_mm'], color='black', linestyle='--', label='Frozen M0')
        ax.set(title=f'Outer fold {fold+1}', xlabel='Residual amplitude', ylabel='Station-macro RMSE (mm)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT/'regularization_rmse.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for i, summary in enumerate(summaries):
        values = np.array(summary['mean_singular_values'])
        ax.semilogy(np.arange(1, len(values)+1), values/values[0], label=f'Fold {i+1}')
    ax.axhline(1e-6, color='gray', linestyle='--', label='Numerical rank threshold')
    ax.set(xlabel='Singular-value index', ylabel='Relative singular value')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'singular_spectrum.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
