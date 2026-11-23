"""Compare final raw/EMA bases using the same frozen local mean subset."""
import run_diagnosis as diagnostic
from pathlib import Path
import json
import time

import numpy as np
import torch

from gnss_field.artifacts import write_json
from gnss_field.data import PackedStore
from gnss_field.data.manifests import build_training_manifest, spatial_folds, stratified_hours
from gnss_field.evaluation import evaluate_arrays
from gnss_field.geometry import basis_observation_matrix, surface_query, coarsen_100km, curvature_matrix
from gnss_field.models.basis import SpatialBasis
from gnss_field.objectives import fit_latent
from gnss_field.runtime import configure
from gnss_field.training.prediction import safe_query_geometry, safe_fine_sites
from types import SimpleNamespace


def main():
    root, project = diagnostic.ROOT, diagnostic.PROJECT
    started = time.monotonic()
    configure(SimpleNamespace(cpus=2), 42)
    store = PackedStore(project/'../../02_data/processed/gnss_field_server_v1', purpose='train')
    manifest = build_training_manifest(store, project/'local_validation/training_manifest.npz')
    settings = json.loads((root/'server/resolved.json').read_text())['science']
    record = json.loads((root/'run_config.json').read_text())
    device = record['device']
    hours_all = np.array(record['contract']['hours'])
    means = np.load(root/'mean_subset.npy', mmap_mode='r')
    supports = np.load(root/'support_subset.npy', mmap_mode='r')
    ids = store.ids('B')
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    xy, heights = safe_query_geometry(store, ids)
    fine_sites = safe_fine_sites(store, ids)
    assignments = spatial_folds(store, settings['study_seed'])
    xyh = torch.tensor(np.c_[store.grid['x'].ravel(), store.grid['y'].ravel(), store.grid['height'].ravel()], device=device, dtype=torch.float32)
    land = torch.tensor(store.grid['region'].ravel(), device=device, dtype=torch.bool)
    scale = store.stats['z0_std']
    results = []
    for fold in range(3):
        state = torch.load(root/f'server/seed_42/basis_cv_{fold}/last.pt', map_location='cpu', weights_only=False)
        fit = assignments != fold
        hours = stratified_hours(manifest.select([2018], settings['rank']+8, fit), 2018)
        truth = store.targets('B', hours)
        qualified = manifest.masks(hours)
        valid = qualified & ~fit
        model = SpatialBasis(settings['rank'], settings['basis_width']).to(device).eval()
        model.load_state_dict(state['model'])
        prediction = np.empty_like(truth)
        coarse_prediction = np.empty_like(truth)
        with torch.no_grad():
            phi = model(xyh, land).detach()
            h = basis_observation_matrix(phi, store.shape, xy, heights)
            coarse_h = surface_query(coarsen_100km(phi.T.reshape(settings['rank'], 1, *store.shape)), xy, heights)[:, 0].T
            for j, hour in enumerate(hours):
                index = int(np.searchsorted(hours_all, hour))
                field = torch.tensor(np.array(means[index:index+1]), device=device)
                support = torch.tensor(np.array(supports[index]), device=device)
                base = surface_query(field, xy, heights).flatten()
                base_coarse = surface_query(coarsen_100km(field), xy, heights).flatten()
                mask = qualified[j] & fit
                residual = (torch.tensor(truth[j, mask], device=device)-base[mask])/scale
                curvature = curvature_matrix(phi, store.shape, support)
                coefficient, _ = fit_latent(h[mask], residual, curvature, penalty=.01, curvature_weight=settings['curvature_weight'])
                prediction[j] = (base+scale*(h@coefficient.float())).cpu().numpy()
                coarse_prediction[j] = (base_coarse+scale*(coarse_h@coefficient.float())).cpu().numpy()
                if (j+1) % 96 == 0:
                    diagnostic.emit(f'raw fold {fold}: {j+1}/{len(hours)}')
        bundle = evaluate_arrays(hours, ids, truth, prediction, valid, valid,
                                 coarsened_prediction=coarse_prediction, metadata=metadata, safe_T=fine_sites)
        result = dict(fold=fold, raw=diagnostic.summarize(bundle))
        z = np.load(root/f'fold_{fold}_predictions.npz')
        selected = np.flatnonzero((z['combinations'][:, 0] == .01) & (z['combinations'][:, 1] == 1.))[0]
        mask = z['valid'] & fine_sites[None, :]
        residual = (z['truth']-z['baseline'])[mask].astype('f8')
        correction = (z['predictions'][selected]-z['baseline'])[mask].astype('f8')
        result['diagnostic_micro_optimal_alpha'] = float(np.dot(residual, correction)/np.dot(correction, correction))
        result['correction_residual_correlation'] = float(np.corrcoef(residual, correction)[0, 1])
        result['ema'] = json.loads((root/f'fold_{fold}_summary.json').read_text())
        # Each hour contributes one equally weighted MSE in the inversion logger.
        for prefix in ('fitted', 'held'):
            if f'{prefix}_micro_rmse_mm' in result['ema']:
                value = result['ema'].pop(f'{prefix}_micro_rmse_mm')
                result['ema'][f'{prefix}_hour_macro_rmse_mm'] = value
        write_json(root/f'fold_{fold}_summary.json', result['ema'])
        results.append(result)
    write_json(root/'raw_comparison.json', dict(folds=results, elapsed_seconds=time.monotonic()-started))
    summary = json.loads((root/'summary.json').read_text())
    summary['folds'] = [row['ema'] for row in results]
    write_json(root/'summary.json', summary)
    report(results, root, metadata)
    diagnostic.emit('Raw/EMA comparison COMPLETE')


def report(results, root, metadata):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with (root/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fold', 'M0_RMSE_mm', 'EMA_ridge_0.01_RMSE_mm',
                         'EMA_ridge_0.1_RMSE_mm', 'EMA_ridge_1_RMSE_mm',
                         'raw_ridge_0.01_RMSE_mm', 'EMA_fit_hour_RMSE_mm',
                         'EMA_held_hour_RMSE_mm', 'diagnostic_micro_alpha'])
        for result in results:
            ema = result['ema']
            rows = [next(r for r in ema['rows'] if r['penalty'] == p and r['alpha'] == 1.)
                    for p in (.01, .1, 1.)]
            writer.writerow([result['fold'], ema['baseline']['rmse_mm'],
                             *[r['rmse_mm'] for r in rows], result['raw']['rmse_mm'],
                             ema['fitted_hour_macro_rmse_mm'], ema['held_hour_macro_rmse_mm'],
                             result['diagnostic_micro_optimal_alpha']])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for result, ax in zip(results, axes):
        z = np.load(root/f'fold_{result["fold"]}_predictions.npz')
        k = np.flatnonzero((z['combinations'][:, 0] == .01) & (z['combinations'][:, 1] == 1.))[0]
        valid = z['valid']
        count = valid.sum(0)
        base_error = np.where(valid, (z['baseline']-z['truth'])**2, 0).sum(0)
        model_error = np.where(valid, (z['predictions'][k]-z['truth'])**2, 0).sum(0)
        delta = np.sqrt(model_error/np.maximum(count, 1))-np.sqrt(base_error/np.maximum(count, 1))
        select = count > 0
        artist = ax.scatter(metadata.x[select]/1000, metadata.y[select]/1000,
                            c=delta[select], cmap='RdBu_r', vmin=-8, vmax=8, s=16)
        ax.set(title=f'Outer fold {result["fold"]+1}', xlabel='Lambert x (km)', ylabel='Lambert y (km)')
        ax.set_aspect('equal')
    fig.colorbar(artist, ax=axes, label='Station RMSE change vs M0 (mm)')
    fig.savefig(root/'held_station_error_map.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
