"""Locked inverse candidate on new hours from the same 2018 held station groups."""
import run_diagnosis as d
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import immutable_json, sha256, write_json
from gnss_field.data import PackedStore, ContextBuilder
from gnss_field.data.manifests import build_training_manifest, spatial_folds
from gnss_field.evaluation import evaluate_arrays
from gnss_field.geometry import surface_query, basis_observation_matrix, curvature_matrix, coarsen_100km
from gnss_field.models.basis import SpatialBasis
from gnss_field.models.mean import MeanField
from gnss_field.objectives import fit_latent, observability
from gnss_field.runtime import configure
from gnss_field.training.fields import predict_patches
from gnss_field.training.prediction import safe_query_geometry, safe_fine_sites


def main():
    started = time.monotonic()
    root, project = d.ROOT, d.PROJECT
    out = root/'confirmation'
    out.mkdir(exist_ok=True)
    configure(SimpleNamespace(cpus=2), 42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    store = PackedStore(project/'../../02_data/processed/gnss_field_server_v1', purpose='train')
    manifest = build_training_manifest(store, project/'local_validation/training_manifest.npz')
    assignment = spatial_folds(store, 20261117)
    old = np.array(json.loads((root/'run_config.json').read_text())['contract']['hours'])
    fold_hours = []
    for fold in range(3):
        allowed = np.setdiff1d(manifest.select([2018], 136, assignment != fold), old)
        dates = pd.to_datetime(allowed, unit='h')
        chosen = []
        for month in range(1, 13):
            for clock in range(24):
                subset = allowed[(dates.month == month) & (dates.hour == clock)]
                if len(subset):
                    chosen.append(subset[len(subset)//2])
        fold_hours.append(np.array(sorted(chosen), dtype='i8'))
    all_hours = np.unique(np.concatenate(fold_hours))
    assert not np.intersect1d(old, all_hours).size
    immutable_json(out/'candidate.lock.json', dict(penalty=1., alpha=.25, curvature=.001,
                   selection='chosen after the 576-hour diagnostic sweep',
                   confirmation='new hours, same held B station groups and already fitted bases',
                   fold_hours=[h.tolist() for h in fold_hours],
                   bases=[sha256(root/f'server/seed_42/basis_cv_{i}/last.pt') for i in range(3)]))
    d.emit(f'LOCKED ridge=1 alpha=.25; {len(all_hours)} new hours')
    cache = out/'mean.npy'
    support_cache = out/'support.npy'
    cursor_path = out/'cursor.json'
    if cache.exists():
        means = np.lib.format.open_memmap(cache, mode='r+')
        supports = np.lib.format.open_memmap(support_cache, mode='r+')
        done = json.loads(cursor_path.read_text())['done']
    else:
        means = np.lib.format.open_memmap(cache, mode='w+', dtype='f4', shape=(len(all_hours), 1, *store.shape))
        supports = np.lib.format.open_memmap(support_cache, mode='w+', dtype='bool', shape=(len(all_hours), *store.shape))
        done = 0
        write_json(cursor_path, dict(done=0))
    mean = MeanField().to(device).eval()
    path = project/'results/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/seed_42/mean_base/candidate_005000.pt'
    mean.load_state_dict(torch.load(path, map_location=device, weights_only=False)['ema'])
    builder = ContextBuilder(store)
    with torch.inference_mode():
        for j in range(done, len(all_hours)):
            context = builder.build_context(all_hours[j:j+1], device=device)
            field = predict_patches(mean, context, store, chunk=4)[0]
            assert torch.isfinite(field).all()
            means[j] = field[0].cpu().numpy()
            supports[j] = context.maps[0, 3].bool().cpu().numpy()
            if (j+1) % 24 == 0 or j+1 == len(all_hours):
                means.flush()
                supports.flush()
                write_json(cursor_path, dict(done=j+1))
                d.emit(f'confirmation M0 {j+1}/{len(all_hours)}')
    del mean, builder
    ids = store.ids('B')
    xy, heights = safe_query_geometry(store, ids)
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    kwargs = dict(metadata=metadata, safe_T=safe_fine_sites(store, ids))
    xyh = torch.tensor(np.c_[store.grid['x'].ravel(), store.grid['y'].ravel(), store.grid['height'].ravel()], device=device, dtype=torch.float32)
    land = torch.tensor(store.grid['region'].ravel(), device=device, dtype=torch.bool)
    summaries = []
    for fold, hours in enumerate(fold_hours):
        fit = assignment != fold
        state = torch.load(root/f'server/seed_42/basis_cv_{fold}/last.pt', map_location='cpu', weights_only=False)
        model = SpatialBasis(128, 256).to(device).eval()
        model.load_state_dict(state['ema'])
        truth = store.targets('B', hours)
        qualified = manifest.masks(hours)
        valid = qualified & ~fit
        baseline, prediction, coarse = (np.empty_like(truth) for _ in range(3))
        rank_cache, ranks = {}, []
        with torch.no_grad():
            phi = model(xyh, land).detach()
            h = basis_observation_matrix(phi, store.shape, xy, heights)
            hc = surface_query(coarsen_100km(phi.T.reshape(128, 1, *store.shape)), xy, heights)[:, 0].T
            for j, hour in enumerate(hours):
                index = np.searchsorted(all_hours, hour)
                field = torch.tensor(np.array(means[index:index+1]), device=device)
                support = torch.tensor(np.array(supports[index]), device=device)
                base = surface_query(field, xy, heights).flatten()
                base_coarse = surface_query(coarsen_100km(field), xy, heights).flatten()
                mask = qualified[j] & fit
                residual = (torch.tensor(truth[j, mask], device=device)-base[mask])/store.stats['z0_std']
                curvature = curvature_matrix(phi, store.shape, support)
                c, _ = fit_latent(h[mask], residual, curvature, penalty=1., curvature_weight=.001)
                c = c.float()
                baseline[j] = base.cpu().numpy()
                prediction[j] = (base+.25*store.stats['z0_std']*(h@c)).cpu().numpy()
                coarse[j] = (base_coarse+.25*store.stats['z0_std']*(hc@c)).cpu().numpy()
                key = mask.tobytes()
                if key not in rank_cache:
                    rank_cache[key] = observability(h[mask])['rank']
                ranks.append(rank_cache[key])
        metrics = evaluate_arrays(hours, ids, truth, prediction, valid, valid, coarsened_prediction=coarse, **kwargs)
        base_metrics = evaluate_arrays(hours, ids, truth, baseline, valid, valid, coarsened_prediction=baseline, **kwargs)
        m, b = d.summarize(metrics), d.summarize(base_metrics)
        passed = (m['rmse_mm'] < min(b['rmse_mm'], m['coarse_rmse_mm']) and
                  m['increment_rmse_mm'] < min(b['increment_rmse_mm'], m['coarse_increment_rmse_mm']) and
                  np.mean(np.array(ranks) >= 64) >= .95)
        summaries.append(dict(fold=fold, hours=len(hours), baseline=b, candidate=m, directional_gate=bool(passed)))
        np.savez_compressed(out/f'fold_{fold}_predictions.npz', hours=hours, station_ids=ids, truth=truth,
                            valid=valid, baseline=baseline, prediction=prediction, coarse_prediction=coarse)
        d.emit(f'confirmation fold {fold}: M0 {b["rmse_mm"]:.4f}, candidate {m["rmse_mm"]:.4f}, gate {passed}')
    write_json(out/'summary.json', dict(folds=summaries, status='COMPLETE',
               elapsed_seconds=time.monotonic()-started, direction_pass=all(r['directional_gate'] for r in summaries),
               interpretation='Post-exploration temporal confirmation within 2018 B; same physical groups and frozen bases.'))


if __name__ == '__main__':
    main()
