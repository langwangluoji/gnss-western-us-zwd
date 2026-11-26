"""Build an immutable B2018 coverage bundle without recomputing the mean model."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from local_subspace_data import ROOT, PilotData, transform_condition
from local_probe_diagnostics import PROJECT, emit, save_npz, MEAN
from local_conditional_gaussian import wet_feature
from gnss_field.artifacts import read_json, write_json, sha256, immutable_json
from gnss_field.data import ContextBuilder
from gnss_field.data.context import cycles
from gnss_field.training.baselines import get_baseline_predictor
from gnss_field.training.observable import coverage_split

DESTINATION = ROOT/'server_bundle'


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(exist_ok=True)
    data = PilotData(device='cpu')
    b = data.arrays(2018)
    paths = sorted((PROJECT/'results/20261119_frozen_m0_audit/monthly').glob('2018-*.npz'))
    assert len(paths) == 12
    records = [dict(np.load(path)) for path in paths]
    hours = np.concatenate([r['hours'] for r in records])
    mean = np.concatenate([r['mean'] for r in records])
    truth = np.concatenate([r['truth'] for r in records])
    valid = data.data.manifest.masks(hours)
    np.testing.assert_array_equal(valid, np.concatenate([r['mask'] for r in records]))
    old = data.data.arrays(2018)
    old_index = np.searchsorted(hours, old['hours'])
    error = abs(mean[old_index]-old['baseline']).max()
    assert error <= .001
    np.testing.assert_allclose(truth[old_index], old['truth'], rtol=0, atol=.001, equal_nan=True)
    mask = coverage_split(hours, valid.sum(1) >= 4, old['hours'][data.development])
    maximum = 2*int(np.ceil(mask.sum()/8))
    immutable_json(DESTINATION/'preparation.lock.json', dict(source=sha256(__file__), source_contract=data.contract,
        old_mean_weights_sha256=sha256(MEAN), cached_backgrounds={p.name: sha256(p) for p in paths},
        training_hours=hours[mask], development_anchors=old['hours'][data.development],
        maximum_updates_per_arm=maximum, policy='complete locked nominal weeks excluded with +/-18h buffers',
        target='B2018 surface ZWD mm', D2020='sealed'))
    cache = DESTINATION/'A_conditions_and_IDW.npz'
    if cache.exists():
        z = dict(np.load(cache))
        condition, idw = z['condition'], z['idw']
    else:
        builder = ContextBuilder(data.data.store)
        predictor = get_baseline_predictor('A_IDW', data.data.store)
        condition, idw = [], []
        started = time.monotonic()
        for start in range(0, len(hours), 8):
            selected = hours[start:start+8]
            context = builder.build_context(selected)
            feature = torch.cat([context.stations[:, :, :10].flatten(1),
                F.adaptive_avg_pool2d(context.coarse.flatten(1, 2), (4, 4)).flatten(1),
                torch.tensor(cycles(selected))], -1).numpy()
            condition.append(transform_condition(feature, data.pca))
            idw.append(predictor(selected, context, old['station_ids'], 1)['mean'])
            if start % 720 == 0:
                emit(f'Prepare fullB A-only conditions and IDW {start}/{len(hours)}; {time.monotonic()-started:.1f}s')
        condition, idw = np.concatenate(condition), np.concatenate(idw)
        save_npz(cache, hours=hours, condition=condition, idw=idw)
    assert abs(idw[old_index]-old['idw']).max() <= .001
    assert abs(condition[old_index]-b['condition'].cpu().numpy()).max() <= 1e-4
    wet, present = wet_feature(data.data.store.context_values(hours))
    g = data.amplitude
    amplitude = np.exp(g['a']+g['b']*np.where(present, (wet-g['A_wetness_center'])/g['A_wetness_std'], 0.))
    save_npz(DESTINATION/'data.npz', hours=hours, station_ids=old['station_ids'],
        mean=mean+data.weight*(idw-mean), truth=truth, valid=valid, condition=condition,
        amplitude=amplitude.astype('f4'), projection=b['projection'].cpu().numpy(), remainder=b['remainder'].cpu().numpy(),
        training_indices=np.flatnonzero(mask), development_indices=np.searchsorted(hours, old['hours'][data.development]))
    # Publication factors are separate from labels and training arrays.
    factor = np.concatenate([s*phi for s, phi in zip(data.sigma, data.bases)], axis=1).astype('f4')
    save_npz(DESTINATION/'spatial.npz', phi=factor, v=data.v.astype('f4'), shape=np.array(data.data.store.shape),
             land=data.data.store.grid['region'], x=data.data.store.grid['x'], y=data.data.store.grid['y'])
    immutable_json(DESTINATION/'manifest.json', dict(role='B', year=2018, data_sha256=sha256(DESTINATION/'data.npz'),
        spatial_sha256=sha256(DESTINATION/'spatial.npz'), frozen_contract=data.contract,
        tau_surface_mm=float(data.sigma[-1]), training_hours=int(mask.sum()), development_hours=288,
        effective_hours=8, maximum_updates_per_arm=maximum, seed=42,
        note='Incremental coverage experiment; frozen baseline statistics previously used development B weather',
        M0_unchanged=sha256(MEAN) == data.data.mean_hash, all_data_B2018=True, D2020_accessed=False))
    write_json(DESTINATION/'validation.json', dict(passed=True, maximum_old_background_error_mm=float(error),
        original_mask_exact=True, full_train_hours=int(mask.sum()), updates_per_arm=maximum,
        data_size_bytes=(DESTINATION/'data.npz').stat().st_size, spatial_size_bytes=(DESTINATION/'spatial.npz').stat().st_size))
    emit(f'Server bundle ready: {mask.sum()} B training hours, {maximum} maximum updates per arm')


if __name__ == '__main__':
    main()
