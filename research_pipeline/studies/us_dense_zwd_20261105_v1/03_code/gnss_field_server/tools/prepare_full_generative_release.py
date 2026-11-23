"""Local immutable C-population and config preparation; never submits a job."""
from pathlib import Path
import io
import json

import numpy as np

from gnss_field.artifacts import atomic_bytes, fingerprint, sha256, write_json
from gnss_field.generative.campaign import build_data
from gnss_field.generative.evaluation import fixed_development_hours
from grid_pair_data import local_runtime


def main():
    project = Path(__file__).resolve().parents[1]
    output = project/'results/20261122_full_generative_design_v01'
    source = project/'results/20261119_frozen_m0_audit'
    assets = json.loads((output/'background_assets.local.json').read_text(encoding='utf-8'))
    calendar = np.arange(np.datetime64('2019-01-01').astype('datetime64[h]').astype('i8'),
                         np.datetime64('2020-01-01').astype('datetime64[h]').astype('i8'))
    config = dict(scientific=dict(seed=42,latent_dim=64,sigma_eff_mm=6.36,
        grid_shape=[144,96],lr=1e-4,minimum_lr=1e-6,reference_count=442.),
        training=dict(weight_decay=1e-4,ema_decay=.999,gradient_clip=1.,warmup_updates=500,
            formal_selection_start_step=201,
            evaluation_check_updates=[200],es_members=4,sampling_steps=18,
            late_decay_fraction=.7),
        data=dict(packed_root=local_runtime()['data_root'],
            cache_root=str(project/'build/full_generative_release_audit_cache'),
            background_source=assets,prepare_batch_hours=8,mapped_month_capacity=128),
        runtime=dict(run_root=str(project/'build/full_generative_release_audit'),device='cuda',
            project_root=str(project),python='D:/anaconda3-2025.12-2/envs/pytorch/python.exe',
            chunk_seconds=7200,backend_runtime=True,diagnostic_pilot_budget_seconds=1800,
            representation_check_budget_seconds=1800),
        diagnostics=dict(execute_pilots=True,oracle_steps=32),
        evaluation=dict(eval_members=16,point_members=64,evaluation_seed=20261123,point_seed=20261124,
            evaluation_chunk=4,evaluation_population_manifest=str(output/'population/C2019.lock.json'),
            development_hours=fixed_development_hours(calendar).tolist()),
        benchmark=dict(budget_seconds=3300),slurm=dict(gpu_partition='A800',cpu_partition='intel'))
    # Build only input statistics and geometry, never a network or background.
    data = build_data(config, purpose='development',allow_background_inference=False)
    old = json.loads((source/'station_prediction_manifest.json').read_text(encoding='utf-8'))
    ids = data.store.ids('C')
    if list(ids) != old['year2019_station_ids']:
        raise ValueError('C station order differs from previous locked population')
    monthly = {}; total = 0; counts = np.zeros(len(ids),'i8'); max_truth_error = 0.
    for month in range(1,13):
        tag = f'2019-{month:02d}'
        path = source/'monthly'/f'{tag}.npz'
        with np.load(path,allow_pickle=False) as archive:
            hours=archive['hours']; mask=archive['mask']; truth=archive['truth']
        raw = data.store.context_values(hours)
        values = data.store.targets('C',hours)
        current = np.isfinite(raw).all(-1)
        actual = np.isfinite(values) & np.stack([data._support(row)[1][ids] for row in current])
        if np.any(mask & ~actual):
            raise ValueError(f'New data geometry drops locked scores in {tag}: {np.sum(mask & ~actual)}')
        max_truth_error=max(max_truth_error,float(np.max(np.abs(values[mask]-truth[mask]))))
        target=output/'population'/f'{tag}.npz'
        buffer=io.BytesIO();np.savez_compressed(buffer,hours=hours,mask=mask)
        atomic_bytes(target,buffer.getvalue())
        monthly[tag]=dict(file=target.name,sha256=sha256(target),source_archive_sha256=sha256(path))
        counts+=mask.sum(0);total+=int(mask.sum())
        print('LOCKED_POPULATION',tag,int(mask.sum()),flush=True)
    if total != 642258 or int((counts>0).sum()) != 76 or max_truth_error > .001:
        raise ValueError('C population/truth reproduction failed')
    write_json(output/'population/C2019.lock.json',dict(schema=1,role='C',year=2019,
        monthly=monthly,station_ids=ids.tolist(),station_order_fingerprint=fingerprint(ids),
        valid_station_hours=total,scored_stations=int((counts>0).sum()),hours=calendar.tolist(),
        source_manifest_sha256=sha256(source/'station_prediction_manifest.json'),
        geometry_covers_all_locked_scores=True,truth_max_abs_error_mm=max_truth_error))
    write_json(output/'config.local.json',config)
    write_json(output/'population_acceptance.json',dict(passed=True,station_hours=total,
        stations=int((counts>0).sum()),hours=len(calendar),truth_max_abs_error_mm=max_truth_error,
        new_support_covers_original_population=True,D_labels_read=False))
    print('POPULATION_ACCEPTED',total,int((counts>0).sum()),flush=True)


if __name__ == '__main__': main()
