"""Exploratory paired intervals on already inspected confirmation predictions."""
import run_diagnosis as d
import numpy as np

from gnss_field.artifacts import write_json
from gnss_field.data import PackedStore
from gnss_field.evaluation.bootstrap import paired_block_bootstrap, crossed_increment_bootstrap
from gnss_field.evaluation.metrics import pair_increment_arrays
from gnss_field.training.prediction import safe_fine_sites


def compact(result):
    return {key: value for key, value in result.items() if not key.startswith('replicate_differences')}


def main():
    store = PackedStore(d.PROJECT/'../../02_data/processed/gnss_field_server_v1', purpose='train')
    ids = store.ids('B')
    safe = safe_fine_sites(store, ids)
    records = []
    for fold in range(3):
        z = np.load(d.ROOT/f'confirmation/fold_{fold}_predictions.npz')
        valid = z['valid'] & safe[None, :]
        sites = valid.any(0)
        metadata = store.stations.iloc[ids[sites]].reset_index(drop=True)
        truth, model = z['truth'][:, sites], z['prediction'][:, sites]
        pop = valid[:, sites]
        row = dict(fold=fold)
        for name, reference in (('M0', z['baseline']), ('coarsened', z['coarse_prediction'])):
            baseline = reference[:, sites]
            row[f'point_vs_{name}'] = compact(paired_block_bootstrap(
                z['hours'], truth, model, baseline, pop, replicates=2000, block_days=7))
            pairs = pair_increment_arrays(truth, model, pop, metadata)
            other = pair_increment_arrays(truth, baseline, pop, metadata)
            selected = pairs['distance_km'] <= 50
            row[f'increment_vs_{name}'] = compact(crossed_increment_bootstrap(
                z['hours'], pairs['truth'][:, selected], pairs['prediction'][:, selected],
                other['prediction'][:, selected], pairs['population'][:, selected],
                pairs['group_ids'][selected], replicates=2000, block_days=7))
        records.append(row)
        d.emit(f'Paired diagnostic fold {fold}: {row}')
    write_json(d.ROOT/'paired_diagnostics.json', dict(
        folds=records, selection='Exploratory post-selection diagnostic; same 2018 B physical groups',
        formal_confirmation=False))


if __name__ == '__main__':
    main()
