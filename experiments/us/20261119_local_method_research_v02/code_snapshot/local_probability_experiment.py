"""Three-parameter coherent field diagnostic with a marginal-matched shuffle."""
import numpy as np
import torch
from scipy.spatial.distance import cdist

from local_probe_diagnostics import ProbeData, PROJECT, emit, save_npz
from local_shared_field import nested_blocks
from local_joint_field import (local_bases, observation_factors, group_records, fit_covariance,
                               covariance_objective, sample_factors, shuffle_members)
from gnss_field.artifacts import immutable_json, write_json, sha256
from gnss_field.evaluation import evaluate_arrays, energy_score_ensemble, variogram_score
from gnss_field.evaluation.bootstrap import _blocks, _draw_blocks, _summary
from gnss_field.training.prediction import safe_fine_sites

ROOT = PROJECT/'results/20261119_local_joint_probability_v01'


def joint_series(truth, samples, population, metadata):
    groups = metadata.dense_group.to_numpy()
    coordinates = metadata[['x', 'y']].to_numpy()
    rows, names = [], []
    for group in dict.fromkeys(groups):
        selection = np.flatnonzero(groups == group)
        if len(selection) < 4:
            continue
        output = np.full((len(truth), 2), np.nan)
        for t in range(len(truth)):
            ids = selection[population[t, selection]]
            if len(ids) < 4:
                continue
            output[t] = [energy_score_ensemble(truth[t, ids], samples[t][:, ids]),
                         variogram_score(truth[t, ids], samples[t][:, ids], cdist(coordinates[ids], coordinates[ids])/1000)]
        # Preserve the station-group axis even for months without valid hours.
        rows.append(output)
        names.append(str(group))
    return np.stack(rows, axis=1), names


def joint_intervals(hours, model, baseline):
    active = np.isfinite(model[..., 0]).any(0)
    if not np.array_equal(active, np.isfinite(baseline[..., 0]).any(0)):
        raise ValueError('Joint populations differ between products')
    model, baseline = model[:, active], baseline[:, active]
    blocks = _blocks(hours, 7)
    rng = np.random.default_rng(20261117)
    def statistic(values):
        return np.nanmean(np.nanmean(values, axis=0), axis=0)
    actual = statistic(model)-statistic(baseline)
    scores = []
    for _ in range(2000):
        idx = _draw_blocks(blocks, rng)
        if not np.isfinite(model[idx]).any(0).all():
            raise ValueError('Bootstrap omitted an active group')
        scores.append(statistic(model[idx])-statistic(baseline[idx]))
    return {name: {k: v for k, v in _summary(actual[j], np.array(scores)[:, j]).items()
                   if not k.startswith('replicate_')} for j, name in enumerate(('ES', 'VS'))}


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True, exist_ok=True)
    data = ProbeData(device='cpu')
    b = data.arrays(2018)
    core, inner, outer = nested_blocks(b['hours'])
    bases = local_bases(data.store)
    bh = observation_factors(data.store, data.store.ids('B'), bases)
    immutable_json(ROOT/'protocol.lock.json', dict(source=sha256(__file__), model_source=sha256(PROJECT/'tools/local_joint_field.py'),
        mean='frozen M0 predictive mean, never fitted to C', basis='fixed compact bisquare, spacings100/300km, radii150/450km, pointwise L2 normalized',
        rank=[v.shape[1] for v in bases], train_hours=b['hours'][~outer], outer_hours=b['hours'][outer],
        fitting='B2018 composite Gaussian NLL on fixed200km groups>=4; three nonnegative variances',
        observation='Z0 coherent local factors queried at surface plus independent surface observation noise',
        comparison='correlated ensemble versus exact per-station member permutations; identical empirical marginals',
        members=[64, 128], seed=42, D2020='sealed'))
    residual = b['truth']-b['baseline']
    folds = []
    for fold in range(3):
        population = b['valid'] & (~outer[:, None]) & (data.assignment != fold)
        records, count = group_records(data.store, b['station_ids'], bh, residual, population)
        state = fit_covariance(records)
        held = b['valid'] & outer[:, None] & (data.assignment == fold)
        validation, validation_count = group_records(data.store, b['station_ids'], bh, residual, held)
        state['held_nll'] = covariance_objective(np.log(state['std']), validation)[0]
        folds.append(dict(fold=fold, fit=state, training=count, held=validation_count))
        emit(f'Gaussian local factors fold {fold}: std={state["std"]}; held NLL={state["held_nll"]:.5f}')
    records, count = group_records(data.store, b['station_ids'], bh, residual, b['valid'])
    final = fit_covariance(records)
    immutable_json(ROOT/'B_only_parameters.lock.json', dict(model=final, training=count, folds=folds))
    c = data.arrays(2019)
    store, ids = data.development, data.development.ids('C')
    factors = observation_factors(store, ids, bases)
    noise = sample_factors(factors, final['std'], c['hours'], ids, 128, 42)
    samples = c['baseline'][:, None]+noise
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    full = np.isfinite(c['truth']) & store.query_valid[ids]
    results = {}
    for k in (64, 128):
        correlated = samples[:, :k]
        shuffled = shuffle_members(correlated, c['hours'], ids, 42)
        metrics = {}
        series = {}
        for name, members in [('correlated', correlated), ('shuffled', shuffled)]:
            metrics[name] = evaluate_arrays(c['hours'], ids, c['truth'], c['baseline'], full, c['valid'],
                samples=members, metadata=metadata)
            series[name], groups = joint_series(c['truth'], members, c['valid'], metadata)
        a, b_metric = metrics['correlated']['primary'], metrics['shuffled']['primary']
        assert abs(a['station_macro_crps_mm']-b_metric['station_macro_crps_mm']) < 1e-12
        assert a['intervals'] == b_metric['intervals']
        intervals = joint_intervals(c['hours'], series['correlated'], series['shuffled'])
        results[str(k)] = dict(metrics=metrics, paired_intervals=intervals, groups=groups,
                               finite_ensemble_mean_rmse_note='Point metrics use the fixed analytic predictive mean M0')
        save_npz(ROOT/f'joint_scores_K{k}.npz', hours=c['hours'], correlated=series['correlated'], shuffled=series['shuffled'])
        emit(f'C2019 K{k}: '+str({name: result['joint'] for name, result in metrics.items()}))
        emit(f'C2019 K{k} paired: {intervals}')
    save_npz(ROOT/'forecast_C2019_K128.npz', hours=c['hours'], station_ids=ids, truth=c['truth'],
             population=c['valid'], analytic_mean=c['baseline'], samples=samples)
    write_json(ROOT/'results.json', dict(final=final, folds=folds, results=results,
        interpretation='Exploratory C2019; sample shuffling isolates dependence with exactly matched marginals; no claim of observed grid truth'))
    emit('Joint probability diagnostic complete; D2020 not read')


if __name__ == '__main__':
    main()
