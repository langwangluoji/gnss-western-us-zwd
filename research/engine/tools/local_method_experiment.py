"""A-only local field features, B-only selection, locked C development check."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.optimize import lsq_linear

from local_probe_diagnostics import ProbeData, PROJECT, emit, save_npz, score_arrays
from local_shared_field import LocalFieldFeatures, balanced_blocks, fit_shared_ridge, apply_shared_ridge
from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json
from gnss_field.geometry import surface_query, coarsen_100km
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.evaluation import point_metrics, paired_block_bootstrap

ROOT = PROJECT/'results/20261119_local_method_research_v01'
PENALTIES = (.01, .1, 1., 10.)
GROUPS = dict(blend=[1], innovation_two_scale=[2, 3], local_basic=list(range(4)), local_rich=list(range(22)))


def cache_features(data, year):
    path = ROOT/f'local_features_{year}.npz'
    source = fingerprint(dict(builder=sha256(Path(__file__).with_name('local_shared_field.py')),
                              condition=data.contract, scales_km=[75, 150], year=year))
    if path.exists():
        assert read_json(path.with_suffix('.json')) == dict(source=source, sha256=sha256(path))
        return dict(np.load(path))
    arrays = data.arrays(year)
    store, role = (data.store, 'B') if year == 2018 else (data.development, 'C')
    hours = arrays['hours']
    raw = store.context_values(hours)
    history = [store.context_values(hours-offset) for offset in (6, 12, 18)]
    build = LocalFieldFeatures(store, data.device)
    xy, height = safe_query_geometry(store, store.ids(role))
    fine, coarse = [], []
    maximum_idw_error, maximum_query_error = 0., 0.
    weights = torch.linspace(-.15, .15, 22, device=data.device)
    started = time.monotonic()
    with torch.no_grad():
        for i, hour in enumerate(hours):
            mean = data.frozen_field(hour)[0] if year == 2018 else arrays['mean_fields'][i]
            field, idw = build.fields(hour, mean, arrays['mean_A'][i], raw[i], [v[i] for v in history])
            q = surface_query(field, xy, height).T
            qc = surface_query(coarsen_100km(field), xy, height).T
            reconstructed_idw = surface_query(idw[None], xy, height)[0].cpu().numpy()
            maximum_idw_error = max(maximum_idw_error, float(abs(reconstructed_idw-arrays['idw'][i]).max()))
            if i % 48 == 0:
                direct = surface_query((field*weights[:, None, None]).sum(0)[None], xy, height)[0]
                maximum_query_error = max(maximum_query_error, float(abs(direct-q@weights).max()))
                emit(f'local feature cache {year}: {i+1}/{len(hours)}; elapsed {time.monotonic()-started:.1f}s')
            fine.append(q.cpu().numpy())
            coarse.append(qc.cpu().numpy())
    assert maximum_idw_error <= .001, maximum_idw_error
    assert maximum_query_error <= .001, maximum_query_error
    result = dict(features=np.array(fine), coarse_features=np.array(coarse), hours=hours)
    save_npz(path, **result)
    write_json(path.with_suffix('.json'), dict(source=source, sha256=sha256(path)))
    write_json(ROOT/f'feature_verification_{year}.json', dict(maximum_idw_error_mm=maximum_idw_error,
               maximum_grid_query_error_mm=maximum_query_error, feature_names=build.names,
               feature_source='A observations, frozen M0, fixed geometry; no target observations'))
    return result


def fit(name, x, residual, population, penalty):
    if name in ('blend', 'innovation_two_scale'):
        selected, y = x[population].astype('f8'), residual[population].astype('f8')
        state = lsq_linear(selected, y, bounds=(0., 1.), tol=1e-12)
        return dict(weights=state.x, penalty=0., station_hours=len(y))
    return fit_shared_ridge(x, residual, population, penalty)


def metric(truth, prediction, population):
    return point_metrics(truth, prediction, population)


def summarize_ci(value):
    return {k: v for k, v in value.items() if not k.startswith('replicate_')}


def main():
    torch.set_num_threads(2)
    np.random.seed(42)
    ROOT.mkdir(exist_ok=True, parents=True)
    data = ProbeData()
    arrays = data.arrays(2018)
    train, dev = balanced_blocks(arrays['hours'])
    lock = dict(source_sha256=sha256(__file__), builder_sha256=sha256(Path(__file__).with_name('local_shared_field.py')),
                foundation=data.contract, training_hours=arrays['hours'][train], development_hours=arrays['hours'][dev],
                penalties=PENALTIES, groups=GROUPS, seed=42,
                role='B2018 fits only, C2019 already-exposed development, D2020 sealed',
                outer='held physical B groups crossed with held whole weeks; M0 already saw B2016-17',
                selection='inner fitted-B held-week station macro RMSE, zero correction always retained')
    immutable_json(ROOT/'local_method.lock.json', lock)
    cached = cache_features(data, 2018)
    x, xc = cached['features'], cached['coarse_features']
    residual = arrays['truth']-arrays['baseline']
    rows, selected_penalties = [], {}
    for name, columns in GROUPS.items():
        candidates, reports = [], []
        for penalty in ((0.,) if name in ('blend', 'innovation_two_scale') else PENALTIES):
            values, fold_records = [], []
            for fold in range(3):
                fitted = data.assignment != fold
                train_population = arrays['valid'] & train[:, None] & fitted
                validation = arrays['valid'] & dev[:, None] & fitted
                outer = arrays['valid'] & dev[:, None] & ~fitted
                state = fit(name, x[..., columns], residual, train_population, penalty)
                prediction = arrays['baseline']+apply_shared_ridge(x[..., columns], state)
                inner_metric = metric(arrays['truth'], prediction, validation)
                values.append(inner_metric['station_macro_rmse_mm'])
                fold_records.append(dict(fold=fold, state=state, inner=inner_metric,
                    outer=metric(arrays['truth'], prediction, outer),
                    outer_M0=metric(arrays['truth'], arrays['baseline'], outer),
                    outer_A_IDW=metric(arrays['truth'], arrays['idw'], outer)))
            candidates.append(dict(penalty=penalty, mean_inner_rmse=float(np.mean(values)), folds=fold_records))
        selected = min(candidates, key=lambda v: v['mean_inner_rmse'])
        selected_penalties[name] = selected['penalty']
        rows.append(dict(model=name, candidates=candidates, selected=selected))
        emit(f'{name}: inner RMSE {selected["mean_inner_rmse"]:.4f}; outer '+
             ', '.join(f'{v["outer"]["station_macro_rmse_mm"]:.3f} / M0 {v["outer_M0"]["station_macro_rmse_mm"]:.3f} / IDW {v["outer_A_IDW"]["station_macro_rmse_mm"]:.3f}' for v in selected['folds']))
    write_json(ROOT/'spatiotemporal_cross_validation.json', dict(rows=rows,
               train_hours=int(train.sum()), dev_hours=int(dev.sum()), selected_penalties=selected_penalties))
    # Refit B-only after all inner selections, lock parameters before C evaluation.
    states = {name: fit(name, x[..., cols], residual, arrays['valid'], selected_penalties[name]) for name, cols in GROUPS.items()}
    immutable_json(ROOT/'B_only_parameters.lock.json', states)
    cached_c = cache_features(data, 2019)
    c = data.arrays(2019)
    results = {}
    for name, state in states.items():
        prediction = c['baseline']+apply_shared_ridge(cached_c['features'][..., GROUPS[name]], state)
        coarse = c['coarse']+apply_shared_ridge(cached_c['coarse_features'][..., GROUPS[name]], state)
        scored = score_arrays(data.development, 'C', c['hours'], c['truth'], prediction, c['valid'], coarse)
        intervals = {baseline: summarize_ci(paired_block_bootstrap(c['hours'], c['truth'], prediction,
                     c[key], c['valid'], replicates=2000)) for baseline, key in [('M0', 'baseline'), ('A_IDW', 'idw')]}
        results[name] = dict(metrics=scored, intervals=intervals, parameters=state)
        save_npz(ROOT/f'forecast_2019_{name}.npz', hours=c['hours'], station_ids=c['station_ids'], truth=c['truth'],
                 population=c['valid'], prediction=prediction, coarsened_prediction=coarse, baseline=c['baseline'], idw=c['idw'])
        emit(f'C2019 {name}: {scored["primary"]["station_macro_rmse_mm"]:.6f} mm; '+
             f'delta IDW {intervals["A_IDW"]["difference_mm"]:+.5f}')
    results['M0'] = dict(metrics=score_arrays(data.development, 'C', c['hours'], c['truth'], c['baseline'], c['valid'], c['coarse']))
    results['A_IDW'] = dict(metrics=score_arrays(data.development, 'C', c['hours'], c['truth'], c['idw'], c['valid']))
    write_json(ROOT/'results.json', results)
    emit('Local shared-field experiment complete; D2020 not accessed')


if __name__ == '__main__':
    main()
