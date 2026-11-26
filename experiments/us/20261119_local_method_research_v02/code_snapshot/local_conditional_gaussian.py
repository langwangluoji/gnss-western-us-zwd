"""One weather-conditioned scale on the frozen coherent Gaussian baseline."""
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from local_probe_diagnostics import PROJECT, ProbeData, emit, save_npz
from local_joint_field import local_bases, observation_factors
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256
from gnss_field.evaluation import evaluate_arrays

ROOT = PROJECT/'results/20261119_local_conditional_probability_v01'
BASE = PROJECT/'results/20261119_local_fused_probability_v02/full2019'


def scale_objective(parameters, data):
    q, dimension, feature = data.T
    x = np.c_[np.ones(len(feature)), feature]
    log_scale = x@parameters
    scaled_q = q*np.exp(-2*log_scale)
    objective = (dimension*log_scale+.5*scaled_q).sum()/dimension.sum()
    gradient = x.T@(dimension-scaled_q)/dimension.sum()
    return float(objective), gradient


def wet_feature(raw):
    valid = np.isfinite(raw).all(-1)
    # Empty-input hours remain declared and use a registered neutral feature.
    value = np.zeros(len(raw), dtype='f8')
    present = valid.any(-1)
    value[present] = np.nanmedian(np.where(valid[present], raw[present, :, 0], np.nan), 1)
    return value, present


def quadratic_features(store, ids, factors, std, residual, population, features):
    _, assignment = np.unique(np.floor(store.xy[ids]/200000).astype(int), axis=0, return_inverse=True)
    rows = []
    for group in np.unique(assignment):
        selected = np.flatnonzero(assignment == group)
        if len(selected) < 4:
            continue
        patterns, inverse = np.unique(population[:, selected], axis=0, return_inverse=True)
        for j, pattern in enumerate(patterns):
            use = selected[pattern]
            if len(use) < 4:
                continue
            e = residual[inverse == j][:, use].astype('f8')
            cov = sum(s*s*(h[use]@h[use].T) for s, h in zip(std, factors))+std[-1]**2*np.eye(len(use))
            cov.flat[::len(use)+1] += 1e-8
            factor = cho_factor(cov, lower=True)
            q = np.einsum('ti,it->t', e, cho_solve(factor, e.T))
            rows.append(np.c_[q, np.full(len(q), len(use)), features[inverse == j]])
    return np.concatenate(rows)


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True, parents=True)
    inputs = sorted(BASE.glob('forecast_2019-*.npz'))
    immutable_json(ROOT/'protocol.lock.json', dict(source=sha256(__file__),
        baseline_parameters=sha256(BASE.parent/'B_only_parameters.lock.json'),
        baseline_forecasts={p.name: sha256(p) for p in inputs},
        hypothesis='A-derived wetness predicts residual amplitude; all spatial and unresolved terms share g(A)',
        model='log g=a+b*(A median Z0 - B-training-hours center)/std',
        parameters=2, bounds=[[-1, 1], [-1, 1]], members=128, seed=42,
        fit='B2018 composite NLL; C used only after parameter lock', D2020='sealed',
        comparison='same mean and spatial correlation; constant versus A-dependent amplitude'))
    data = ProbeData(device='cpu')
    b = data.arrays(2018)
    locked = read_json(BASE.parent/'B_only_parameters.lock.json')
    std, weight = np.array(locked['model']['std']), locked['weight']
    wet, available = wet_feature(data.store.context_values(b['hours']))
    assert available.all()
    center, deviation = wet.mean(), wet.std()
    feature = (wet-center)/deviation
    bases = local_bases(data.store)
    factors = observation_factors(data.store, b['station_ids'], bases)
    mean = b['baseline']+weight*(b['idw']-b['baseline'])
    summaries = quadratic_features(data.store, b['station_ids'], factors, std, b['truth']-mean, b['valid'], feature)
    result = minimize(scale_objective, np.zeros(2), args=(summaries,), jac=True,
                      method='L-BFGS-B', bounds=[(-1., 1.)]*2, options=dict(maxiter=100, ftol=1e-12))
    if not result.success:
        raise RuntimeError(result.message)
    state = dict(a=float(result.x[0]), b=float(result.x[1]), A_wetness_center=float(center), A_wetness_std=float(deviation),
        objective=result.fun, initial_objective=scale_objective(np.zeros(2), summaries)[0], iterations=result.nit,
        group_hours=len(summaries), station_hours=int(summaries[:, 1].sum()))
    immutable_json(ROOT/'B_only_parameters.lock.json', state)
    emit(f'B-only conditional Gaussian {state}')
    store, ids = data.development, data.development.ids('C')
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    ch = observation_factors(store, ids, bases)
    sigma = np.sqrt(sum(s*s*np.sum(h*h, axis=1) for s, h in zip(std, ch))+std[-1]**2)
    series = {name: [] for name in ('conditional', 'constant')}
    values = {name: [] for name in ('hours', 'truth', 'mean', 'population', 'F0', 'std')}
    for path in inputs:
        z = dict(np.load(path))
        wet, present = wet_feature(store.context_values(z['hours']))
        x = np.where(present, (wet-center)/deviation, 0.)
        multiplier = np.exp(state['a']+state['b']*x)
        samples = z['analytic_mean'][:, None]+(z['samples'].astype('f8')-z['analytic_mean'][:, None])*multiplier[:, None, None]
        for name, members in [('conditional', samples), ('constant', z['samples'])]:
            output, names = joint_series(z['truth'], members, z['population'], metadata)
            series[name].append(output)
        m0 = dict(np.load(PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'/path.name))
        np.testing.assert_array_equal(m0['hours'], z['hours'])
        for name, key in [('hours', 'hours'), ('truth', 'truth'), ('mean', 'analytic_mean'), ('population', 'population')]:
            values[name].append(z[key])
        values['F0'].append(m0['population_F'])
        values['std'].append(multiplier[:, None]*sigma[None])
        save_npz(ROOT/path.name, hours=z['hours'], station_ids=ids, analytic_mean=z['analytic_mean'],
                 samples=samples.astype('f4'), population=z['population'], std=values['std'][-1], multiplier=multiplier)
        emit(f'Conditional full2019 {path.stem[-7:]}')
    series = {k: np.concatenate(v) for k, v in series.items()}
    active = np.isfinite(series['conditional'][..., 0]).any(0)
    series = {k: v[:, active] for k, v in series.items()}
    values = {k: np.concatenate(v) for k, v in values.items()}
    assert len(values['hours']) == 8760 and values['population'].sum() == 642258
    metrics = evaluate_arrays(values['hours'], ids, values['truth'], values['mean'], values['F0'], values['population'],
                              std=values['std'], metadata=metadata)
    joint = {k: dict(ES_mm=float(np.nanmean(np.nanmean(v[..., 0], 0))), VS_mm=float(np.nanmean(np.nanmean(v[..., 1], 0)))) for k, v in series.items()}
    difference = joint_intervals(values['hours'], series['conditional'], series['constant'])
    save_npz(ROOT/'joint_scores.npz', hours=values['hours'], **series)
    write_json(ROOT/'results.json', dict(parameters=state, marginal=metrics, joint=joint, paired_intervals=difference,
        inference='B-fitted, post-diagnostic candidate evaluated on already exposed full C2019; D2020 sealed'))
    emit(f'Conditional full2019 CRPS {metrics["primary"]["station_macro_crps_mm"]}; intervals {metrics["primary"]["intervals"]}; joint {joint}; difference {difference}')


if __name__ == '__main__':
    main()
