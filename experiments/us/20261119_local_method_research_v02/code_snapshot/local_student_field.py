"""One predeclared heavy-tail control, fitted only to training B residuals."""
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import gammaln, betaln
from scipy.stats import t
from local_probe_diagnostics import PROJECT, ProbeData, emit, save_npz
from local_joint_field import local_bases, observation_factors
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256

ROOT = PROJECT/'results/20261119_local_student_probability_v01'
GAUSSIAN = PROJECT/'results/20261119_local_fused_probability_v02/full2019'


def student_crps(residual, scale, df):
    z = np.asarray(residual)/scale
    constant = 2*np.sqrt(df)/(df-1)*np.exp(betaln(.5, df-.5)-2*betaln(.5, df/2))
    return scale*(z*(2*t.cdf(z, df)-1)+2*t.pdf(z, df)*(df+z*z)/(df-1)-constant)


def radial_scales(hours, members, df, multiplier, seed):
    result = []
    for hour in hours:
        row = []
        for member in range(members):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(hour), member, 841119]))
            row.append(multiplier/np.sqrt(rng.chisquare(df)/df))
        result.append(row)
    return np.array(result)


def quadratic_data(store, ids, factors, std, residual, population):
    _, assignment = np.unique(np.floor(store.xy[ids]/200000).astype(int), axis=0, return_inverse=True)
    output = []
    for group in np.unique(assignment):
        selection = np.flatnonzero(assignment == group)
        if len(selection) < 4:
            continue
        patterns, inverse = np.unique(population[:, selection], axis=0, return_inverse=True)
        for j, pattern in enumerate(patterns):
            selected = selection[pattern]
            if len(selected) < 4:
                continue
            e = residual[inverse == j][:, selected].astype('f8')
            covariance = sum(s*s*(h[selected]@h[selected].T) for s, h in zip(std, factors))+std[-1]**2*np.eye(len(selected))
            covariance.flat[::len(selected)+1] += 1e-8
            factor = cho_factor(covariance, lower=True)
            q = np.einsum('ti,it->t', e, cho_solve(factor, e.T))
            output.append(np.c_[q, np.full(len(q), len(selected)), np.full(len(q), 2*np.log(np.diag(factor[0])).sum())])
    return np.concatenate(output)


def tail_objective(parameters, summaries):
    multiplier, df = np.exp(parameters[0]), 2+np.exp(parameters[1])
    q, dimension, logdet = summaries.T
    likelihood = (gammaln((df+dimension)/2)-gammaln(df/2)
                  -.5*(dimension*np.log(df*np.pi)+logdet+2*dimension*np.log(multiplier))
                  -(df+dimension)/2*np.log1p(q/(df*multiplier**2)))
    return -float(likelihood.sum()/dimension.sum())


def macro(values, population):
    counts = population.sum(0)
    keep = counts > 0
    return float((np.where(population, values, 0).sum(0)[keep]/counts[keep]).mean())


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True, parents=True)
    immutable_json(ROOT/'protocol.lock.json', dict(source=sha256(__file__), baseline_parameters=sha256(GAUSSIAN.parent/'B_only_parameters.lock.json'),
        method='Student-t common radial scale mixture on same fixed Gaussian field and unresolved-observation component',
        motivation='B2018 blend residual excess kurtosis4.09; test one heavy-tail model without C fitting',
        parameters=['positive global scale multiplier', 'degrees of freedom >2.1'],
        fit='same B2018 200km group composite NLL', maximum_iterations=100,
        members=128, seed=42, D2020='sealed',
        caveat='Shared radial scale links unresolved error amplitude to field amplitude; not independent instrument noise'))
    data = ProbeData(device='cpu')
    b = data.arrays(2018)
    locked = read_json(GAUSSIAN.parent/'B_only_parameters.lock.json')
    std, weight = np.array(locked['model']['std']), locked['weight']
    bases = local_bases(data.store)
    bh = observation_factors(data.store, b['station_ids'], bases)
    mean = b['baseline']+weight*(b['idw']-b['baseline'])
    summaries = quadratic_data(data.store, b['station_ids'], bh, std, b['truth']-mean, b['valid'])
    result = minimize(tail_objective, np.log([.9, 6.]), args=(summaries,), method='L-BFGS-B',
                      bounds=[(np.log(.2), np.log(3.)), (np.log(.1), np.log(198.))], options=dict(maxiter=100, ftol=1e-12))
    if not result.success:
        raise RuntimeError(result.message)
    multiplier, df = np.exp(result.x[0]), 2+np.exp(result.x[1])
    state = dict(multiplier=float(multiplier), df=float(df), objective=result.fun, iterations=result.nit,
                 gaussian_objective=locked['model']['objective'], group_hours=len(summaries), station_hours=int(summaries[:, 1].sum()))
    immutable_json(ROOT/'B_only_parameters.lock.json', state)
    emit(f'B-only Student fit {state}')
    store, ids = data.development, data.development.ids('C')
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    ch = observation_factors(store, ids, bases)
    sigma = np.sqrt(sum(s*s*np.sum(h*h, 1) for s, h in zip(std, ch))+std[-1]**2)*multiplier
    series = {k: [] for k in ('student', 'gaussian')}
    hours, truths, means, populations = [], [], [], []
    for file in sorted(GAUSSIAN.glob('forecast_2019-*.npz')):
        z = dict(np.load(file))
        scales = radial_scales(z['hours'], 128, df, multiplier, 42)
        members = z['analytic_mean'][:, None]+(z['samples'].astype('f8')-z['analytic_mean'][:, None])*scales[..., None]
        for name, samples in [('student', members), ('gaussian', z['samples'])]:
            values, names = joint_series(z['truth'], samples, z['population'], metadata)
            series[name].append(values)
        hours.append(z['hours']); truths.append(z['truth']); means.append(z['analytic_mean']); populations.append(z['population'])
        emit(f'Student full2019 {file.stem[-7:]}')
    hours, truth, mean, population = [np.concatenate(v) for v in (hours, truths, means, populations)]
    series = {k: np.concatenate(v) for k, v in series.items()}
    active = np.isfinite(series['student'][..., 0]).any(0)
    series = {k: v[:, active] for k, v in series.items()}
    residual = truth-mean
    score = student_crps(np.where(population, residual, 0), sigma, df)
    intervals = {}
    for level in (.8, .9, .95):
        half = t.ppf((1+level)/2, df)*sigma
        intervals[str(round(level*100))] = dict(coverage=macro(abs(residual) <= half, population),
             mean_width_mm=macro(np.broadcast_to(2*half, population.shape), population))
    joint = {name: dict(ES_mm=float(np.nanmean(np.nanmean(v[..., 0], 0))),
                       VS_mm=float(np.nanmean(np.nanmean(v[..., 1], 0)))) for name, v in series.items()}
    difference = joint_intervals(hours, series['student'], series['gaussian'])
    save_npz(ROOT/'joint_scores.npz', hours=hours, **series)
    write_json(ROOT/'results.json', dict(parameters=state, joint=joint, paired_intervals=difference,
        station_macro_crps_mm=macro(score, population), intervals=intervals, n_station_hours=int(population.sum()),
        point='Same analytic blend mean:9.995855mm', interpretation='One exploratory B-fitted heavy-tail alternative; no C scale tuning'))
    emit(f'Full2019 Student CRPS {macro(score, population)}; intervals {intervals}; joint {joint}; paired {difference}')


if __name__ == '__main__':
    main()
