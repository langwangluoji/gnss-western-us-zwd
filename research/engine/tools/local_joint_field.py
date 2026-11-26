"""Fixed local field factors and small composite Gaussian likelihood experiment."""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gnss_field.geometry import surface_query


def covariance_objective(log_std, records):
    variance = np.exp(2*np.asarray(log_std))
    objective, gradient, denominator = 0., np.zeros(len(variance)), 0
    for count, second, kernels in records:
        covariance = np.einsum('k,kij->ij', variance, kernels)
        covariance.flat[::len(covariance)+1] += 1e-8
        factor = cho_factor(covariance, lower=True, check_finite=False)
        inverse = cho_solve(factor, np.eye(len(covariance)), check_finite=False)
        objective += .5*count*(2*np.log(np.diag(factor[0])).sum()+np.sum(inverse*second.T)+len(covariance)*np.log(2*np.pi))
        derivative = inverse-inverse@second@inverse
        gradient += count*variance*np.einsum('ij,kji->k', derivative, kernels)
        denominator += count*len(covariance)
    if not denominator:
        raise ValueError('No qualified B station-group hours')
    return objective/denominator, gradient/denominator


def local_bases(store):
    xy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
    bases = []
    for spacing, radius in ((100000., 150000.), (300000., 450000.)):
        axes = [np.arange(np.floor(xy[:, i].min()/spacing)-1, np.ceil(xy[:, i].max()/spacing)+2)*spacing for i in (0, 1)]
        xx, yy = np.meshgrid(*axes)
        centres = np.c_[xx.ravel(), yy.ravel()]
        distance = cdist(xy, centres)/radius
        basis = np.maximum(1-distance**2, 0)**2
        basis = basis[:, (basis > 0).any(0)]
        basis /= np.sqrt((basis*basis).sum(1, keepdims=True)).clip(1e-15)
        bases.append(basis)
    return bases


def observation_factors(store, ids, bases):
    from gnss_field.training.prediction import safe_query_geometry
    xy, height = safe_query_geometry(store, ids)
    return [surface_query(torch.tensor(v.T.reshape(-1, *store.shape)), xy, height).numpy().T for v in bases]


def group_records(store, ids, factors, residual, population):
    # 200km groups are fixed from coordinates, not from observed residuals.
    coordinates = store.xy[ids]
    keys = np.floor(coordinates/200000).astype(int)
    _, assignment = np.unique(keys, axis=0, return_inverse=True)
    records = []
    group_hours, station_hours = 0, 0
    for group in np.unique(assignment):
        selection = np.flatnonzero(assignment == group)
        if len(selection) < 4:
            continue
        patterns, inverse = np.unique(population[:, selection], axis=0, return_inverse=True)
        for j, pattern in enumerate(patterns):
            selected = selection[pattern]
            if len(selected) < 4:
                continue
            values = residual[inverse == j][:, selected].astype('f8')
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite declared training residual')
            kernels = [v[selected]@v[selected].T for v in factors]+[np.eye(len(selected))]
            records.append((len(values), values.T@values/len(values), np.array(kernels)))
            group_hours += len(values)
            station_hours += values.size
    return records, dict(group_hours=group_hours, station_hours=station_hours,
                         patterns=len(records), grouping='fixed projected 200km cells, >=4 valid B stations')


def fit_covariance(records):
    result = minimize(covariance_objective, np.log([4., 4., 4.]), args=(records,), jac=True,
                      method='L-BFGS-B', bounds=[(np.log(.01), np.log(60.))]*3,
                      options=dict(maxiter=100, ftol=1e-11, gtol=1e-7))
    if not result.success:
        raise RuntimeError(result.message)
    return dict(std=np.exp(result.x), objective=result.fun, iterations=result.nit,
                gradient_norm=float(np.linalg.norm(result.jac)), success=bool(result.success))


def sample_factors(factors, std, hours, station_ids, members, seed):
    rows = []
    for hour in hours:
        draws = [[] for _ in factors]
        noise = []
        for member in range(members):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(hour), member, 982476]))
            for group, factor in enumerate(factors):
                draws[group].append(rng.standard_normal(factor.shape[1]))
            noise.append(rng.standard_normal(905)[station_ids])
        value = std[-1]*np.array(noise)
        for index, factor in enumerate(factors):
            value += std[index]*np.einsum('nr,kr->kn', factor, np.array(draws[index]), optimize=False)
        rows.append(value)
    return np.array(rows)


def shuffle_members(values, hours, station_ids, seed):
    result = np.array(values, copy=True)
    for t, hour in enumerate(hours):
        for n, station in enumerate(station_ids):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(hour), int(station), 14119]))
            result[t, :, n] = values[t, rng.permutation(values.shape[1]), n]
    return result


def sample_diagonal(std, hours, station_ids, members, seed):
    rows = []
    for hour in hours:
        value = []
        for member in range(members):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(hour), member, 941119]))
            value.append(rng.standard_normal(905)[station_ids]*std)
        rows.append(value)
    return np.array(rows)
