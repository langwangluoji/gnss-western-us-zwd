"""Registered array evaluation. All ZWD arrays are millimetres, xy metres.

Masks are declared scientific populations, never finite-prediction filters.
Ensemble scores use the finite forecast V-statistic, including diagonal terms.
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from scipy.spatial.distance import cdist


def _arrays(truth, prediction, population):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    population = np.asarray(population)
    if truth.ndim != 2 or prediction.shape != truth.shape or population.shape != truth.shape:
        raise ValueError('Expected matching [T,N] truth, prediction and population')
    if population.dtype != bool:
        raise ValueError('Population must be boolean')
    if not np.isfinite(truth[population]).all() or not np.isfinite(prediction[population]).all():
        raise ValueError('Truth and predictions must be finite throughout declared population')
    return truth, prediction, population


def _station_macro(values, mask):
    count = mask.sum(axis=0)
    keep = count > 0
    if not keep.any():
        return None
    sums = np.where(mask, values, 0.).sum(axis=0)
    return float(np.mean(sums[keep] / count[keep]))


def point_metrics(truth, prediction, population):
    """Surface-ZWD station macro and station-hour micro metrics, already in mm."""
    truth, prediction, population = _arrays(truth, prediction, population)
    error = np.where(population, prediction-truth, 0.)
    count = population.sum(axis=0)
    keep = count > 0
    total = int(count.sum())
    per_station = np.sqrt(np.sum(error**2, axis=0)[keep]/count[keep])
    return dict(station_macro_rmse_mm=float(per_station.mean()) if total else None,
                micro_rmse_mm=float(np.sqrt(np.sum(error**2)/total)) if total else None,
                station_macro_mae_mm=_station_macro(abs(error), population),
                micro_mae_mm=float(abs(error).sum()/total) if total else None,
                station_macro_bias_mm=_station_macro(error, population),
                bias_mm=float(error.sum()/total) if total else None,
                p95_absolute_error_mm=float(np.quantile(abs(error[population]), .95, method='linear')) if total else None,
                n_stations=int(keep.sum()), n_station_hours=total)


def crps_ensemble(truth, samples, axis=-1):
    """Empirical CRPS with denominator K squared and linear sorted-sample cost."""
    samples = np.moveaxis(np.asarray(samples, float), axis, -1)
    truth = np.asarray(truth, float)
    if samples.shape[-1] < 1 or not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise ValueError('CRPS requires finite truth and at least one finite member')
    ordered = np.sort(samples, axis=-1)
    k = samples.shape[-1]
    weights = 2*np.arange(1, k+1)-k-1
    return np.mean(abs(samples-truth[..., None]), axis=-1) - np.sum(ordered*weights, axis=-1)/(k*k)


def energy_score_ensemble(truth, samples):
    """L2 norm to power one, divided by sqrt(N), finite-ensemble V-statistic."""
    truth, samples = np.asarray(truth, float), np.asarray(samples, float)
    if truth.ndim != 1 or samples.ndim != 2 or samples.shape[1] != len(truth) or not len(truth) or not len(samples):
        raise ValueError('Expected truth[N] and samples[K,N]')
    if not np.isfinite(truth).all() or not np.isfinite(samples).all():
        raise ValueError('Joint scores require finite members')
    return float((np.linalg.norm(samples-truth, axis=1).mean() - .5*cdist(samples, samples).mean()) / np.sqrt(len(truth)))


def variogram_score(truth, samples, distance_km):
    """VS p=.5, upper-triangle inverse-distance weights normalized to sum one."""
    truth, samples = np.asarray(truth, float), np.asarray(samples, float)
    distance_km = np.asarray(distance_km, float)
    energy_score_ensemble(truth, samples)  # Shared shape/finite validation.
    if distance_km.shape != (len(truth), len(truth)) or not np.isfinite(distance_km).all() or (distance_km < 0).any():
        raise ValueError('Invalid pair distance matrix')
    i, j = np.triu_indices(len(truth), 1)
    if not len(i):
        raise ValueError('VS requires at least two stations')
    weights = 1/np.maximum(distance_km[i, j], 25.)
    weights /= weights.sum()
    observed = np.sqrt(abs(truth[i]-truth[j]))
    forecast = np.sqrt(abs(samples[:, i]-samples[:, j])).mean(axis=0)
    return float(np.sum(weights*(observed-forecast)**2))


def _probability(truth, prediction, population, samples, std):
    result = {}
    if samples is not None:
        # Index only the fixed population before score computation. Missing truth
        # outside F is permitted and cannot enter the finite-value validator.
        values = np.zeros_like(truth)
        values[population] = crps_ensemble(truth[population], samples.transpose(0, 2, 1)[population])
        bounds = {level: np.quantile(samples, [(1-level)/2, (1+level)/2], axis=1, method='linear')
                  for level in (.8, .9, .95)}
        result['crps_method'] = 'finite_ensemble_V_statistic'
        result['members'] = int(samples.shape[1])
        result['quantile_method'] = 'numpy.quantile linear Hyndman-Fan7'
        result['numpy_version'] = np.__version__
    elif std is not None:
        z = np.zeros_like(truth)
        z[population] = (truth[population]-prediction[population])/std[population]
        values = std*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
        bounds = {level: (prediction-norm.ppf((1+level)/2)*std,
                          prediction+norm.ppf((1+level)/2)*std) for level in (.8, .9, .95)}
        nll = np.zeros_like(truth)
        nll[population] = .5*np.log(2*np.pi*std[population]**2)+.5*z[population]**2
        result['station_macro_gaussian_nll'] = _station_macro(nll, population)
        result['micro_gaussian_nll'] = float(nll[population].mean()) if population.any() else None
        result['crps_method'] = 'analytic_Gaussian'
    else:
        return result
    result['station_macro_crps_mm'] = _station_macro(values, population)
    result['micro_crps_mm'] = float(values[population].mean()) if population.any() else None
    result['intervals'] = {}
    for level, (lower, upper) in bounds.items():
        covered = (truth >= lower) & (truth <= upper)
        width = upper-lower
        coverage = _station_macro(covered, population)
        result['intervals'][str(round(level*100))] = dict(
            nominal=level, coverage=coverage, mean_width_mm=_station_macro(width, population),
            micro_coverage=float(covered[population].mean()) if population.any() else None,
            n_covered=int(covered[population].sum()), n_station_hours=int(population.sum()),
            n_stations=int((population.sum(axis=0)>0).sum()),
            engineering_pass=abs(coverage-level) <= .03 if coverage is not None else None)
    return result


def pair_increment_arrays(truth, prediction, population, metadata):
    """Paired same-hour Z0 increments in declared dense groups, mm and km.

    Returned columns preserve fixed global pair/group IDs for crossed bootstrap.
    No current input observations or labels are loaded by this primitive.
    """
    truth, prediction, population = _arrays(truth, prediction, population)
    xy = np.asarray(metadata[['x', 'y']], float)
    height = np.asarray(metadata['height'], float)
    groups = np.asarray(metadata['dense_group'])
    i, j = np.triu_indices(len(groups), 1)
    same = (groups[i] == groups[j]) & ~np.asarray(metadata['dense_group'].isna())[i]
    i, j = i[same], j[same]
    distance = np.linalg.norm(xy[i]-xy[j], axis=1)/1000.
    scale = np.exp(height/3000.)
    return dict(truth=truth[:, j]*scale[j]-truth[:, i]*scale[i],
                prediction=prediction[:, j]*scale[j]-prediction[:, i]*scale[i],
                surface_truth=truth[:, j]-truth[:, i],
                surface_prediction=prediction[:, j]-prediction[:, i],
                population=population[:, i] & population[:, j], distance_km=distance,
                same_height=abs(height[i]-height[j]) <= 100., group_ids=groups[i],
                pair_indices=np.stack((i, j), axis=1))


def _group_increment_metrics(truth, prediction, population, group_ids):
    errors = prediction-truth
    rows = []
    count = 0
    for group in dict.fromkeys(group_ids):
        selection = population[:, group_ids == group]
        values = errors[:, group_ids == group][selection]
        if values.size:
            rows.append(float(np.sqrt(np.mean(values**2))))
            count += values.size
    return dict(group_macro_rmse_mm=float(np.mean(rows)) if rows else None,
                n_groups=len(rows), n_pair_hours=int(count),
                n_pairs=int((population.sum(axis=0)>0).sum()))


def increment_metrics(truth, prediction, population, metadata):
    pairs = pair_increment_arrays(truth, prediction, population, metadata)
    distance = pairs['distance_km']
    result = {}
    selections = {'0-25': (distance >= 0) & (distance < 25),
                  '25-50': (distance >= 25) & (distance < 50),
                  '50-100': (distance >= 50) & (distance < 100),
                  '100-200': (distance >= 100) & (distance <= 200),
                  '0-50': (distance >= 0) & (distance <= 50)}
    for label, selection in selections.items():
        mask = pairs['population'] & selection[None, :]
        result[label] = _group_increment_metrics(pairs['truth'], pairs['prediction'], mask, pairs['group_ids'])
        result[label]['same_height_surface'] = _group_increment_metrics(
            pairs['surface_truth'], pairs['surface_prediction'], mask & pairs['same_height'][None, :], pairs['group_ids'])
    return result


def joint_metrics(truth, samples, population, metadata):
    groups = np.asarray(metadata['dense_group'])
    coordinates = np.asarray(metadata[['x', 'y']], float)
    rows = []
    station_hours = 0
    for group in dict.fromkeys(groups[~np.asarray(metadata['dense_group'].isna())]):
        scores = []
        for hour in range(len(truth)):
            indices = np.flatnonzero((groups == group) & population[hour])
            if len(indices) < 4:
                continue
            observed = truth[hour, indices]
            members = samples[hour][:, indices]
            distances = cdist(coordinates[indices], coordinates[indices])/1000.
            scores.append((energy_score_ensemble(observed, members), variogram_score(observed, members, distances)))
            station_hours += len(indices)
        if scores:
            rows.append(dict(group_id=str(group), es_mm=float(np.mean(scores, axis=0)[0]),
                             vs_mm=float(np.mean(scores, axis=0)[1]), n_hours=len(scores)))
    return dict(group_macro_es_mm=float(np.mean([r['es_mm'] for r in rows])) if rows else None,
                group_macro_vs_mm=float(np.mean([r['vs_mm'] for r in rows])) if rows else None,
                n_groups=len(rows), n_group_hours=sum(r['n_hours'] for r in rows),
                n_station_hours=station_hours, groups=rows,
                member_contract='same global member index across every station',
                estimator='finite_forecast_V_statistic')


def evaluate_arrays(hours, station_ids, truth, prediction, population_F, normal_support_S0, *,
                    samples=None, std=None, metadata, coarsened_prediction=None, safe_T=None,
                    coarsened_samples=None):
    """Evaluate immutable F/S0 populations with metadata rows aligned to station_ids.

    Input truth/mean/members/std are surface ZWD mm. x/y/height are metres.
    Metadata must already match N rows. station_ids remain stable global indices.
    """
    truth, prediction, population_F = _arrays(truth, prediction, population_F)
    support = np.asarray(normal_support_S0)
    if support.dtype != bool or support.shape != truth.shape or (support & ~population_F).any():
        raise ValueError('S0 must be a boolean subset of F with shape [T,N]')
    ids, hours = np.asarray(station_ids), np.asarray(hours)
    if ids.shape != (truth.shape[1],) or len(np.unique(ids)) != len(ids) or hours.shape != (truth.shape[0],):
        raise ValueError('Station IDs and hours must match [T,N], with unique station IDs')
    columns = {'x', 'y', 'height', 'dense_group', 'cohort'}
    if len(metadata) != len(ids) or not columns.issubset(metadata.columns):
        raise ValueError('Metadata must match N and include x,y,height,dense_group,cohort')
    if not np.isfinite(np.asarray(metadata[['x', 'y', 'height']], float)).all():
        raise ValueError('Metadata coordinates/heights must be finite')
    if samples is not None:
        samples = np.asarray(samples, float)
        if samples.ndim != 3 or samples.shape[::2] != truth.shape or samples.shape[1] < 1:
            raise ValueError('Samples must have shape [T,K,N]')
        if not np.isfinite(samples.transpose(0, 2, 1)[population_F]).all():
            raise ValueError('All forecast members must be finite throughout F')
    if std is not None:
        std = np.asarray(std, float)
        if std.shape != truth.shape or not np.isfinite(std[population_F]).all() or (std[population_F] <= 0).any():
            raise ValueError('Gaussian std must be finite and positive throughout F')
    if samples is not None and std is not None:
        raise ValueError('Supply ensemble samples or Gaussian std for a single forecast product')
    result = {}
    for name, population in [('full', population_F), ('primary', support), ('unsupported', population_F & ~support)]:
        result[name] = point_metrics(truth, prediction, population)
        result[name].update(_probability(truth, prediction, population, samples, std))
    result['increments'] = increment_metrics(truth, prediction, support, metadata)
    if samples is not None:
        result['joint'] = joint_metrics(truth, samples, support, metadata)
    if coarsened_samples is not None and coarsened_prediction is None:
        raise ValueError('Coarsened samples require coarsened_prediction')
    if coarsened_prediction is not None:
        if safe_T is None:
            raise ValueError('Coarsened evaluation requires explicit common safe_T')
        safe = np.asarray(safe_T)
        if safe.dtype != bool or safe.shape not in ((len(ids),), truth.shape):
            raise ValueError('safe_T must be boolean [N] or [T,N]')
        population = support & np.broadcast_to(safe, truth.shape)
        _, coarse, _ = _arrays(truth, coarsened_prediction, population)
        coarse_members = None
        if coarsened_samples is not None:
            coarse_members = np.asarray(coarsened_samples, float)
            if samples is None or coarse_members.shape != samples.shape:
                raise ValueError('Coarsened samples require matching model [T,K,N] global members')
            if not np.isfinite(coarse_members.transpose(0, 2, 1)[population]).all():
                raise ValueError('Coarsened members must be finite on common T')
        fine = dict(model=point_metrics(truth, prediction, population),
                    coarsened=point_metrics(truth, coarse, population),
                    model_increments=increment_metrics(truth, prediction, population, metadata),
                    coarsened_increments=increment_metrics(truth, coarse, population, metadata),
                    scope='common complete coarsening stencils intersected with S0')
        fine['model'].update(_probability(truth, prediction, population, samples, std))
        fine['coarsened'].update(_probability(truth, coarse, population, coarse_members, None))
        result['fine'] = fine
    return result
