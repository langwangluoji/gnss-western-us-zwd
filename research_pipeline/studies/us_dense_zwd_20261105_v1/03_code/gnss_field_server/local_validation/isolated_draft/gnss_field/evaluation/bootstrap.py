"""Paired uncertainty with synchronized calendar-time blocks and fixed groups."""
import numpy as np

from .metrics import _arrays, point_metrics, _group_increment_metrics


class BootstrapExhaustedError(RuntimeError):
    """Bounded resampling failure with inspectable denominator-rejection counts."""

    def __init__(self, attempts, rejected_draws, accepted_replicates, requested_replicates):
        self.attempts = attempts
        self.rejected_draws = rejected_draws
        self.accepted_replicates = accepted_replicates
        self.requested_replicates = requested_replicates
        self.rejection_rate = rejected_draws/attempts if attempts else 0.
        super().__init__(f'Bootstrap exhausted {attempts} attempts with {rejected_draws} rejected draws '
                         f'and {accepted_replicates}/{requested_replicates} complete replicates')


def _attempt_limit(replicates, max_attempts):
    limit = max(1000, replicates*100) if max_attempts is None else max_attempts
    if not isinstance(limit, (int, np.integer)) or limit < replicates:
        raise ValueError('max_attempts must be an integer at least replicates')
    return limit


def _blocks(hours, block_days):
    hours = np.asarray(hours)
    if hours.ndim != 1 or not len(hours) or block_days <= 0:
        raise ValueError('Nonempty one-dimensional hours and positive block_days required')
    if np.issubdtype(hours.dtype, np.datetime64):
        elapsed = (hours-hours[0])/np.timedelta64(1, 'h')
    else:
        elapsed = hours.astype(float)-float(hours[0])
    if not np.isfinite(elapsed).all() or (np.diff(elapsed) <= 0).any():
        raise ValueError('Hours must be unique, finite and strictly increasing')
    labels = np.floor(elapsed/(block_days*24)).astype(int)
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _draw_blocks(blocks, rng):
    # Whole nonoverlapping calendar blocks preserve synchronized station series.
    # The trailing partial block is a declared block and is sampled as a whole.
    selected = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in selected])


def _summary(observed, values, unit='mm'):
    suffix = '_'+unit if unit else ''
    return {'difference'+suffix: float(observed),
            'ci95'+suffix: np.quantile(values, [.025, .975], method='linear').tolist(),
            'replicate_differences'+suffix: np.asarray(values),
            'replicates': len(values), 'direction': 'model_minus_baseline'}


def paired_block_bootstrap(hours, truth, prediction, baseline, population, *,
                           replicates=2000, block_days=7, seed=20261117, return_indices=False,
                           max_attempts=None):
    """Recompute station-macro RMSE for each paired all-station time resample."""
    truth, prediction, population = _arrays(truth, prediction, population)
    _, baseline, _ = _arrays(truth, baseline, population)
    blocks = _blocks(hours, block_days)
    if len(hours) != len(truth) or replicates < 1 or not population.any():
        raise ValueError('Bootstrap requires aligned nonempty population and positive replicates')
    rng = np.random.default_rng(seed)
    active_stations = population.any(axis=0)
    limit = _attempt_limit(replicates, max_attempts)
    observed = point_metrics(truth, prediction, population)['station_macro_rmse_mm']-point_metrics(truth, baseline, population)['station_macro_rmse_mm']
    values, indices_record = [], []
    attempts, rejected = 0, 0
    while len(values) < replicates:
        if attempts == limit:
            raise BootstrapExhaustedError(attempts, rejected, len(values), replicates)
        attempts += 1
        indices = _draw_blocks(blocks, rng)
        pop = population[indices]
        if not pop.any(axis=0)[active_stations].all():
            rejected += 1
            continue
        values.append(point_metrics(truth[indices], prediction[indices], pop)['station_macro_rmse_mm']-
                      point_metrics(truth[indices], baseline[indices], pop)['station_macro_rmse_mm'])
        if return_indices:
            indices_record.append(indices)
    result = _summary(observed, values)
    result.update(block_days=block_days, seed=seed, n_blocks=len(blocks),
                  n_stations=int(active_stations.sum()), active_station_indices=np.flatnonzero(active_stations).tolist(),
                  attempts=attempts, rejected_draws=rejected, rejection_rate=rejected/attempts,
                  max_attempts=limit,
                  resampling='nonoverlapping calendar blocks, all stations synchronized')
    if return_indices:
        result['indices'] = indices_record
    return result


def crossed_increment_bootstrap(hours, truth, prediction, baseline, population, group_ids, *,
                                replicates=2000, block_days=7, seed=20261117, max_attempts=None):
    """Crossed paired resampling of time blocks and fixed predeclared dense groups.

    Columns are fixed same-hour station pairs. Use pair_increment_arrays and
    restrict its declared population to the preregistered distance endpoint.
    """
    truth, prediction, population = _arrays(truth, prediction, population)
    _, baseline, _ = _arrays(truth, baseline, population)
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (truth.shape[1],) or len(hours) != len(truth) or replicates < 1:
        raise ValueError('Pair groups and hours must match arrays')
    all_groups = list(dict.fromkeys(group_ids))
    groups = [group for group in all_groups if population[:, group_ids == group].any()]
    if not groups or not population.any():
        raise ValueError('Crossed bootstrap requires observed fixed groups')
    blocks = _blocks(hours, block_days)
    rng = np.random.default_rng(seed)
    limit = _attempt_limit(replicates, max_attempts)
    observed = _group_increment_metrics(truth, prediction, population, group_ids)['group_macro_rmse_mm']-_group_increment_metrics(truth, baseline, population, group_ids)['group_macro_rmse_mm']
    differences = []
    attempts, rejected = 0, 0
    while len(differences) < replicates:
        if attempts == limit:
            raise BootstrapExhaustedError(attempts, rejected, len(differences), replicates)
        attempts += 1
        indices = _draw_blocks(blocks, rng)
        selected = rng.integers(0, len(groups), len(groups))
        selected_columns = [group_ids == groups[selection] for selection in selected]
        masks = [population[indices][:, columns] for columns in selected_columns]
        if not all(mask.any() for mask in masks):
            rejected += 1
            continue
        model_values, baseline_values = [], []
        for columns, mask in zip(selected_columns, masks):
            model_error = (prediction[indices][:, columns]-truth[indices][:, columns])[mask]
            baseline_error = (baseline[indices][:, columns]-truth[indices][:, columns])[mask]
            model_values.append(np.sqrt(np.mean(model_error**2)))
            baseline_values.append(np.sqrt(np.mean(baseline_error**2)))
        differences.append(np.mean(model_values)-np.mean(baseline_values))
    result = _summary(observed, differences)
    result.update(block_days=block_days, seed=seed, n_groups=len(groups),
                  eligible_group_ids=np.asarray(groups).tolist(),
                  excluded_group_ids=np.asarray([group for group in all_groups if group not in groups]).tolist(),
                  attempts=attempts, rejected_draws=rejected, rejection_rate=rejected/attempts,
                  max_attempts=limit,
                  resampling='crossed synchronized calendar blocks and fixed dense group IDs')
    return result


def event_block_bootstrap(model_values, baseline_values, event_ids, *, replicates=2000, seed=20261117):
    """Paired mean-statistic bootstrap of whole named weather processes."""
    model, baseline, ids = np.asarray(model_values, float), np.asarray(baseline_values, float), np.asarray(event_ids)
    if model.ndim != 1 or model.shape != baseline.shape or ids.shape != model.shape or not len(model):
        raise ValueError('Expected aligned nonempty event observations')
    if not np.isfinite(model).all() or not np.isfinite(baseline).all() or replicates < 1:
        raise ValueError('Finite event observations and positive replicates required')
    blocks = [np.flatnonzero(ids == event) for event in dict.fromkeys(ids)]
    rng = np.random.default_rng(seed)
    delta = model-baseline
    values = [delta[_draw_blocks(blocks, rng)].mean() for _ in range(replicates)]
    result = _summary(delta.mean(), values, unit='')
    result.update(n_events=len(blocks), seed=seed, statistic='paired observation mean, whole events resampled')
    return result
