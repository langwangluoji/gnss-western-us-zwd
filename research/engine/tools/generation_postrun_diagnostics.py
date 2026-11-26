"""Post-hoc matched-member diagnostics; no model training or label-based tuning."""
import numpy as np


def paired_block_rmse(truth, baseline, candidate, population, block, draws=2000, seed=20261122):
    """Paired time-cluster resampling, fixed station cohort; exploratory interval."""
    truth, baseline, candidate = [np.asarray(x, dtype='f8') for x in (truth, baseline, candidate)]
    mask = np.asarray(population, bool)
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or not np.isfinite(truth[mask]).all():
        raise ValueError('Nonfinite bootstrap inputs')
    _, block = np.unique(block, return_inverse=True)
    groups = np.unique(block)
    counts = np.array([mask[block == b].sum(0) for b in groups], dtype='f8')
    use = counts.sum(0) > 0
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(groups), np.ones(len(groups))/len(groups), size=draws)
    count = weights @ counts[:, use]
    if np.any(count == 0):
        raise ValueError('Bootstrap omitted a registered station; cannot silently drop it')

    def calculate(prediction):
        error = np.where(mask, prediction-truth, 0.)**2
        sums = np.array([error[block == b].sum(0) for b in groups])[:, use]
        point = np.sqrt(sums.sum(0)/counts[:, use].sum(0)).mean()
        boot = np.sqrt((weights @ sums)/count).mean(1)
        return point, boot

    a, aa = calculate(baseline); b, bb = calculate(candidate)
    return dict(difference_mm=float(b-a), interval95_mm=np.quantile(bb-aa, [.025, .975]).tolist(),
                blocks=len(groups), draws=draws, seed=seed,
                interpretation='candidate minus baseline; unadjusted exploratory interval on sampled weather-week clusters')


def fair_crps(truth, samples):
    """Ferro's iid-member U estimator, supplemental to the locked V score."""
    truth, samples = np.asarray(truth, dtype='f8'), np.asarray(samples, dtype='f8')
    k = samples.shape[1]
    if k < 2 or not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise ValueError('Fair CRPS requires finite inputs and at least two members')
    weights = (2*np.arange(1, k+1)-k-1)[None, :, None]
    spread = (np.sort(samples, axis=1)*weights).sum(1) / (k*(k-1))
    return abs(samples-truth[:, None]).mean(1)-spread


def matched_decomposition(gaussian, generated):
    for key in ('hours', 'station_ids', 'truth', 'population'):
        if not np.array_equal(gaussian[key], generated[key], equal_nan=True):
            raise ValueError(f'Unmatched {key}')
    for value in (gaussian, generated):
        for key in ('samples', 'mean'):
            if not np.isfinite(value[key]).all():
                raise ValueError(f'Nonfinite {key}')
    if gaussian['samples'].shape != generated['samples'].shape:
        raise ValueError('Unmatched member count or array shape')
    shift = np.asarray(generated['mean'], dtype='f8') - gaussian['mean']
    correction = np.asarray(generated['samples'], dtype='f8') - gaussian['samples']
    recovered = gaussian['mean'] + correction.mean(1)
    # Saved means use the known-zero prior control variate. Matching keyed draws
    # should reproduce them after subtraction, up to FP32 arithmetic.
    if np.max(np.abs(recovered - generated['mean'])) > .001:
        raise ValueError('Members are not paired, or mean estimator changed')
    return dict(mean_shift=shift, correction=correction,
                gaussian_shifted=gaussian['samples'] + shift[:, None],
                generated_recentered=generated['samples'] - shift[:, None])


def independent_centering(gaussian, generated, pilot, evaluation):
    """Fixed independent pilot shift; validation targets never fit shift or scale."""
    pilot, evaluation = np.asarray(pilot), np.asarray(evaluation)
    if np.intersect1d(pilot, evaluation).size:
        raise ValueError('Pilot and evaluated member sets overlap')
    if len(pilot) < 2 or len(evaluation) < 2:
        raise ValueError('Need two nonempty member sets with at least two members')
    decomposition = matched_decomposition(gaussian, generated)
    correction = decomposition['correction']
    shift = correction[:, pilot].mean(1)
    # One scalar per hour, estimated using only fixed-station forecast members.
    # No current observation availability or target value enters this control.
    before = np.var(gaussian['samples'][:, pilot], axis=1, ddof=1).mean(1)
    after = np.var(generated['samples'][:, pilot], axis=1, ddof=1).mean(1)
    if np.any(before <= 0):
        raise ValueError('Degenerate Gaussian pilot variance')
    scale = np.sqrt(after / before)
    baseline_samples = gaussian['samples'][:, evaluation]
    return dict(samples=generated['samples'][:, evaluation] - shift[:, None],
                mean=gaussian['mean'] + correction[:, evaluation].mean(1) - shift,
                baseline_samples=baseline_samples, pilot_shift=shift, scale=scale,
                scaled_gaussian=gaussian['mean'][:, None] + scale[:, None, None] *
                                (baseline_samples - gaussian['mean'][:, None]))


def rank_histogram(truth, samples, population, seed=20261122):
    truth, samples = np.asarray(truth), np.asarray(samples)
    mask = np.asarray(population, bool)
    if not np.isfinite(samples).all() or not np.isfinite(truth[mask]).all():
        raise ValueError('Nonfinite rank inputs')
    less = np.sum(samples < truth[:, None], axis=1)
    ties = np.sum(samples == truth[:, None], axis=1)
    rng = np.random.default_rng(seed)
    ranks = less + rng.integers(ties + 1)
    bins = samples.shape[1] + 1
    counts = np.bincount(ranks[mask], minlength=bins)
    if not counts.sum():
        raise ValueError('Empty rank population')
    per_station = [np.bincount(ranks[:, j][mask[:, j]], minlength=bins) / mask[:, j].sum()
                   for j in range(mask.shape[1]) if mask[:, j].any()]
    return dict(counts=counts.tolist(), frequency=(counts / counts.sum()).tolist(),
                station_macro_frequency=np.mean(per_station, axis=0).tolist(),
                reference_probability=1 / bins, members=samples.shape[1],
                tie_policy='uniform random rank among target and tied members')

