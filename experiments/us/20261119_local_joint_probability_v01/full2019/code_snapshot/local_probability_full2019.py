"""Full-development-year fixed-mean covariance validation, no parameter search."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from local_probe_diagnostics import ProbeData, PROJECT, emit, save_npz
from local_joint_field import local_bases, observation_factors, sample_factors, shuffle_members, sample_diagonal
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import immutable_json, write_json, read_json, sha256
from gnss_field.evaluation import evaluate_arrays, point_metrics

ROOT = PROJECT/'results/20261119_local_joint_probability_v01/full2019'
M0 = PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'


def main():
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=True, parents=True)
    data = ProbeData(device='cpu')
    state = read_json(ROOT.parent/'B_only_parameters.lock.json')['model']
    std = np.array(state['std'])
    files = sorted(M0.glob('forecast_*.npz'))
    assert len(files) == 12
    immutable_json(ROOT/'protocol.lock.json', dict(source=sha256(__file__), generator_source=sha256(PROJECT/'tools/local_joint_field.py'),
        frozen_parameters_sha256=sha256(ROOT.parent/'B_only_parameters.lock.json'),
        inputs={f.name: sha256(f) for f in files}, members=128, seed=42,
        alternatives=['correlated', 'exact_diagonal_Gaussian', 'empirically_marginal_matched_shuffle'],
        mean='fixed M0 analytic mean for all alternatives', no_C_parameter_fitting=True, D2020='sealed'))
    store, ids = data.development, data.development.ids('C')
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    bases = local_bases(store)
    factors = observation_factors(store, ids, bases)
    marginal_std = np.sqrt(sum(s*s*np.sum(h*h, axis=1) for s, h in zip(std, factors))+std[-1]**2)
    series, hours, truth, prediction, f0, s0 = {k: [] for k in ('correlated', 'diagonal', 'shuffled')}, [], [], [], [], []
    started = time.monotonic()
    monthly = []
    for path in files:
        z = dict(np.load(path))
        np.testing.assert_array_equal(z['station_ids'], ids)
        members = z['prediction'][:, None]+sample_factors(factors, std, z['hours'], ids, 128, 42)
        independent = z['prediction'][:, None]+sample_diagonal(marginal_std, z['hours'], ids, 128, 42)
        shuffled = shuffle_members(members, z['hours'], ids, 42)
        population = z['normal_support_S0'] & z['population_F']
        if not np.isfinite(members.transpose(0, 2, 1)[population]).all():
            raise ValueError('Nonfinite primary forecast')
        for name, samples in [('correlated', members), ('diagonal', independent), ('shuffled', shuffled)]:
            values, names = joint_series(z['truth'], samples, population, metadata)
            series[name].append(values)
        hours.append(z['hours'])
        truth.append(z['truth'])
        prediction.append(z['prediction'])
        f0.append(z['population_F'])
        s0.append(population)
        # Store scored station members, never annual full-grid ensembles.
        save_npz(ROOT/path.name, hours=z['hours'], station_ids=ids, analytic_mean=z['prediction'],
                 samples=members.astype('f4'), population=population, truth=z['truth'])
        month = path.stem[-7:]
        monthly.append(dict(month=month, M0=point_metrics(z['truth'], z['prediction'], population),
            marginal_std_mm=float(marginal_std[population.any(0)].mean())))
        emit(f'Full2019 joint {month} finished; elapsed {time.monotonic()-started:.1f}s')
    hours, truth, prediction, f0, s0 = [np.concatenate(v) for v in (hours, truth, prediction, f0, s0)]
    assert len(hours) == 8760 and s0.sum() == 642258 and s0.any(0).sum() == 76
    series = {k: np.concatenate(v) for k, v in series.items()}
    metrics = {name: dict(ES_mm=float(np.nanmean(np.nanmean(value[..., 0], 0))),
                         VS_mm=float(np.nanmean(np.nanmean(value[..., 1], 0))), n_groups=len(names)) for name, value in series.items()}
    intervals = {name: joint_intervals(hours, series['correlated'], series[name]) for name in ('diagonal', 'shuffled')}
    # Analytic Gaussian marginal scores are exactly common to all three laws.
    marginal = evaluate_arrays(hours, ids, truth, prediction, f0, s0,
                              std=np.broadcast_to(marginal_std, truth.shape), metadata=metadata)
    seasonal = []
    month = pd.to_datetime(hours, unit='h').month.to_numpy()
    for label, months in [('DJF', [12, 1, 2]), ('MAM', [3, 4, 5]), ('JJA', [6, 7, 8]), ('SON', [9, 10, 11])]:
        pop = s0 & np.isin(month, months)[:, None]
        seasonal.append(dict(season=label, metrics=point_metrics(truth, prediction, pop)))
    save_npz(ROOT/'joint_scores.npz', hours=hours, **series)
    write_json(ROOT/'results.json', dict(parameters=state, joint=metrics, intervals=intervals,
        analytic_common_marginal=marginal, seasonal=seasonal, monthly=monthly,
        total_seconds=time.monotonic()-started, n_hours=len(hours), n_station_hours=int(s0.sum()),
        groups=names, comparison='same fixed analytic mean and variance; shuffled also exactly matches finite marginals',
        noise_interpretation='surface observation plus unresolved error; excluded from published latent grid',
        inference='C2019 development, post-selection exploratory intervals; D2020 sealed'))
    emit(str(metrics))
    emit(str(intervals))


if __name__ == '__main__':
    main()
