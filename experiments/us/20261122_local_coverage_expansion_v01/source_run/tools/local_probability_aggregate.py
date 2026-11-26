"""Recover full-year group aggregation from already saved monthly members."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from local_probe_diagnostics import ProbeData, PROJECT, emit, save_npz
from local_joint_field import local_bases, observation_factors, sample_diagonal, shuffle_members
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import immutable_json, read_json, sha256, write_json
from gnss_field.evaluation import evaluate_arrays, point_metrics


def main(directory):
    torch.set_num_threads(2)
    root = Path(directory)
    if not root.is_absolute():
        root = PROJECT/root
    files = sorted(root.glob('forecast_2019-*.npz'))
    assert len(files) == 12, 'All twelve monthly forecasts required'
    state = read_json(root.parent/'B_only_parameters.lock.json')['model']
    std = np.array(state['std'])
    data = ProbeData(device='cpu')
    store, ids = data.development, data.development.ids('C')
    metadata = store.stations.iloc[ids].reset_index(drop=True)
    factors = observation_factors(store, ids, local_bases(store))
    sigma = np.sqrt(sum(s*s*np.sum(h*h, axis=1) for s, h in zip(std, factors))+std[-1]**2)
    immutable_json(root/'aggregation.lock.json', dict(source=sha256(__file__), helper=sha256(PROJECT/'tools/local_probability_experiment.py'),
        forecasts={f.name: sha256(f) for f in files}, parameters=sha256(root.parent/'B_only_parameters.lock.json'),
        population='Fixed named groups across months; missing groups retain NaN slots; no valid forecast deleted',
        stored_forecast_precision='FP32; all scores FP64; no model or scientific parameter changes',
        alternatives=['correlated', 'exact independent Gaussian diag(Sigma)', 'marginal-preserving shuffle']))
    series, arrays = {k: [] for k in ('correlated', 'diagonal', 'shuffled')}, {k: [] for k in ('hours', 'truth', 'mean', 'population', 'F0')}
    for file in files:
        z = dict(np.load(file))
        np.testing.assert_array_equal(z['station_ids'], ids)
        m0 = dict(np.load(PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'/file.name))
        np.testing.assert_array_equal(z['hours'], m0['hours'])
        np.testing.assert_array_equal(z['truth'], m0['truth'])
        np.testing.assert_array_equal(z['population'], m0['population_F'] & m0['normal_support_S0'])
        samples = z['samples'].astype('f8')
        diagonal = z['analytic_mean'][:, None]+sample_diagonal(sigma, z['hours'], ids, 128, 42)
        shuffled = shuffle_members(samples, z['hours'], ids, 42)
        for name, members in [('correlated', samples), ('diagonal', diagonal), ('shuffled', shuffled)]:
            value, groups = joint_series(z['truth'], members, z['population'], metadata)
            series[name].append(value)
        for name, key in [('hours', 'hours'), ('truth', 'truth'), ('mean', 'analytic_mean'), ('population', 'population')]:
            arrays[name].append(z[key])
        arrays['F0'].append(m0['population_F'])
        emit(f'Aggregate {file.stem[-7:]}: named groups retained')
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    series = {k: np.concatenate(v) for k, v in series.items()}
    active = np.isfinite(series['correlated'][..., 0]).any(0)
    groups = list(np.array(groups)[active])
    series = {k: v[:, active] for k, v in series.items()}
    assert len(arrays['hours']) == 8760 and arrays['population'].sum() == 642258
    joint = {name: dict(ES_mm=float(np.nanmean(np.nanmean(value[..., 0], 0))),
                       VS_mm=float(np.nanmean(np.nanmean(value[..., 1], 0))), n_groups=len(groups),
                       n_group_hours=int(np.isfinite(value[..., 0]).sum())) for name, value in series.items()}
    intervals = {name: joint_intervals(arrays['hours'], series['correlated'], series[name]) for name in ('diagonal', 'shuffled')}
    marginal = evaluate_arrays(arrays['hours'], ids, arrays['truth'], arrays['mean'], arrays['F0'], arrays['population'],
                              std=np.broadcast_to(sigma, arrays['truth'].shape), metadata=metadata)
    seasonal = []
    months = pd.to_datetime(arrays['hours'], unit='h').month
    for label, selection in [('DJF', [12, 1, 2]), ('MAM', [3, 4, 5]), ('JJA', [6, 7, 8]), ('SON', [9, 10, 11])]:
        population = arrays['population'] & np.isin(months, selection)[:, None]
        seasonal.append(dict(season=label, point=point_metrics(arrays['truth'], arrays['mean'], population)))
    save_npz(root/'joint_scores.npz', hours=arrays['hours'], **series)
    result = dict(parameters=state, joint=joint, intervals=intervals, analytic_common_marginal=marginal,
                  seasonal=seasonal, n_hours=8760, n_station_hours=int(arrays['population'].sum()), groups=groups,
                  mean_source=read_json(root/'protocol.lock.json')['mean'], members=128,
                  inference='Full C2019 development; parameters fitted only on B2018; D2020 sealed',
                  observation_noise='Combined unresolved/observation surface error excluded from latent grid')
    write_json(root/'results.json', result)
    emit(str(joint))
    emit(str(intervals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    main(parser.parse_args().directory)
