"""Independent latent-mean integration for the locked annual EDM forecast."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from local_subspace_data import ROOT, PilotData
from local_probe_diagnostics import emit, save_npz
from local_conditional_gaussian import wet_feature
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256
from gnss_field.models.diffusion import keyed_noise
from gnss_field.models.subspace import SubspaceEDM
from gnss_field.evaluation import point_metrics


def main():
    torch.set_num_threads(2)
    data = PilotData()
    selection = read_json(ROOT/'selection.lock.json')
    c = data.arrays(2019)
    cache = dict(np.load(ROOT/'full2019/A_conditions.npz'))
    wet, present = wet_feature(data.data.development.context_values(cache['hours']))
    a = data.amplitude
    g = np.exp(a['a']+a['b']*np.where(present, (wet-a['A_wetness_center'])/a['A_wetness_std'], 0.))
    model = SubspaceEDM().to(data.device).eval()
    model.load_state_dict(torch.load(selection['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    immutable_json(ROOT/'full2019/attribution.lock.json', dict(source=sha256(__file__), selected=sha256(ROOT/'selection.lock.json'),
        condition_cache=sha256(ROOT/'full2019/A_conditions.npz'), mean_pool_seed=20261121, mean_pool_members=128,
        inference='New latent-noise pool independent of forecast pool; paired EDM-minus-initial draws; no labels used to estimate location'))
    shifts, errors, shifts64 = [], [], []
    with torch.no_grad():
        for start in range(0, len(cache['hours']), 8):
            stop = min(start+8, len(cache['hours']))
            noise = keyed_noise(cache['hours'][start:stop], 128, 20261121, (64,), device=data.device)
            condition = torch.tensor(cache['condition'][start:stop], device=data.device)
            delta = (model.sample_noise(condition, noise)-noise)@c['projection'].T
            delta *= torch.tensor(g[start:stop], dtype=torch.float32, device=data.device)[:, None, None]
            shifts.append(delta.mean(1).cpu().numpy())
            shifts64.append(delta[:, :64].mean(1).cpu().numpy())
            errors.append((delta.std(1)/np.sqrt(128)).cpu().numpy())
            if start % 720 == 0:
                emit(f'Annual independent mean integration {start}/{len(cache["hours"])}')
    shift, shift64, se = np.concatenate(shifts), np.concatenate(shifts64), np.concatenate(errors)
    annual = dict(np.load(ROOT/'full2019/annual_means.npz'))
    truth, pop = annual['truth'], annual['population']
    metadata = c['store'].stations.iloc[c['station_ids']].reset_index(drop=True)
    names = ('Gaussian', 'mean_only_Gaussian', 'EDM', 'shape_only_EDM')
    series = {name: [] for name in names}
    start = 0
    for path in sorted((ROOT/'full2019').glob('forecast_2019-*.npz')):
        z = dict(np.load(path))
        stop = start+len(z['hours'])
        for name, samples in [('Gaussian', z['reference']), ('mean_only_Gaussian', z['reference']+shift[start:stop, None]),
                              ('EDM', z['samples']), ('shape_only_EDM', z['samples']-shift[start:stop, None])]:
            score, groups = joint_series(truth[start:stop], samples, pop[start:stop], metadata)
            series[name].append(score)
        start = stop
    series = {k: np.concatenate(v) for k, v in series.items()}
    active = np.isfinite(series['EDM'][..., 0]).any(0)
    series = {k: v[:, active] for k, v in series.items()}
    joint = {k: dict(ES_mm=float(np.nanmean(np.nanmean(v[..., 0], 0))), VS_mm=float(np.nanmean(np.nanmean(v[..., 1], 0)))) for k, v in series.items()}
    intervals = dict(EDM_minus_mean_only=joint_intervals(cache['hours'], series['EDM'], series['mean_only_Gaussian']),
                     shape_only_minus_Gaussian=joint_intervals(cache['hours'], series['shape_only_EDM'], series['Gaussian']))
    mean128, mean64 = annual['analytic_mean']+shift, annual['analytic_mean']+shift64
    result = dict(joint=joint, intervals=intervals,
        independent_integrated_mean_point=point_metrics(truth, mean128, pop),
        independent_K64_mean_point=point_metrics(truth, mean64, pop),
        shift_estimation_SE_RMS_mm=float(np.sqrt(np.mean(se[pop]**2))),
        maximum_primary_shift_SE_mm=float(se[pop].max()),
        interpretation='Forecast-independent mean estimate; residual shape control is diagnostic; no new fit, no C reselection', D2020='sealed')
    save_npz(ROOT/'full2019/attribution_scores.npz', hours=cache['hours'], **series)
    save_npz(ROOT/'full2019/integrated_mean.npz', hours=cache['hours'], shift=shift, mean_estimation_SE=se)
    write_json(ROOT/'full2019/attribution_results.json', result)
    emit(str(result))


if __name__ == '__main__':
    main()
