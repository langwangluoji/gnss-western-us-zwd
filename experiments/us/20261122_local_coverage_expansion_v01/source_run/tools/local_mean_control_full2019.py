"""Annual fixed-candidate mean-only comparison using cached common members."""
import numpy as np
import torch
from local_subspace_data import ROOT, PilotData
from local_probe_diagnostics import emit, save_npz
from local_probability_experiment import joint_series, joint_intervals
from local_student_field import macro
from gnss_field.artifacts import read_json, immutable_json, write_json, sha256
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.evaluation import point_metrics, crps_ensemble


def main():
    torch.set_num_threads(2)
    data = PilotData()
    directory = ROOT/'mean_shift_control_v02'
    selected = read_json(directory/'selection.lock.json')
    immutable_json(directory/'annual_protocol.lock.json', dict(source=sha256(__file__),
        checkpoint=sha256(selected['checkpoint']), condition_cache=sha256(ROOT/'full2019/A_conditions.npz'),
        purpose='Fixed B-selected mean-control annual comparison; no C reselection', members=128, D2020='sealed'))
    model = SubspaceMeanShift().to(data.device).eval()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    c = data.arrays(2019)
    cache = dict(np.load(ROOT/'full2019/A_conditions.npz'))
    wet = data.data.development.context_values(cache['hours'])
    from local_conditional_gaussian import wet_feature
    wet, present = wet_feature(wet)
    a = data.amplitude
    g = np.exp(a['a']+a['b']*np.where(present, (wet-a['A_wetness_center'])/a['A_wetness_std'], 0.))
    with torch.no_grad():
        shift = (model.net(torch.tensor(cache['condition'], device=data.device))@c['projection'].T).cpu().numpy()*g[:, None]
    meta = c['store'].stations.iloc[c['station_ids']].reset_index(drop=True)
    records = sorted((ROOT/'full2019').glob('forecast_2019-*.npz'))
    annual = dict(np.load(ROOT/'full2019/annual_means.npz'))
    truth, valid = annual['truth'], annual['population']
    scores = {k: [] for k in ('mean_control', 'EDM', 'Gaussian')}
    means, crps = [], []
    start = 0
    for path in records:
        z = dict(np.load(path))
        stop = start+len(z['hours'])
        members = z['reference']+shift[start:stop, None]
        for name, samples in [('mean_control', members), ('EDM', z['samples']), ('Gaussian', z['reference'])]:
            s, groups = joint_series(truth[start:stop], samples, valid[start:stop], meta)
            scores[name].append(s)
        means.append(members.mean(1))
        value = np.zeros(valid[start:stop].shape)
        value[valid[start:stop]] = crps_ensemble(truth[start:stop][valid[start:stop]], members.transpose(0, 2, 1)[valid[start:stop]])
        crps.append(value)
        start = stop
    scores = {k: np.concatenate(v) for k, v in scores.items()}
    active = np.isfinite(scores['mean_control'][..., 0]).any(0)
    scores = {k: v[:, active] for k, v in scores.items()}
    joint = {k: dict(ES_mm=float(np.nanmean(np.nanmean(v[..., 0], 0))), VS_mm=float(np.nanmean(np.nanmean(v[..., 1], 0)))) for k, v in scores.items()}
    intervals = {name: joint_intervals(cache['hours'], scores['mean_control'], scores[name]) for name in ('EDM', 'Gaussian')}
    # Exact conditional Gaussian mean is available for this control.
    result = dict(selected=selected, joint=joint, intervals=intervals,
        exact_point=point_metrics(truth, annual['analytic_mean']+shift, valid),
        ensemble_point=point_metrics(truth, np.concatenate(means), valid),
        station_macro_CRPS_mm=macro(np.concatenate(crps), valid),
        interpretation='Mean-only Gaussian has exact analytic point product; compare EDM finite means separately', D2020='sealed')
    save_npz(directory/'annual_scores.npz', hours=cache['hours'], **scores)
    save_npz(directory/'annual_shift.npz', hours=cache['hours'], shift=shift)
    write_json(directory/'annual_results.json', result)
    emit(str(result))


if __name__ == '__main__':
    main()
