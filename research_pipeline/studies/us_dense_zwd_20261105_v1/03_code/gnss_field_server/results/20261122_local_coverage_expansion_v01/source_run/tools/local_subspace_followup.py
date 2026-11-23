"""Independent-noise checks and a separately estimated mean-only counterfactual."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from local_subspace_data import ROOT, PilotData
from local_probe_diagnostics import emit, save_npz
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.artifacts import read_json, write_json, sha256, immutable_json
from gnss_field.models.subspace import SubspaceEDM
from gnss_field.models.diffusion import keyed_noise
from gnss_field.objectives.scores import energy_score
from gnss_field.evaluation import evaluate_arrays


def main():
    torch.set_num_threads(2)
    previous_pca = sha256(ROOT/'condition_pca.npz')
    data = PilotData()
    assert sha256(ROOT/'condition_pca.npz') == previous_pca
    selection = read_json(ROOT/'selection.lock.json')
    model = SubspaceEDM().to(data.device).eval()
    model.load_state_dict(torch.load(selection['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    immutable_json(ROOT/'followup.lock.json', dict(source=sha256(__file__), selected=sha256(ROOT/'selection.lock.json'),
        data=data.contract, independent_B_noise_seed=20261120, independent_mean_seed=20261121,
        mean_members=512, original_C_noise_seed=20261119,
        note='Post-selection audit; no checkpoint or parameter re-selection; no new scientific updates'))
    arrays = data.arrays(2018)
    indices = np.flatnonzero(data.development)
    scores = {name: [] for name in ('EDM', 'Gaussian')}
    with torch.no_grad():
        for start in range(0, len(indices), 4):
            ids = indices[start:start+4]
            for name in scores:
                samples = data.samples(model, arrays, ids, 64, seed=20261120, gaussian=name == 'Gaussian')
                for j, index in enumerate(ids):
                    scores[name].append(float(energy_score(samples[j:j+1], arrays['truth'][index:index+1], arrays['valid'][index:index+1])))
    b_result = {name: float(np.mean(value)) for name, value in scores.items()}
    emit(f'Independent B K64 check: {b_result}')
    c = data.arrays(2019)
    shift, mc_standard_error = [], []
    with torch.no_grad():
        for start in range(0, len(c['hours']), 4):
            selected = np.arange(start, min(start+4, len(c['hours'])))
            noise = keyed_noise(c['hours'][selected], 512, 20261121, shape=(64,), device=data.device)
            z = model.sample_noise(c['condition'][selected], noise)
            difference = ((z-noise)@c['projection'].T)*c['amplitude'][selected, None, None]
            shift.append(difference.mean(1).cpu().numpy())
            mc_standard_error.append((difference.std(1)/np.sqrt(512)).cpu().numpy())
    shift, mc_standard_error = np.concatenate(shift), np.concatenate(mc_standard_error)
    forecasts = dict(np.load(ROOT/'C2019_forecast.npz'))
    truth, population, mean = forecasts['truth'], forecasts['population'], forecasts['analytic_mean']
    mean_only = forecasts['reference']+shift[:, None]
    metadata = c['store'].stations.iloc[c['station_ids']].reset_index(drop=True)
    full = np.isfinite(truth) & c['store'].query_valid[c['station_ids']]
    series, metrics = {}, {}
    for name, members in [('EDM', forecasts['samples']), ('mean_only_Gaussian', mean_only), ('Gaussian', forecasts['reference'])]:
        metrics[name] = evaluate_arrays(c['hours'], c['station_ids'], truth, members.mean(1), full, population, samples=members, metadata=metadata)
        series[name], groups = joint_series(truth, members, population, metadata)
    intervals = {name: joint_intervals(c['hours'], series['EDM'], series[name]) for name in ('mean_only_Gaussian', 'Gaussian')}
    result = dict(independent_B_K64=b_result, metrics=metrics, joint_intervals=intervals,
        independent_mean_shift_rms_mm=float(np.sqrt(np.mean(shift[population]**2))),
        independent_shift_estimation_SE_rms_mm=float(np.sqrt(np.mean(mc_standard_error[population]**2))),
        interpretation='Gaussian is translated using independent generator-only draws; no labels used in estimating the shift',
        source_sha256=sha256(__file__))
    save_npz(ROOT/'mean_only_counterfactual.npz', shift=shift, estimation_standard_error=mc_standard_error, hours=c['hours'])
    write_json(ROOT/'followup_results.json', result)
    emit(f'Independent mean-only check: {[(k,v["joint"]["group_macro_es_mm"]) for k,v in metrics.items()]}; {intervals}')


if __name__ == '__main__':
    main()
