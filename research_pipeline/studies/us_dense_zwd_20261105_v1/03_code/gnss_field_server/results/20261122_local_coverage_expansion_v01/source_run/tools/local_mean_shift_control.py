"""Matched conditional mean Gaussian control; same B split, ES and budget."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from local_subspace_data import ROOT, PilotData
from local_subspace_pilot import science, development
from local_probe_diagnostics import PROJECT, emit, save_npz
from local_probability_experiment import joint_series, joint_intervals
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.training.engine import StageSpec, TrainEngine, atomic_torch
from gnss_field.objectives.scores import energy_score
from gnss_field.evaluation import evaluate_arrays
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256


def main():
    torch.set_num_threads(2)
    data = PilotData()
    a = data.arrays(2018)
    directory = ROOT/'mean_shift_control_v02'
    directory.mkdir(exist_ok=True)
    immutable_json(directory/'protocol.lock.json', dict(source=sha256(__file__), data=data.contract,
        comparison='same allowed conditions, projection, Gaussian covariance, K4 ES, learning rate, EMA, hours, budget; mean-only head',
        note='Post-EDM attribution experiment; predeclared now before its fit, not an independent C confirmation',
        seed=42, maximum_updates=800, selected_on='B incremental K16 ES, zero included', D2020='sealed'))
    torch.manual_seed(42)
    model = SubspaceMeanShift().to(data.device)
    spec = StageSpec('observable_mean_shift_refine', 800, 800, 8, 8, 42,
        dict(**data.contract, runner_source=sha256(__file__), model_source=sha256(PROJECT/'src/gnss_field/models/mean_shift.py')))
    engine = TrainEngine(model, directory, a['hours'][data.train], spec, science(), max_seconds=1800,
        batch_telemetry=lambda hours: dict(supervised_station_hours=int(a['valid'][np.searchsorted(a['hours'], hours)].sum())))
    def objective(hours, model, step):
        indices = np.searchsorted(a['hours'], hours)
        samples = data.samples(model, a, indices, 4)
        loss = energy_score(samples, a['truth'][indices], a['valid'][indices])
        return dict(loss=loss, observation_ES=loss)
    if not (directory/'initial.pt').exists():
        with torch.no_grad():
            initial = development(data, engine.ema, 0)
        checkpoint = directory/'initial.pt'
        engine.products['joint'] = [dict(**initial['joint'], step=0, checkpoint=str(checkpoint.resolve()))]
        atomic_torch(checkpoint, engine.state())
        write_json(directory/'initial_metrics.json', initial)
    engine.run(objective, evaluator=lambda model, step: development(data, model, step))
    selected = min(engine.products['joint'], key=lambda r: (r['value'], r['step']))
    immutable_json(directory/'selection.lock.json', dict(**selected, sha256=sha256(selected['checkpoint'])))
    model = SubspaceMeanShift().to(data.device).eval()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    c = data.arrays(2019)
    forecasts = dict(np.load(ROOT/'C2019_forecast.npz'))
    with torch.no_grad():
        shift = (model.net(c['condition'])@c['projection'].T)*c['amplitude'][:, None]
        shift = shift.cpu().numpy()
    samples = forecasts['reference']+shift[:, None]
    meta = c['store'].stations.iloc[c['station_ids']].reset_index(drop=True)
    full = np.isfinite(forecasts['truth']) & c['store'].query_valid[c['station_ids']]
    metrics, series = {}, {}
    for name, members in [('mean_shift_Gaussian', samples), ('EDM', forecasts['samples']), ('Gaussian', forecasts['reference'])]:
        metrics[name] = evaluate_arrays(c['hours'], c['station_ids'], forecasts['truth'], members.mean(1), full,
            forecasts['population'], samples=members, metadata=meta)
        series[name], groups = joint_series(forecasts['truth'], members, forecasts['population'], meta)
    intervals = {name: joint_intervals(c['hours'], series['mean_shift_Gaussian'], series[name]) for name in ('EDM', 'Gaussian')}
    save_npz(directory/'forecast_C2019.npz', hours=c['hours'], shift=shift, samples=samples)
    write_json(directory/'results.json', dict(selected=selected, metrics=metrics, intervals=intervals,
        shift_RMS_mm=float(np.sqrt(np.mean(shift[forecasts['population']]**2))), D2020='sealed'))
    emit(f'Mean-control C2019: '+str({name: value['joint'] for name, value in metrics.items()}))


if __name__ == '__main__':
    main()
