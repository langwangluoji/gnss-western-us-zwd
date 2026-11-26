"""Bounded actual-path EDM pilot, exact resume and B-only checkpoint selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from local_subspace_data import ROOT, PilotData
from local_probe_diagnostics import PROJECT, emit, save_npz, MEAN
from gnss_field.artifacts import immutable_json, read_json, sha256, write_json
from gnss_field.contracts import ScienceConfig
from gnss_field.models.subspace import SubspaceEDM
from gnss_field.objectives.scores import energy_score
from gnss_field.runtime import configure
from gnss_field.training.engine import StageSpec, TrainEngine, atomic_torch
from gnss_field.evaluation import evaluate_arrays, energy_score_ensemble
from gnss_field.evaluation.metrics import point_metrics
from local_probability_experiment import joint_series, joint_intervals


def science():
    return ScienceConfig(rank=64, condition_dim=32, sigma_data=1., learning_rate=1e-4,
        refinement_lr=1e-4, warmup=100, evaluate_every=100, startup_evaluation=0,
        log_every=100, minimum_refinement_steps=300, patience=3)


def build_engine(data, directory):
    arrays = data.arrays(2018)
    torch.manual_seed(42)
    model = SubspaceEDM().to(data.device)
    spec = StageSpec('observable_edm_refine', 800, 800, 8, 8, 42,
        dict(**data.contract, model_source=sha256(PROJECT/'src/gnss_field/models/subspace.py'),
             runner_source=sha256(__file__), sampler_steps=18, calibration='schedule-only fixed Gaussian gain'))
    engine = TrainEngine(model, directory, arrays['hours'][data.train], spec, science(), max_seconds=3600,
        pause_paths=[ROOT/'PAUSE'], batch_telemetry=lambda hours: dict(
            supervised_station_hours=int(arrays['valid'][np.searchsorted(arrays['hours'], hours)].sum())))
    def objective(hours, model, step):
        indices = np.searchsorted(arrays['hours'], hours)
        samples = data.samples(model, arrays, indices, 4)
        loss = energy_score(samples, arrays['truth'][indices], arrays['valid'][indices], scale=1.)
        return dict(loss=loss, observation_ES=loss)
    return engine, objective


def development(data, model, step):
    arrays = data.arrays(2018)
    indices = np.flatnonzero(data.development)
    values, mean, squared = [], [], []
    for start in range(0, len(indices), 8):
        selected = indices[start:start+8]
        samples = data.samples(model, arrays, selected, 16, seed=20261119)
        for j, index in enumerate(selected):
            valid = arrays['valid'][index]
            members = samples[j, :, valid].reshape(16, -1)
            # Training uses an unbiased U estimate; development shares the same
            # definition with fixed members, no diagonal finite-K penalty.
            values.append(float(energy_score(samples[j:j+1], arrays['truth'][index:index+1], arrays['valid'][index:index+1])))
        mean.append(samples.mean(1).cpu().numpy())
    prediction = np.concatenate(mean)
    valid = arrays['valid'][indices].cpu().numpy()
    metrics = point_metrics(arrays['truth'][indices].cpu().numpy(), prediction, valid)
    value = float(np.mean(values))
    emit(f'B incremental development step {step}: ES={value:.6f}, ensemble-mean RMSE={metrics["station_macro_rmse_mm"]:.6f}')
    return dict(joint=dict(value=value, ensemble_mean_rmse_mm=metrics['station_macro_rmse_mm'],
        population='B2018 held head-weather blocks; frozen baseline previously used these observations', members=16))


def equal_states(a, b, name):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), name
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), name
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for key in a:
            equal_states(a[key], b[key], f'{name}.{key}')
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), name
        for i, (first, second) in enumerate(zip(a, b)):
            equal_states(first, second, f'{name}.{i}')
    else:
        assert a == b, name


def preflight(data):
    path = ROOT/'preflight.json'
    if path.exists():
        return read_json(path)
    arrays = data.arrays(2018)
    model = SubspaceEDM().to(data.device).eval()
    with torch.no_grad():
        indices = np.array([0, 3, 40])
        reference = data.samples(model, arrays, indices, 8, seed=42, gaussian=True)
        prediction = data.samples(model, arrays, indices, 8, seed=42)
        maximum = float(abs(reference-prediction).max())
        assert maximum <= .001
        pieces = torch.cat([data.samples(model, arrays, indices[i:i+1], 8, seed=42) for i in range(3)])
        batch_error = float(abs(pieces-prediction).max())
        assert batch_error <= .001
        prefix = data.samples(model, arrays, indices, 4, seed=42)
        prefix_error = float(abs(prefix-prediction[:, :4]).max())
        assert prefix_error <= .001
    if data.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    continuous, objective = build_engine(data, ROOT/'engineering/continuous')
    continuous.run(objective, stop_after=200)
    expected = continuous.state()
    split, objective = build_engine(data, ROOT/'engineering/split')
    if split.step < 100:
        split.run(objective, stop_after=100)
    resumed, objective = build_engine(data, ROOT/'engineering/split')
    resumed.run(objective, stop_after=200)
    actual = resumed.state()
    for key in ('model', 'ema', 'optimizers', 'rng', 'cursor', 'step'):
        equal_states(expected[key], actual[key], key)
    assert expected['model']['net.4.weight'].abs().sum() > 0
    result = dict(passed=True, maximum_initial_Gaussian_error_mm=maximum,
        maximum_batch_error_mm=batch_error, maximum_prefix_error_mm=prefix_error,
        gaussian_heun_gain=float(model.gaussian_gain), real_100_plus100_equals200=True,
        engineering_updates=400, benchmark_seconds=time.monotonic()-started,
        estimated_seconds_per_update=(time.monotonic()-started)/400,
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
    write_json(path, result)
    emit(f'Real sampler/update/resume preflight PASS: {result}')
    return result


def evaluate_c(data, model, selected):
    a = data.arrays(2019)
    ids, store = a['station_ids'], a['store']
    truth, population = a['truth'].cpu().numpy(), a['valid'].cpu().numpy()
    mean = a['mean'].cpu().numpy()
    meta = store.stations.iloc[ids].reset_index(drop=True)
    full = np.isfinite(truth) & store.query_valid[ids]
    samples, reference = [], []
    with torch.no_grad():
        for start in range(0, len(a['hours']), 4):
            indices = np.arange(start, min(start+4, len(a['hours'])))
            samples.append(data.samples(model, a, indices, 128, seed=20261119).cpu().numpy())
            reference.append(data.samples(model, a, indices, 128, seed=20261119, gaussian=True).cpu().numpy())
            if start % 48 == 0:
                emit(f'C2019 paired EDM samples {start}/{len(a["hours"])}')
    samples, reference = np.concatenate(samples), np.concatenate(reference)
    result = {}
    for k in (64, 128):
        metrics, series = {}, {}
        for name, members in [('EDM', samples[:, :k]), ('conditional_Gaussian', reference[:, :k])]:
            average = members.mean(1)
            metrics[name] = evaluate_arrays(a['hours'], ids, truth, average, full, population, samples=members, metadata=meta)
            series[name], groups = joint_series(truth, members, population, meta)
        intervals = joint_intervals(a['hours'], series['EDM'], series['conditional_Gaussian'])
        drift = samples[:, :k].mean(1)-reference[:, :k].mean(1)
        result[str(k)] = dict(metrics=metrics, joint_intervals=intervals,
            paired_mean_drift_rms_mm=float(np.sqrt(np.mean(drift[population]**2))),
            paired_mean_drift_bias_mm=float(drift[population].mean()),
            note='Matched factorization and keyed noise. Point product remains analytic mixture; generated mean reported separately.')
        emit(f'C2019 K{k}: EDM ES {metrics["EDM"]["joint"]["group_macro_es_mm"]:.6f}; Gaussian {metrics["conditional_Gaussian"]["joint"]["group_macro_es_mm"]:.6f}; paired {intervals}')
    save_npz(ROOT/'C2019_forecast.npz', hours=a['hours'], station_ids=ids, truth=truth, population=population,
             samples=samples, reference=reference, analytic_mean=mean)
    write_json(ROOT/'C2019_results.json', dict(selected=selected, results=result, D2020='sealed'))


def main():
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    data = PilotData()
    arrays = data.arrays(2018)
    immutable_json(ROOT/'pilot.lock.json', dict(dependencies=data.contract, science=science(), seed=42,
        training_hours=arrays['hours'][data.train], development_hours=arrays['hours'][data.development],
        maximum_scientific_updates=800, effective_hours=8, train_members=4,
        initialization='analytic Gaussian, output layer zero; fixed schedule-only gain',
        selection='B incremental held-time ES U estimate K16; zero-update candidate retained', D2020='sealed'))
    preflight(data)
    engine, objective = build_engine(data, ROOT/'scientific')
    if not (engine.directory/'initial.pt').exists():
        with torch.no_grad():
            initial = development(data, engine.ema, 0)
        initial_path = engine.directory/'initial.pt'
        engine.products['joint'] = [dict(**initial['joint'], step=0, checkpoint=str(initial_path.resolve()))]
        atomic_torch(initial_path, engine.state())
        write_json(engine.directory/'initial_metrics.json', initial)
    engine.run(objective, evaluator=lambda model, step: development(data, model, step))
    selected = min(engine.products['joint'], key=lambda r: (r['value'], r['step']))
    selected = dict(**selected, sha256=sha256(selected['checkpoint']))
    immutable_json(ROOT/'selection.lock.json', selected)
    model = SubspaceEDM().to(data.device).eval()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    assert sha256(MEAN) == data.data.mean_hash
    if not (ROOT/'C2019_results.json').exists():
        evaluate_c(data, model, selected)
    write_json(ROOT/'status.json', dict(status='COMPLETE', selected_step=selected['step'], final_step=engine.step,
        M0_unchanged=True, D2020='sealed', server_training_started=False))
    emit('Local EDM pilot complete')


if __name__ == '__main__':
    main()
