"""Single-configuration, physical-coefficient linear probes with exact resume."""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from local_probe_diagnostics import (OUTPUT, MEAN, ProbeData, emit, save_npz,
                                     score_arrays, split_head_hours)
from gnss_field.data.manifests import stratified_hours
from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json
from gnss_field.contracts import ScienceConfig
from gnss_field.geometry import basis_observation_matrix, surface_query, coarsen_100km
from gnss_field.objectives import masked_mse
from gnss_field.runtime import configure
from gnss_field.training.engine import TrainEngine, StageSpec, atomic_torch
from gnss_field.training.prediction import safe_query_geometry


class LinearCoefficientProbe(nn.Module):
    """Outputs actual c, so a zero output is exactly the frozen M0 field."""
    def __init__(self, input_dim, rank=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, rank)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        return self.linear(features)

    def station_prediction(self, features, mean, observation_matrix, scale):
        return mean+scale*(self(features)@observation_matrix.T)


def observation_objective(model, features, base, target, valid, matrix, curvature_gram, scale):
    c = model(features)
    prediction = base+scale*(c@matrix.T)
    mse = masked_mse(prediction, target, valid, scale)
    coefficient = c.square().mean()
    curvature = torch.einsum('bi,bij,bj->b', c, curvature_gram, c).mean()
    return dict(loss=mse+coefficient+.001*curvature,
                mean_loss=mse, coefficient_loss=coefficient, curvature_loss=curvature)


def scientific_configuration():
    return ScienceConfig(learning_rate=1e-4, refinement_lr=1e-4, coefficient_penalty=1., warmup=100,
                         evaluate_every=200, startup_evaluation=100, log_every=100,
                         minimum_refinement_steps=600, patience=3)


def build_training(data, fold, directory, science, *, tensors=None):
    arrays = data.arrays(2018)
    train, development = split_head_hours(arrays['hours'])
    phi, digest = data.basis(fold)
    bxy, bh = safe_query_geometry(data.store, data.store.ids('B'))
    matrix = basis_observation_matrix(phi, data.store.shape, bxy, bh)
    fit = data.assignment != fold
    train_indices = np.searchsorted(arrays['hours'], train)
    center = arrays['features'][train_indices].mean(0, dtype='f8').astype('f4')
    deviation = arrays['features'][train_indices].std(0, dtype='f8').clip(1e-6).astype('f4')
    features = (arrays['features']-center)/deviation
    if tensors is None:
        tensors = {name: torch.as_tensor(value, device=data.device) for name, value in dict(
            features=features, baseline=arrays['baseline'], truth=arrays['truth'],
            valid=arrays['valid'] & fit,
            curvature=np.load(OUTPUT/f'curvature_gram_fold_{fold}.npy')).items()}
    dependencies = dict(data.contract, basis=digest, training_hours=train.tolist(),
                        development_hours=development.tolist(), fit=fit,
                        feature_sha256=fingerprint(dict(center=center, std=deviation)),
                        source_sha256=sha256(__file__), feature_schema='A_10_flat_coarse60_pool4x4_cycles4_M0_pool4x4')
    spec = StageSpec('linear_probe_refine', 1000, 1000, 32, 32, 42, dependencies)
    model = LinearCoefficientProbe(features.shape[1]).to(data.device)
    engine = TrainEngine(model, directory, train, spec, science,
                         pause_paths=[OUTPUT/'PAUSE'], max_seconds=3600)

    def objective(hours, current, step):
        idx = torch.as_tensor(np.searchsorted(arrays['hours'], hours), device=data.device)
        return observation_objective(current, tensors['features'][idx], tensors['baseline'][idx],
                                     tensors['truth'][idx], tensors['valid'][idx], matrix,
                                     tensors['curvature'][idx], data.scale)

    def evaluator(current, step):
        idx = np.searchsorted(arrays['hours'], development)
        prediction = current.station_prediction(tensors['features'][idx], tensors['baseline'][idx],
                                                  matrix, data.scale).detach().cpu().numpy()
        metrics = score_arrays(data.store, 'B', development, arrays['truth'][idx], prediction,
                               arrays['valid'][idx] & fit)
        return {'point': dict(value=metrics['primary']['station_macro_rmse_mm'],
                              mae_mm=metrics['primary']['station_macro_mae_mm'],
                              bias_mm=metrics['primary']['station_macro_bias_mm'],
                              population='2018 fitted B on held head-time blocks; Phi previously used these labels')}

    return engine, objective, evaluator, matrix, center, deviation, tensors


def verify_real_resume(data, science):
    folder = OUTPUT/'resume_validation'
    engine, objective, _, _, _, _, tensors = build_training(data, 0, folder/'continuous', science)
    engine.run(objective, stop_after=200)
    expected = engine.state()
    split, loss, *_ = build_training(data, 0, folder/'split', science, tensors=tensors)
    if split.step < 100:
        split.run(loss, stop_after=100)
    resumed, loss, *_ = build_training(data, 0, folder/'split', science, tensors=tensors)
    resumed.run(loss, stop_after=200)
    actual = resumed.state()
    for key in ('model', 'ema'):
        for name, value in expected[key].items():
            assert torch.equal(value, actual[key][name]), (key, name)
    assert np.array_equal(expected['cursor']['order'], actual['cursor']['order'])
    assert expected['cursor']['position'] == actual['cursor']['position']
    for first, second in zip(expected['optimizers'], actual['optimizers']):
        for index, record in first['state'].items():
            for name, value in record.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, second['state'][index][name])
    write_json(folder/'verification.json', dict(passed=True, continuous=200, split=[100, 100],
               model=True, ema=True, optimizer=True, cursor=True,
               interpretation='Engineering validation updates are separate from each 1000-step scientific fit'))
    emit('Real-data 100+resume100 equals continuous200: PASS')


def evaluate_selected(data, fold, engine, center, deviation, matrix, evaluator):
    directory = engine.directory
    initial = read_json(directory/'initial_metrics.json')
    candidates = [dict(value=initial['point']['value'], step=0, checkpoint=str(directory/'initial.pt'))]
    candidates += engine.products.get('point', [])
    selected = min(candidates, key=lambda r: (r['value'], r['step']))
    lock = dict(selected=selected, rule='2018 fitted-B head-time development station-macro RMSE; zero correction eligible',
                source_checkpoint_sha256=sha256(selected['checkpoint']))
    immutable_json(directory/'selection.lock.json', lock)
    model = LinearCoefficientProbe(len(center)).to(data.device).eval()
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    phi, digest = data.basis(fold)
    results = {}
    for year in (2018, 2019):
        arrays = data.arrays(year)
        role, store = ('B', data.store) if year == 2018 else ('C', data.development)
        xy, heights = safe_query_geometry(store, store.ids(role))
        h = basis_observation_matrix(phi, store.shape, xy, heights)
        hc = surface_query(coarsen_100km(phi.T.reshape(128, 1, *store.shape)), xy, heights)[:, 0].T
        with torch.no_grad():
            features = torch.tensor((arrays['features']-center)/deviation, device=data.device)
            coefficients = model(features)
            prediction = arrays['baseline']+data.scale*(coefficients@h.T).cpu().numpy()
            coarse = arrays['coarse']+data.scale*(coefficients@hc.T).cpu().numpy()
        valid = arrays['valid'].copy()
        if year == 2018:
            valid &= data.assignment == fold
        metrics = score_arrays(store, role, arrays['hours'], arrays['truth'], prediction, valid, coarse)
        baseline = score_arrays(store, role, arrays['hours'], arrays['truth'], arrays['baseline'], valid, arrays['coarse'])
        idw = score_arrays(store, role, arrays['hours'], arrays['truth'], arrays['idw'], valid)
        results[str(year)] = dict(model=metrics, M0=baseline, A_IDW=idw)
        save_npz(directory/f'forecast_{year}.npz', hours=arrays['hours'], station_ids=arrays['station_ids'],
                 truth=arrays['truth'], population=valid, prediction=prediction, coarsened_prediction=coarse,
                 baseline=arrays['baseline'], idw=arrays['idw'])
    write_json(directory/'results.json', dict(fold=fold, selection=lock, results=results))
    emit(f'fold {fold} selected step {selected["step"]}; C2019 RMSE {results["2019"]["model"]["primary"]["station_macro_rmse_mm"]:.4f}')
    return dict(fold=fold, selection=lock, results=results)


def plots(rows):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for fold, ax in enumerate(axes):
        path = OUTPUT/f'fold_{fold}/development.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        ax.plot([r['step'] for r in records], [r['metrics']['point']['value'] for r in records], marker='o')
        initial = read_json(OUTPUT/f'fold_{fold}/initial_metrics.json')['point']['value']
        ax.axhline(initial, color='gray', linestyle='--', label='Zero correction')
        ax.set(title=f'Fold {fold+1}, fitted-B head-time development', xlabel='Update', ylabel='Station-macro RMSE (mm)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'training_curves.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(3)
    for j, (name, color) in enumerate([('M0', '#888888'), ('A_IDW', '#c2721c'), ('model', '#246c9a')]):
        values = [r['results']['2019'][name]['primary']['station_macro_rmse_mm'] for r in rows]
        bars = ax.bar(x+(j-1)*.23, values, width=.23, color=color, label=name)
        ax.bar_label(bars, fmt='%.3f', fontsize=8)
    ax.set(xticks=x, xticklabels=['Frozen basis 1', 'Frozen basis 2', 'Frozen basis 3'],
           ylabel='C2019 subset station-macro RMSE (mm)')
    ax.legend()
    fig.savefig(OUTPUT/'development_2019_comparison.png', dpi=170)
    plt.close(fig)


def main():
    started = time.monotonic()
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    data = ProbeData()
    science = scientific_configuration()
    arrays = data.arrays(2018)
    train, development = split_head_hours(arrays['hours'])
    immutable_json(OUTPUT/'probe.lock.json', dict(
        source_sha256=sha256(__file__), helpers_sha256=sha256(Path(__file__).with_name('local_probe_diagnostics.py')),
        science=science, training_hours=train, head_development_hours=development,
        future_evaluation_hours=stratified_hours(np.array(data.development.hours), 2019, per_cell=1), maximum_updates_per_fold=1000,
        covariance_model=None, physical_coefficient_output=True,
        interpretation='Exploratory local condition probe; existing basis previously used fitted B head-development labels'))
    if not (OUTPUT/'resume_validation/verification.json').exists():
        verify_real_resume(data, science)
    rows = []
    for fold in range(3):
        folder = OUTPUT/f'fold_{fold}'
        if (folder/'results.json').exists():
            rows.append(read_json(folder/'results.json'))
            continue
        engine, objective, evaluator, matrix, center, deviation, tensors = build_training(data, fold, folder, science)
        if not (folder/'initial.pt').exists():
            assert engine.step == 0
            with torch.no_grad():
                actual = engine.model.station_prediction(tensors['features'][:16], tensors['baseline'][:16], matrix, data.scale)
                assert torch.equal(actual, tensors['baseline'][:16])
            write_json(folder/'initial_metrics.json', evaluator(engine.ema, 0))
            atomic_torch(folder/'initial.pt', engine.state())
            save_npz(folder/'feature_normalization.npz', center=center, std=deviation)
        if engine.step < 100:
            engine.run(objective, evaluator=evaluator, stop_after=100)
            state = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
            assert state['step'] == 100
            assert any(p.abs().sum() > 0 for p in state['model'].values())
            emit(f'fold {fold}: real weight update and 100-step checkpoint verified')
            del engine
            engine, objective, evaluator, matrix, center, deviation, _ = build_training(data, fold, folder, science, tensors=tensors)
        result = engine.run(objective, evaluator=evaluator)
        if result['status'] != 'COMPLETE':
            raise RuntimeError(f'Probe paused at {engine.step}; retained checkpoint')
        rows.append(evaluate_selected(data, fold, engine, center, deviation, matrix, evaluator))
    write_json(OUTPUT/'summary.json', dict(status='COMPLETE', folds=rows, elapsed_seconds=time.monotonic()-started,
               mean_sha256_after=sha256(MEAN), mean_sha256_before=data.mean_hash))
    assert sha256(MEAN) == data.mean_hash
    plots(rows)
    emit('All three local linear probes COMPLETE')


if __name__ == '__main__':
    main()
