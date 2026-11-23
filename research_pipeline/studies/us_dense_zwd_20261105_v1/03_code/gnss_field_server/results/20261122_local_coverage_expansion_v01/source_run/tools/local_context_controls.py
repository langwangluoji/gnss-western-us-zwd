"""Inner-selected linear and scalar controls for the same grid publication path."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import torch
from local_context_data import RUN
from local_context_train import StudyData
from local_probe_diagnostics import save_npz
from gnss_field.artifacts import immutable_json, write_json, read_json, sha256
from gnss_field.runtime import configure


def ridge_statistics(design, residual, population):
    x = np.asarray(design, 'f8')[population]
    y = np.asarray(residual, 'f8')[population]
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid linear-control training observations')
    return x.T@x, x.T@y, len(y)


def query_features(data, indices):
    # A linear head commutes with interpolation. Each channel is still defined
    # at grid vertices, then queried with the registered height restoration.
    values = data.features_for(indices).cpu().numpy().astype('f8')
    bias = np.ones((*values.shape[:-1], 1))
    values = np.concatenate([values, bias], -1)
    weight = data.a['query_weights']
    result = (values[:, data.a['query_indices']]*weight[None, :, :, None]).sum(2)
    return result*data.settings['grid_scale']*data.a['amplitude'][indices, None, None]*data.a['gate'][indices, None, None]


def predict_linear(data, indices, coefficient):
    result = []
    for start in range(0, len(indices), 8):
        ix = indices[start:start+8]
        result.append(data.a['mean'][ix]+query_features(data, ix)@coefficient)
    return np.concatenate(result).astype('f4')


def fit_controls(data, fold):
    directory = RUN/f'fold_{fold}'/'linear_control'
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory/'contract.json', dict(source=sha256(__file__),
        dataset=sha256(data.root/'complete.json'), fold=fold,
        alphas=[.0001, .001, .01, .1, 1.], selection='inner ES K16 with zero and scalar candidates',
        labels='core B fitted groups only', C2019=False, D2020=False))
    fit = data.a['folds'] != fold
    core, inner, outer = (data.a[k] for k in ('core', 'inner', 'outer'))
    normal, rhs, count = np.zeros((100, 100)), np.zeros(100), 0
    for start in range(0, len(core), 8):
        ix = core[start:start+8]
        x = query_features(data, ix)
        pop = data.a['valid'][ix] & fit[None]
        n, r, c = ridge_statistics(x, data.a['truth'][ix]-data.a['mean'][ix], pop)
        normal += n; rhs += r; count += c
    candidates = [dict(name='zero', coefficient=np.zeros(100))]
    for alpha in (.0001, .001, .01, .1, 1.):
        candidates.append(dict(name=f'ridge_{alpha}', coefficient=np.linalg.solve(
            normal/count+alpha*np.eye(100), rhs/count)))
    metadata = pd.read_csv(data.root/'stations.csv')
    slope = np.exp(-metadata.height.to_numpy()/3000)
    pop = data.a['valid'][core] & fit[None]
    error = np.where(pop, data.a['truth'][core]-data.a['mean'][core], 0.)
    weights = pop/np.maximum(pop.sum(0), 1)
    scalar = float(np.sum(weights*error*slope)/np.sum(weights*slope**2))
    scored = []
    for candidate in candidates:
        prediction = predict_linear(data, inner, candidate['coefficient'])
        metrics = data.metrics(prediction, inner, fit)
        scored.append(dict(name=candidate['name'], coefficient=candidate['coefficient'],
            inner_ES=metrics['ES_mm'], inner_RMSE=metrics['station_macro_rmse_mm']))
    scalar_prediction = data.a['mean'][inner]+scalar*slope
    metrics = data.metrics(scalar_prediction.astype('f4'), inner, fit)
    scored.append(dict(name='scalar_Z0', delta_Z0_mm=scalar, inner_ES=metrics['ES_mm'],
        inner_RMSE=metrics['station_macro_rmse_mm']))
    selected = min(scored, key=lambda r:(r['inner_ES'], r['name']))
    immutable_json(directory/'selection.lock.json', dict(candidates=scored, selected=selected,
        rule='Unique inner B ES selection; outer labels unused'))
    if selected['name'] == 'scalar_Z0':
        prediction = (data.a['mean'][outer]+scalar*slope).astype('f4')
    else:
        prediction = predict_linear(data, outer, np.asarray(selected['coefficient']))
    held = ~fit
    metrics = data.metrics(prediction, outer, held, members=64, seed=20261118)
    save_npz(directory/'outer_predictions.npz', hours=data.a['hours'][outer], mean=prediction,
        truth=data.a['truth'][outer], population=data.a['valid'][outer] & held[None])
    result = dict(fold=fold, selected=selected, outer_B=metrics, training_station_hours=count)
    write_json(directory/'results.json', result)
    print('LINEAR_CONTROL', fold, selected['name'], metrics['station_macro_rmse_mm'], flush=True)
    return result


def main():
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    data = StudyData()
    rows = [fit_controls(data, fold) for fold in range(2)]
    write_json(RUN/'controls_complete.json', dict(status='COMPLETE', folds=rows, C2019=False, D2020=False))


if __name__ == '__main__':
    main()
