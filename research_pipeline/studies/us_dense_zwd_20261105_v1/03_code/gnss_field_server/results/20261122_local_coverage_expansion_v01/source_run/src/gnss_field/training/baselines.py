"""A-only deployment baselines; geostatistical parameters use training B only."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares

from gnss_field.artifacts import fingerprint, immutable_json, read_json
from gnss_field.geometry import input_geometry, surface_query, coarsen_100km, query_grid
from .prediction import safe_query_geometry


def ordinary_kriging(xy, values, queries, parameters):
    if not len(xy):
        raise ValueError('Kriging needs input observations or an explicit fallback')
    if len(xy) == 1:
        return np.full(len(queries), values[0], dtype='f8')
    sill, length, nugget = (parameters[k] for k in ('sill', 'range_km', 'nugget'))
    cov = sill*np.exp(-cdist(xy, xy)/(length*1000))
    cov.flat[::len(cov)+1] += max(nugget, 1e-8*sill)
    factor = cho_factor(cov, lower=True, check_finite=True)
    one = cho_solve(factor, np.ones(len(xy)))
    mean = float(one@values/one.sum())
    alpha = cho_solve(factor, values-mean)
    return mean+sill*np.exp(-cdist(queries, xy)/(length*1000))@alpha


def fit_geostat(store, destination):
    if store.purpose != 'train':
        raise PermissionError('Covariance estimation requires training B view')
    path = Path(destination)
    contract = fingerprint(dict(training=store.manifest['training_hash'], model='ordinary_exponential_v1'))
    if path.exists():
        data = read_json(path)
        if data['contract'] != contract:
            raise ValueError('Kriging training contract differs')
        return data
    dates = pd.to_datetime(store.hours, unit='h')
    hours = np.asarray(store.hours)[np.isin(dates.year, [2016, 2017, 2018])]
    hours = hours[np.unique(np.linspace(0, len(hours)-1, min(len(hours), 1096)).astype(int))]
    ids = store.ids('B')
    height = store.stations.height.to_numpy()[ids]
    values = store.targets('B', hours).astype('f8')*np.exp(height/3000)
    ii, jj = np.triu_indices(len(ids), 1)
    if len(ii) > 5000:
        choice = np.random.default_rng(20261117).choice(len(ii), 5000, replace=False)
        ii, jj = ii[choice], jj[choice]
    distance = np.linalg.norm(store.xy[ids[ii]]-store.xy[ids[jj]], axis=-1)/1000
    with np.errstate(invalid='ignore'):
        gamma = .5*np.nanmean((values[:, ii]-values[:, jj])**2, axis=0)
    good = np.isfinite(gamma) & (distance > 0)
    distance, gamma = distance[good], gamma[good]
    if len(gamma) < 3:
        raise ValueError('Insufficient training covariance pairs')
    def residual(p):
        sill, length, nugget = np.exp(p)
        return np.log1p(nugget+sill*(1-np.exp(-distance/length)))-np.log1p(gamma)
    variance = max(1., float(np.nanvar(values)))
    result = least_squares(residual, np.log([variance, 200., max(.1, variance*.01)]),
                           bounds=(np.log([1e-3, 25., 1e-5]), np.log([1e7, 1500., 1e6])), loss='soft_l1')
    sill, length, nugget = np.exp(result.x)
    data = dict(contract=contract, sill=float(sill), range_km=float(length), nugget=float(nugget),
                training_hours=len(hours), station_pairs=len(gamma), source='B2016_2018_only', units='Z0_mm')
    immutable_json(path, data)
    return data


def _idw(xy, values, queries, default):
    if not len(xy):
        return np.full(len(queries), default)
    if len(xy) == 1:
        return np.full(len(queries), values[0])
    geometry = input_geometry(xy, queries)
    field = (values[geometry['indices']]*geometry['weights']).sum(-1)
    field[geometry['weights'].sum(-1) == 0] = default
    return field


def get_baseline_predictor(name, store, runtime=None, *, mean_cache=None, parameters=None):
    aliases = {'coarse_bilinear': 'coarse_bilinear', 'A_IDW': 'A_IDW', 'A_kriging': 'A_kriging',
               'M0_innovation_IDW': 'M0_innovation_IDW', 'M0_innovation_idw': 'M0_innovation_IDW'}
    if name not in aliases:
        raise ValueError(f'Unknown registered baseline {name}')
    name = aliases[name]
    input_ids = store.ids('A')
    allxy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
    def predict(hours, context, ids, members):
        rows = []
        for b, hour in enumerate(hours):
            valid = ~context.padding[b].cpu().numpy()
            xyz = store.xy[input_ids][valid]
            values = store.context_values([hour])[0, valid, 0].astype('f8')
            if name == 'coarse_bilinear':
                field = context.baseline[b, 0].cpu().numpy()
            elif name == 'A_IDW':
                field = _idw(xyz, values, allxy, store.stats['input_mean'][0]).reshape(store.shape)
            elif name == 'A_kriging':
                if parameters is None:
                    raise ValueError('Kriging parameters must be fitted on training data first')
                field = (ordinary_kriging(xyz, values, allxy, parameters) if len(xyz) else
                         np.full(len(allxy), store.stats['input_mean'][0])).reshape(store.shape)
            else:
                if mean_cache is None or mean_cache.scenario_hash != context.scenario_hash:
                    raise ValueError('Innovation IDW requires a matching frozen mean cache')
                mean = mean_cache.get([hour])[0, 0]
                q = store.qxy[input_ids][valid]
                within = store.query_valid[input_ids][valid]
                innovation = values[within]-query_grid(mean, q[within]).numpy()
                correction = _idw(xyz[within], innovation, allxy, 0).reshape(store.shape)
                field = mean.numpy()+correction
            rows.append(field)
        fields = torch.as_tensor(np.array(rows)[:, None], dtype=torch.float32)
        xy, height = safe_query_geometry(store, ids)
        return dict(mean=surface_query(fields, xy, height).squeeze(1).numpy(),
                    coarsened=surface_query(coarsen_100km(fields), xy, height).squeeze(1).numpy())
    return predict
