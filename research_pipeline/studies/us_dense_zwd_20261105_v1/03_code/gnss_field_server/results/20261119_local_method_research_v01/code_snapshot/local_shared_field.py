"""Small shared local-field experiments, separate from formal stage trainers."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gnss_field.geometry import input_geometry, surface_query, coarsen_100km
from gnss_field.data.context import cycles


def setconv(kernel, values, valid):
    """Normalized A-only Gaussian aggregation, zero for an empty neighbourhood."""
    valid = torch.as_tensor(valid, device=kernel.device, dtype=torch.bool)
    if not torch.isfinite(values[..., valid]).all():
        raise ValueError('Nonfinite values declared valid')
    weight = kernel*valid
    density = weight.sum(-1)
    field = torch.nan_to_num(values)@weight.T/density.clamp_min(1e-12)
    return field, density


def balanced_blocks(hours):
    """One central observed whole seven-day block per month, no label access."""
    hours = np.asarray(hours, dtype='i8')
    date = pd.to_datetime(hours, unit='h')
    block = hours//168
    selected = []
    for month in range(1, 13):
        choices = np.unique(block[date.month == month])
        if len(choices) < 2:
            raise ValueError('Need at least two observed weeks in each month')
        # Prefer a smaller block to retain training coverage, then the midmonth.
        selected.append(min(choices, key=lambda b: (
            int((block == b).sum()), abs(float(np.median(date.day[block == b]))-16), int(b))))
    development = np.isin(block, selected)
    if set(date[~development].month) != set(range(1, 13)):
        raise ValueError('Whole-week split removes an entire training month')
    return ~development, development


def fit_shared_ridge(features, residual, population, penalty):
    """FP64 station-hour MSE. Scale-only normalization preserves field queries."""
    x = np.asarray(features, 'f8')[population]
    y = np.asarray(residual, 'f8')[population]
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Empty or nonfinite declared training population')
    scale = np.sqrt(np.mean(x*x, axis=0)).clip(1e-8)
    x = x/scale
    gram, rhs = x.T@x/len(y), x.T@y/len(y)
    beta = np.linalg.lstsq(gram+float(penalty)*np.eye(x.shape[1]), rhs, rcond=1e-12)[0]
    return dict(weights=beta/scale, scale=scale, penalty=float(penalty), station_hours=len(y))


def apply_shared_ridge(features, state):
    return np.asarray(features)@np.asarray(state['weights'])


class LocalFieldFeatures:
    """Build actual Z0 feature grids; no B/C observations or fit parameters."""
    def __init__(self, store, device='cuda'):
        self.store, self.device = store, device
        self.a = store.ids('A')
        self.gridxy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
        self.axy = store.xy[self.a]
        distance = cdist(self.gridxy, self.axy)
        self.kernels = [torch.tensor(np.where(distance <= 3*s, np.exp(-.5*(distance/s)**2), 0),
                                     dtype=torch.float32, device=device) for s in (75000., 150000.)]
        self.height = torch.tensor(store.grid['height'], dtype=torch.float32, device=device)/3000.
        self.ah = torch.tensor(store.stations.height.to_numpy()[self.a]/3000., dtype=torch.float32, device=device)
        self.names = ['constant_Z0', 'IDW_minus_M0', 'innovation_75', 'innovation_150',
                      'height_lapse_75', 'height_lapse_150', 'density_75', 'density_150',
                      'innovation_density_75', 'innovation_height', 'blend_height',
                      'local_variable_1', 'local_variable_2', 'local_variable_3', 'local_variable_4',
                      'change_6h', 'change_12h', 'change_18h',
                      'blend_annual_sin', 'blend_annual_cos', 'innovation_annual_sin', 'innovation_annual_cos']

    def fields(self, hour, mean, mean_a, values, history):
        store, device = self.store, self.device
        mean = torch.as_tensor(mean, device=device, dtype=torch.float32).reshape(store.shape)
        raw = torch.as_tensor(values, device=device, dtype=torch.float32)
        valid = torch.isfinite(raw).all(-1)
        innovation_valid = valid & torch.tensor(store.query_valid[self.a], device=device)
        observed_z0 = raw[:, 0]
        innovation = observed_z0-torch.as_tensor(mean_a, device=device)*torch.exp(self.ah)
        innovation[~innovation_valid] = 0
        good = valid.cpu().numpy()
        if good.any():
            geo = input_geometry(self.axy[good], self.gridxy)
            idw = (np.nan_to_num(values[good, 0])[geo['indices']]*geo['weights']).sum(-1)
            idw[geo['weights'].sum(-1) == 0] = store.stats['input_mean'][0]
        else:
            idw = np.full(len(self.gridxy), store.stats['input_mean'][0])
        idw = torch.tensor(idw, dtype=torch.float32, device=device).reshape(store.shape)
        smooth, density, lapse = [], [], []
        for kernel in self.kernels:
            e, _ = setconv(kernel, innovation[None], innovation_valid)
            local, d = setconv(kernel, torch.stack((torch.nan_to_num(observed_z0), self.ah)), valid)
            smooth.append(e[0].reshape(store.shape))
            density.append(torch.log1p(d).reshape(store.shape))
            lapse.append(local[0].reshape(store.shape)*(self.height-local[1].reshape(store.shape)))
        normalized = (raw-torch.tensor(store.stats['input_mean'], device=device))/torch.tensor(store.stats['input_std'], device=device)
        other, _ = setconv(self.kernels[0], normalized[:, 1:].T, valid)
        changes = []
        current, _ = setconv(self.kernels[0], observed_z0[None], valid)
        for prior in history:
            prior = torch.as_tensor(prior, device=device)
            previous, _ = setconv(self.kernels[0], prior[:, 0][None], torch.isfinite(prior).all(-1))
            changes.append((current-previous).reshape(store.shape))
        delta = idw-mean
        annual = cycles([hour])[0][:2]
        fields = [torch.ones_like(mean), delta, *smooth, *lapse, *density,
                  smooth[0]*density[0], smooth[0]*self.height, delta*self.height,
                  *other.reshape(4, *store.shape), *changes,
                  delta*float(annual[0]), delta*float(annual[1]),
                  smooth[0]*float(annual[0]), smooth[0]*float(annual[1])]
        result = torch.stack(fields)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite local feature field')
        return result, idw
