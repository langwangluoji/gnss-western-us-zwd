"""Numerical components for the local, pointwise innovation direction experiment."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize


def aggregate_innovation(values, base_at_A, kernel, valid):
    values, base = np.asarray(values), np.asarray(base_at_A)
    valid = np.asarray(valid, bool) & np.isfinite(values) & np.isfinite(base)
    w = np.asarray(kernel, 'f8')*valid[None]
    return (w@np.where(valid, values-base, 0.))/np.maximum(w.sum(1), 1e-30)


def nested_split(hours, allowed, physical, dense, seed=42):
    hours = np.sort(np.asarray(hours, 'i8'))
    allowed = np.asarray(allowed, bool)
    ids = np.flatnonzero(allowed)
    # Connected components keep aliases and whole 50-km station cells together.
    parent = np.arange(len(allowed))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for labels in (np.asarray(physical), np.asarray(dense)):
        seen = {}
        for i in ids:
            label = str(labels[i])
            if label in seen: parent[find(i)] = find(seen[label])
            else: seen[label] = i
    components = {}
    for i in ids: components.setdefault(find(i), []).append(i)
    keys = np.random.default_rng(seed).permutation(sorted(components))
    hold = []
    for key in keys:
        hold.extend(components[key])
        if len(hold) >= .2*len(ids): break
    inner = np.zeros(len(allowed), bool); inner[hold] = True
    fit = allowed & ~inner
    if not fit.any(): raise ValueError('No fitting station groups remain')
    dates = hours.astype('datetime64[h]')
    months = dates.astype('datetime64[M]').astype(int)%12+1
    starts = []
    for quarter in range(4):
        available = hours[(months-1)//3 == quarter]
        if not len(available): raise ValueError('Each quarter needs cached training hours')
        starts.append(int(available[0]//24*24))
    validation = np.zeros(len(hours), bool); excluded = validation.copy()
    for start in starts:
        validation |= (hours >= start) & (hours < start+168)
        excluded |= (hours >= start-18) & (hours < start+168+18)
    return dict(fit=fit, inner=inner, train_hours=hours[~excluded],
                inner_hours=hours[validation], week_starts=starts)


class InnovationHead(nn.Module):
    def __init__(self, arm, scale):
        super().__init__()
        if arm not in ('constant', 'gated', 'free'): raise ValueError('Unknown point arm')
        self.arm, self.scale = arm, float(scale)
        if arm == 'constant': self.beta = nn.Parameter(torch.zeros(2))
        else:
            self.net = nn.Sequential(nn.Linear(67,32), nn.SiLU(), nn.Linear(32,16),
                                     nn.SiLU(), nn.Linear(16,1 if arm == 'free' else 2))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, innovation):
        if self.arm == 'free': return self.net(features).squeeze(-1)*self.scale
        beta = self.beta.tanh() if self.arm == 'constant' else self.net(features).tanh()
        return (beta*innovation).sum(-1)/2


def query_nodes(field, indices, weights):
    if not torch.isfinite(field).all(): raise ValueError('Nonfinite field')
    return (field[...,indices]*weights).sum(-1)


def station_macro_mse(prediction, truth, mask, counts, total_hours):
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth[mask]).all():
        raise ValueError('Nonfinite active prediction or label')
    use = counts > 0
    if not use.any(): raise ValueError('No training stations')
    error = torch.where(mask, prediction-torch.nan_to_num(truth), 0.)
    weights = torch.where(use, total_hours/counts.clamp_min(1)/use.sum(), 0.)
    return (error.square()*weights).sum(-1).mean()


def epoch_batches(hours, batch, seed, epochs):
    rng = np.random.default_rng(seed); result = []
    for _ in range(epochs):
        order = rng.permutation(hours)
        result.extend(order[i:i+batch] for i in range(0,len(order),batch))
    return result


def gaussian_records(error, mask, amplitude, covariance, groups):
    error, mask = np.asarray(error,'f8'), np.asarray(mask,bool)
    amplitude = np.asarray(amplitude,'f8')
    if not np.isfinite(error[mask]).all() or not np.isfinite(amplitude).all() or (amplitude<=0).any():
        raise ValueError('Invalid Gaussian fit population')
    records = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids)<2: continue
        patterns, inverse = np.unique(mask[:,ids],axis=0,return_inverse=True)
        group_records = []
        for pi, pattern in enumerate(patterns):
            selected = ids[pattern]
            if len(selected)<2: continue
            rows = np.flatnonzero(inverse==pi)
            cov = covariance[np.ix_(selected,selected)]
            eigen, vectors = np.linalg.eigh(cov)
            if eigen.min() < -1e-8*max(1.,eigen.max()): raise ValueError('Non-PSD field covariance')
            z = (error[np.ix_(rows,selected)]/amplitude[rows,None])@vectors
            group_records.append(dict(eigen=np.maximum(eigen,0), squared=np.mean(z*z,axis=0),
                log_g=float(np.log(amplitude[rows]**2).mean()), count=len(rows)))
        if group_records:
            count = sum(x['count'] for x in group_records)
            for row in group_records: row['weight'] = row['count']/count
            records.extend(group_records)
    total = sum(r['weight'] for r in records)
    if not total: raise ValueError('No valid Gaussian groups')
    for row in records: row['weight'] /= total
    return records


def gaussian_objective(log_scales, records, tau):
    f2, n2 = np.exp(2*np.asarray(log_scales)); value=0.; gradient=np.zeros(2)
    for row in records:
        var = f2*row['eigen']+n2*tau*tau
        v = .5*np.mean(np.log(var)+row['squared']/var+row['log_g']+np.log(2*np.pi))
        d = .5*(1/var-row['squared']/var**2)
        value += row['weight']*v
        gradient += row['weight']*np.array([np.mean(d*2*f2*row['eigen']), np.mean(d*2*n2*tau*tau)])
    return float(value),gradient


def fit_gaussian(records, tau, kind, gradient_tolerance=1e-8):
    def expand(z):
        return np.array([z[0],0.]) if kind=='field_only' else np.repeat(z[0],2) if kind=='common' else z
    def objective(z):
        value,g=gaussian_objective(expand(z),records,tau)
        return value, g[:1] if kind=='field_only' else np.array([g.sum()]) if kind=='common' else g
    dim = 2 if kind=='two_scales' else 1
    result=minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',bounds=[(np.log(.1),np.log(1.8))]*dim,
                    options=dict(ftol=1e-13,gtol=gradient_tolerance,maxiter=200))
    if not result.success: raise ValueError(f'Gaussian optimization failed: {result.message}')
    f,n=np.exp(expand(result.x))
    return dict(kind=kind,field=float(f),nugget=float(n),NLL=float(result.fun),iterations=int(result.nit),
                gradient_norm=float(np.linalg.norm(result.jac)),success=True,gradient_tolerance=gradient_tolerance)
