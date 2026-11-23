"""Standalone local spike: matched small U-Nets with/without existing ERA5 covariates.

Inputs use only role A. B groups and weather blocks are isolated before labels
are read. No legacy learned backgrounds or representations are imported.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from gnss_field.artifacts import write_json, read_json, immutable_json, sha256
from gnss_field.data import PackedStore
from gnss_field.geometry.operators import input_geometry, surface_query, coarsen_100km

ROOT = PROJECT/'results/20261123_simple_weather_unet_v02'
DATA = (PROJECT/'../../02_data/processed/gnss_field_server_v1').resolve()
OFFSETS = np.array([0, 6, 12, 18])
MET_CHANNELS = np.array([f*5+j for f in range(4) for j in range(1, 5)])


def idw_fields(xy, values, queries, default):
    good = np.isfinite(values).all(-1)
    if not good.any():
        return np.broadcast_to(default, (len(queries), len(default))).copy(), np.zeros(len(queries), bool)
    points, values = np.asarray(xy)[good], np.asarray(values)[good]
    if len(points) == 1:
        near = np.linalg.norm(queries-points[0], axis=-1) <= 150000
        return np.where(near[:, None], values[0], default), near
    geom = input_geometry(points, queries)
    out = (values[geom['indices']]*geom['weights'][..., None]).sum(1)
    supported = geom['weights'].sum(1) > 0
    out[~supported] = default
    return out, supported


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class SimpleUNet(nn.Module):
    def __init__(self, channels=42):
        super().__init__()
        self.first = DoubleConv(channels, 16)
        self.second = DoubleConv(16, 32)
        self.bottom = DoubleConv(32, 64)
        self.up2 = DoubleConv(64+32, 32)
        self.up1 = DoubleConv(32+16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        a = self.first(x)
        b = self.second(F.avg_pool2d(a, 2))
        c = self.bottom(F.avg_pool2d(b, 2))
        d = self.up2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b), 1))
        e = self.up1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a), 1))
        return self.head(e)


def metrics(pred, target, mask):
    pred, target, mask = np.asarray(pred, 'f8'), np.asarray(target, 'f8'), np.asarray(mask, bool)
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Nonfinite prediction or scored target')
    if not mask.any():
        raise ValueError('Empty scoring population')
    err = np.where(mask, pred-target, 0.)
    counts = mask.sum(0)
    active = counts > 0
    station_rmse = np.sqrt((err**2).sum(0)/np.maximum(counts, 1))
    return dict(macro_rmse=float(station_rmse[active].mean()),
                micro_rmse=float(np.sqrt((err**2).sum()/mask.sum())),
                mae=float(np.abs(err).sum()/mask.sum()), bias=float(err.sum()/mask.sum()),
                stations=int(active.sum()), station_hours=int(mask.sum()),
                coverage=float(mask.mean()))


def select_calendar():
    train, inner, outer = [], [], []
    inner_starts, outer_starts = [], []
    for month in range(1, 13):
        s = int(np.datetime64(f'2018-{month:02d}-01', 'h').astype('i8'))
        inner_starts.append(s)
        outer_starts.append(s+15*24)
        inner.extend(s+np.arange(0, 72, 6))
        outer.extend(s+15*24+np.arange(0, 72, 6))
    for year in (2016, 2017, 2018):
        for month in range(1, 13):
            begin = int(np.datetime64(f'{year}-{month:02d}-01', 'h').astype('i8'))
            end = int((np.datetime64(f'{year}-{month:02d}', 'M')+1).astype('datetime64[h]').astype('i8'))
            pool = np.arange(begin+24, end, 6)
            for start in inner_starts+outer_starts:
                pool = pool[(pool < start-18) | (pool >= start+72+18)]
            train.extend(pool[np.rint(np.linspace(0, len(pool)-1, 24)).astype(int)])
    dev = [int(np.datetime64(f'2019-{m:02d}-{d:02d}T12', 'h').astype('i8'))
           for m in range(1, 13) for d in (5, 12, 19, 26)]
    return dict(train=np.array(sorted(train)), inner=np.array(sorted(inner)),
                outer=np.array(sorted(outer)), development=np.array(dev),
                inner_starts=inner_starts, outer_starts=outer_starts)


def spatial_split(stations):
    b = stations[stations.role == 'train']
    groups = b['dense_group'].unique().copy()
    np.random.default_rng(20261117).shuffle(groups)
    # Whole 50km groups, with target proportions measured by station count.
    group_counts = b.groupby('dense_group').size().to_dict()
    cumulative = np.cumsum([group_counts[g] for g in groups])
    cut1 = int(np.argmin(abs(cumulative-.7*len(b))))+1
    cut2 = int(np.argmin(abs(cumulative-.85*len(b))))+1
    return {name: b.index[b.dense_group.isin(gs)].to_numpy()
            for name, gs in [('fit', groups[:cut1]), ('inner', groups[cut1:cut2]), ('outer', groups[cut2:])]}


def query_table(store):
    q = store.qxy.copy()
    q[:, 0] = q[:, 0].clip(0, store.shape[1]-1)
    q[:, 1] = q[:, 1].clip(0, store.shape[0]-1)
    ix = np.floor(q[:, 0]).astype(int).clip(0, store.shape[1]-2)
    iy = np.floor(q[:, 1]).astype(int).clip(0, store.shape[0]-2)
    dx, dy = q[:, 0]-ix, q[:, 1]-iy
    indices = np.stack([iy*store.shape[1]+ix, iy*store.shape[1]+ix+1,
                        (iy+1)*store.shape[1]+ix, (iy+1)*store.shape[1]+ix+1], -1)
    weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], -1)
    valid = store.query_valid & np.asarray(store.grid['region']).ravel()[indices].all(1)
    return q, indices, weights, valid


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'data_complete.json').exists():
        receipt = read_json(ROOT/'data_complete.json')
        if receipt['sha256'] != sha256(ROOT/'data.npz') or receipt['script_sha256'] != sha256(__file__):
            raise ValueError('Completed pilot cache or implementation changed')
        return
    store = PackedStore(DATA, purpose='context')
    cal = select_calendar()
    splits = spatial_split(store.stations)
    aids, bids, cids = store.ids('A'), store.ids('B'), store.ids('C')
    hours = np.unique(np.concatenate([cal[k] for k in ('train', 'inner', 'outer', 'development')]))
    lock = dict(schema='simple_weather_unet_local_v2_cross_year_buffer', calendar=cal, B_groups=splits,
                data=store.manifest['fingerprint'], script_sha256=sha256(__file__),
                sources='A product Z0, ERA5 station sampled T/P/Tm/PWV; no independent full meteorological grid',
                target='B true surface ZWD; C2019 final development only; D2020 not read',
                old_learned_background=False, seeds=[42, 17], steps=1000, batch=8, lr=.001,
                minimum_lr=.00002, widths=[16, 32, 64], selection='inner B stations at inner weather hours',
                publication_query='25km common-height field, bilinear, exp(-H/3000)',
                user_confirmed_existing_meteorology=True)
    immutable_json(ROOT/'experiment.lock.json', lock)
    history_hours = np.unique((hours[:, None]-OFFSETS).ravel())
    a_values = store.context_values(history_hours).astype('f4')
    apos = np.searchsorted(history_hours, hours[:, None]-OFFSETS)
    train_history = np.unique((cal['train'][:, None]-OFFSETS).ravel())
    train_a = a_values[np.searchsorted(history_hours, train_history)].astype('f8')
    train_a[~np.isfinite(train_a).all(-1)] = np.nan
    mean = np.nanmean(train_a, axis=(0, 1))
    std = np.maximum(np.nanstd(train_a, axis=(0, 1)), 1.)
    if not np.isfinite(mean).all():
        raise ValueError('Missing input statistics')
    H, W = store.shape
    fine_xy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
    cy, cx = np.meshgrid(np.arange(H//4), np.arange(W//4), indexing='ij')
    coarse_xy = store.grid['origin']+np.c_[(cx.ravel()+.5)*100000, (cy.ravel()+.5)*100000]
    q, qindices, qweights, qvalid = query_table(store)
    label_near = cKDTree(store.xy[splits['fit']]).query(store.xy)[0] <= 100000
    qvalid &= label_near
    frame_cache = {}
    coarse = np.empty((len(hours), 20, H//4, W//4), 'f4')
    cmask = np.empty((len(hours), 4, H//4, W//4), 'f4')
    sparse = np.zeros((len(hours), 8, H, W), 'f4')
    support = np.zeros((len(hours), len(store.stations)), bool)
    baseline = np.empty((len(hours), 3, len(store.stations)), 'f4')
    height = store.stations.height.to_numpy()
    scale_h = np.exp(-height/3000)
    aq = store.qxy[aids]
    ax = np.floor(aq[:, 0]).astype(int).clip(0, W-2)
    ay = np.floor(aq[:, 1]).astype(int).clip(0, H-2)
    adx, ady = (aq[:, 0]-ax).clip(0, 1), (aq[:, 1]-ay).clip(0, 1)
    asplat_i = np.stack([ay*W+ax, ay*W+ax+1, (ay+1)*W+ax, (ay+1)*W+ax+1], -1)
    asplat_w = np.stack([(1-adx)*(1-ady), adx*(1-ady), (1-adx)*ady, adx*ady], -1)
    ainside = (aq >= 0).all(1) & (aq[:, 0] <= W-1) & (aq[:, 1] <= H-1)
    started = time.monotonic()
    for t, hour in enumerate(hours):
        for f, ai in enumerate(apos[t]):
            if int(ai) not in frame_cache:
                values = a_values[ai]
                good = np.isfinite(values).all(-1)
                normalized = (values-mean)/std
                vals, ok = idw_fields(store.xy[aids], normalized, coarse_xy, np.zeros(5))
                raw = np.zeros((2, H*W), 'f4')
                weights = asplat_w*(good & ainside)[:, None]
                sums = np.bincount(asplat_i.ravel(), weights=(np.nan_to_num(normalized[:, 0])[:, None]*weights).ravel(), minlength=H*W)
                density = np.bincount(asplat_i.ravel(), weights=weights.ravel(), minlength=H*W)
                raw[0] = np.divide(sums, density, out=np.zeros_like(sums), where=density > 0)
                raw[1] = np.log1p(density)
                frame_cache[int(ai)] = vals.T.reshape(5, H//4, W//4), ok.reshape(H//4, W//4), raw.reshape(2, H, W)
            v, m, s = frame_cache[int(ai)]
            coarse[t, f*5:f*5+5], cmask[t, f], sparse[t, f*2:f*2+2] = v, m, s
        current = a_values[apos[t, 0]]
        good = np.isfinite(current).all(-1)
        if good.sum() >= 3:
            support[t] = input_geometry(store.xy[aids][good], store.xy)['support'] & qvalid
        coarse_field = F.interpolate(torch.from_numpy(coarse[t:t+1, :1]*std[0]+mean[0]),
                                     size=store.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
        idw, _ = idw_fields(store.xy[aids], current, fine_xy, mean)
        era_values = current.copy()
        # Derived from e in hPa: PWV/ZWD = 1e8/[rho_w R_v (k2'+k3/Tm)].
        era_values[:, 0] = current[:, 4]*(1000.*461.5*(22.1+373900./current[:, 3]))*1e-8*np.exp(height[aids]/3000)
        era, _ = idw_fields(store.xy[aids], era_values, fine_xy, mean)
        for k, field in enumerate((coarse_field, idw[:, 0].reshape(H, W), era[:, 0].reshape(H, W))):
            baseline[t, k] = (field.ravel()[qindices]*qweights).sum(-1)*scale_h
        if (t+1) % 100 == 0:
            print(json.dumps(dict(stage='prepare', hours=t+1, total=len(hours), seconds=round(time.monotonic()-started, 1))), flush=True)
        # Inputs are small, but sparse full-grid frames need a bounded cache.
        if len(frame_cache) > 96:
            active = set(apos[t].tolist())
            for key in list(frame_cache):
                if key not in active:
                    del frame_cache[key]
                if len(frame_cache) <= 64:
                    break
    north, east = np.gradient(store.grid['height'].astype('f8'), 25000, 25000)
    xy_norm = 2*(fine_xy-fine_xy.min(0))/np.maximum(np.ptp(fine_xy, axis=0), 1)-1
    static = np.stack([store.grid['height']/3000, east/.1, north/.1,
                       xy_norm[:, 0].reshape(H, W), xy_norm[:, 1].reshape(H, W), store.grid['region']]).astype('f4')
    dates = pd.to_datetime(hours, unit='h')
    phase_d = dates.hour.to_numpy()/24*2*np.pi
    phase_y = (dates.dayofyear.to_numpy()-1+dates.hour.to_numpy()/24)/365.25*2*np.pi
    cycles = np.stack([np.sin(phase_d), np.cos(phase_d), np.sin(phase_y), np.cos(phase_y)], -1).astype('f4')
    # Labels are fetched only after the condition arrays have been built.
    labels = np.full((len(hours), len(store.stations)), np.nan, 'f4')
    train_view, dev_view = PackedStore(DATA, purpose='train'), PackedStore(DATA, purpose='development')
    old_rows = np.flatnonzero(dates.year <= 2018)
    dev_rows = np.flatnonzero(dates.year == 2019)
    labels[np.ix_(old_rows, bids)] = train_view.targets('B', hours[old_rows])
    labels[np.ix_(dev_rows, cids)] = dev_view.targets('C', hours[dev_rows])
    mask = support & np.isfinite(labels)
    np.savez(ROOT/'data.npz', hours=hours, coarse=coarse, cmask=cmask, sparse=sparse,
             static=static, cycles=cycles, labels=labels, mask=mask, baseline=baseline,
             qxy=q, qindices=qindices, qweights=qweights, heights=height, land=store.grid['region'],
             xy=store.xy, station_names=store.stations.station.to_numpy(dtype='U16'),
             norm_mean=mean, norm_std=std, **{f'split_{k}': v for k, v in splits.items()},
             C_ids=cids, **{f'hours_{k}': cal[k] for k in ('train', 'inner', 'outer', 'development')})
    write_json(ROOT/'data_complete.json', dict(sha256=sha256(ROOT/'data.npz'), script_sha256=sha256(__file__),
               statistics=dict(mean=mean, std=std, role='A_training_hours_only'),
               hours=len(hours), train_hours=len(cal['train']), shape=list(store.shape),
               role_counts={k: len(v) for k, v in splits.items()}, seconds=time.monotonic()-started,
               D_labels_read=False, C_input=False, old_weights=False))


class PilotData:
    def __init__(self):
        archive = np.load(ROOT/'data.npz', allow_pickle=False)
        self.a = {k: archive[k] for k in archive.files}
        self.hours = self.a['hours']
        self.coarse = torch.from_numpy(self.a['coarse']).cuda()
        self.cmask = torch.from_numpy(self.a['cmask']).cuda()
        self.sparse = torch.from_numpy(self.a['sparse'])
        self.static = torch.from_numpy(self.a['static']).cuda()
        self.cycles = torch.from_numpy(self.a['cycles']).cuda()
        self.qi = torch.from_numpy(self.a['qindices']).cuda()
        self.qw = torch.from_numpy(self.a['qweights'].astype('f4')).cuda()
        self.height_factor = torch.from_numpy(np.exp(-self.a['heights']/3000).astype('f4')).cuda()
        self.labels = torch.from_numpy(np.nan_to_num(self.a['labels'])).cuda()
        self.mask = torch.from_numpy(self.a['mask']).cuda()

    def rows(self, name):
        return np.searchsorted(self.hours, self.a[f'hours_{name}'])

    def batch(self, rows, weather):
        v = self.coarse[rows].clone()
        shape = self.static.shape[-2:]
        base = F.interpolate(v[:, :1]*self.a['norm_std'][0]+self.a['norm_mean'][0], size=shape, mode='bilinear', align_corners=False)
        if not weather:
            v[:, MET_CHANNELS] = 0
        c = F.interpolate(torch.cat([v, self.cmask[rows]], 1), size=shape, mode='bilinear', align_corners=False)
        x = torch.cat([c, self.sparse[rows].cuda(), self.static[None].expand(len(rows), -1, -1, -1),
                       self.cycles[rows, :, None, None].expand(-1, -1, *shape)], 1)
        return x, base

    def query(self, field, ids):
        return (field.flatten(1)[:, self.qi[ids]]*self.qw[ids]).sum(-1)*self.height_factor[ids]


@torch.no_grad()
def predict(model, data, rows, weather, ids, *, smooth=False):
    model.eval()
    result = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        x, base = data.batch(batch, weather)
        field = (base+10*model(x))[:, 0]
        if not torch.isfinite(field).all():
            raise ValueError('Nonfinite complete field')
        if smooth:
            field = coarsen_100km(field)
        result.append(data.query(field, ids).cpu().numpy())
    return np.concatenate(result)


def score(model, data, rows, weather, ids):
    pred = predict(model, data, rows, weather, ids)
    return metrics(pred, data.a['labels'][np.ix_(rows, ids)], data.a['mask'][np.ix_(rows, ids)])


def train_one(data, weather, seed, steps):
    name = f'{"gnss_weather_terrain" if weather else "gnss_terrain"}_seed{seed}'
    out = ROOT/name
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        return read_json(out/'complete.json')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SimpleUNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    train_rows, inner_rows = data.rows('train'), data.rows('inner')
    fit_ids, inner_ids = data.a['split_fit'], data.a['split_inner']
    fixed_train = train_rows[np.rint(np.linspace(0, len(train_rows)-1, 48)).astype(int)]
    rng = np.random.default_rng(seed)
    # Fixed schedule is identical between weather arms with the same seed.
    schedule = np.concatenate([rng.permutation(train_rows) for _ in range(math.ceil(steps*8/len(train_rows)))])[:steps*8].reshape(steps, 8)
    initial = score(model, data, inner_rows, weather, inner_ids)
    best, best_step = initial['macro_rmse'], 0
    torch.save(dict(model=model.state_dict(), step=0, seed=seed, weather=weather), out/'best.pt')
    history = [dict(step=0, inner=initial, train=score(model, data, fixed_train, weather, fit_ids))]
    started = time.monotonic()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        if (ROOT/'PAUSE').exists():
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step-1,
                            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), schedule=schedule), out/'paused.pt')
            write_json(ROOT/'status.json', dict(stage='PAUSED', model=name, step=step-1))
            raise RuntimeError('Pilot paused at user-requested PAUSE marker')
        model.train()
        rows = schedule[step-1]
        x, base = data.batch(rows, weather)
        field = (base+10*model(x))[:, 0]
        pred = data.query(field, fit_ids)
        mask = data.mask[rows][:, fit_ids]
        target = data.labels[rows][:, fit_ids]
        if not torch.isfinite(pred).all() or not mask.any():
            raise ValueError('Invalid training prediction or empty supervision')
        error = (pred-target)/10
        loss = (error.square()*mask).sum()/mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise ValueError('Nonfinite gradient')
        lr = .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(step-1)/max(steps-1, 1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        losses.append(float(loss.detach())*100)
        if step % 100 == 0 or step == steps:
            inner = score(model, data, inner_rows, weather, inner_ids)
            fixed = score(model, data, fixed_train, weather, fit_ids)
            if inner['macro_rmse'] < best:
                best, best_step = inner['macro_rmse'], step
                torch.save(dict(model=model.state_dict(), step=step, seed=seed, weather=weather,
                                inner=inner, optimizer=optimizer.state_dict()), out/'best.pt')
            event = dict(step=step, mean_loss_mm2=float(np.mean(losses)), inner=inner, train=fixed,
                         lr=lr, gradient_norm=float(grad), seconds=time.monotonic()-started,
                         best_rmse=best, best_step=best_step,
                         gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
            losses = []
            history.append(event)
            with (out/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')
            write_json(out/'curve.json', history)
            write_json(ROOT/'status.json', dict(stage='TRAINING', model=name, updated=time.time(), **event))
            print(json.dumps(dict(model=name, **event)), flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=steps), out/'last.pt')
    selected = torch.load(out/'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(selected['model'])
    populations = dict(spatial=(fixed_train, data.a['split_outer']),
                       temporal=(data.rows('outer'), fit_ids),
                       spatial_temporal=(data.rows('outer'), data.a['split_outer']),
                       C2019_development=(data.rows('development'), data.a['C_ids']))
    results = {}
    for label, (rows, ids) in populations.items():
        pred = predict(model, data, rows, weather, ids)
        smooth = predict(model, data, rows, weather, ids, smooth=True)
        y = data.a['labels'][np.ix_(rows, ids)]
        m = data.a['mask'][np.ix_(rows, ids)]
        methods = {'unet': metrics(pred, y, m), 'unet_100km': metrics(smooth, y, m)}
        base = data.a['baseline'][rows][:, :, ids]
        for j, method in enumerate(('A_coarse_bilinear', 'A_IDW_grid', 'ERA5_station_PWV_Tm_grid')):
            methods[method] = metrics(base[:, j], y, m)
        results[label] = methods
        np.savez(out/f'{label}_predictions.npz', prediction=pred, coarsened=smooth,
                 target=y, mask=m, baseline=base, hour_ids=data.hours[rows],
                 station_ids=ids, station_names=data.a['station_names'][ids])
    receipt = dict(name=name, weather=weather, seed=seed, steps=steps, selected_step=selected['step'],
                   best_inner_rmse=best, parameters=sum(p.numel() for p in model.parameters()),
                   seconds=time.monotonic()-started, results=results, checkpoint_sha256=sha256(out/'best.pt'))
    write_json(out/'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 17])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    prepare()
    if args.prepare_only:
        return
    data = PilotData()
    results = []
    for seed in args.seeds:
        for weather in (False, True):
            results.append(train_one(data, weather, seed, args.steps))
    write_json(ROOT/'results.json', dict(status='COMPLETE', runs=results, D2020=False))
    write_json(ROOT/'status.json', dict(stage='COMPLETE', updated=time.time(), runs=len(results)))
    print('LOCAL_UNET_PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
