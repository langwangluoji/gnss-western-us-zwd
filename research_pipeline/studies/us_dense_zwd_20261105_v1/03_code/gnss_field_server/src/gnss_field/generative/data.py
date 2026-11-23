"""Compact, role-separated full-hour observations for conditional field models.

No target array is touched by the condition path.  Cache shards contain coarse
inputs and raw frozen backgrounds; 37-channel fine grids are assembled in RAM.
The frozen background is an explicit external dependency, never a fitted target.
"""
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree

from gnss_field.artifacts import (fingerprint, immutable_json,
                                  atomic_numpy, read_json, sha256, write_json)
from gnss_field.data import PackedStore
from gnss_field.geometry.operators import input_geometry


INPUT_VARIABLES = ('Z0_mm', 'temperature_k', 'pressure_hpa', 'tm_k', 'era5_pwv_mm')
INPUT_UNITS = ('mm', 'K', 'hPa', 'K', 'mm')
FRAME_OFFSETS = (0, 6, 12, 18)


@dataclass(frozen=True)
class FieldDataConfig:
    packed_root: str
    cache_root: str | None = None
    background_assets: dict | str | None = None
    train_years: tuple[int, ...] = (2016, 2017, 2018)
    development_year: int = 2019
    frame_offsets: tuple[int, ...] = FRAME_OFFSETS
    prepare_batch_hours: int = 8
    mapped_month_capacity: int = 128

    def __post_init__(self):
        if tuple(self.train_years) != (2016, 2017, 2018):
            raise ValueError('Full training requires all 2016–2018 years')
        if self.development_year != 2019 or tuple(self.frame_offsets) != FRAME_OFFSETS:
            raise ValueError('Development year and historical frame offsets are frozen')
        if self.prepare_batch_hours < 1 or self.mapped_month_capacity < 1:
            raise ValueError('Preparation batch and mapped month capacities must be positive')


class VerifiedBackgroundProvider:
    """Verify external immutable assets before invoking an input-only callback.

    ``callback(hours[B], raw_frames[B,4,A,5])`` returns raw common-height
    ``[B,1,H,W]`` millimetres. Callback implementations retain their own frozen
    normalization. They must neither read current targets nor train parameters.
    Include callback source and every learned/static dependency in ``assets``.
    """

    def __init__(self, callback, assets, *, identity=None):
        if not callable(callback) or not isinstance(assets, dict) or not assets:
            raise ValueError('Frozen background requires a callback and explicit hashed assets')
        self.assets = {str(Path(p).resolve()): digest for p, digest in assets.items()}
        for path, digest in self.assets.items():
            if not Path(path).is_file() or sha256(path) != digest:
                raise ValueError(f'Frozen background asset mismatch: {path}')
        self.callback = callback
        self.identity = fingerprint(dict(assets=self.assets, operator=identity))

    def __call__(self, hours, frames):
        return self.callback(hours, frames)


class PrecomputedBackgroundProvider:
    """Explicit precomputed frozen fields, verified without any training imports.

    Manifest schema: ``schema:1, units:'common_height_Z0_mm', grid_id:SHA256,
    assets:{relative_or_absolute_path:SHA256}, months:{'YYYY-MM':{
    file:'relative.npz', sha256:SHA256}}``. NPZ columns are ``hour_ids`` and
    ``background`` [T,1,H,W]. Asset and field hashes are validated independently.
    """

    def __init__(self, manifest_path, *, grid_id):
        path = Path(manifest_path).resolve()
        self.root, self.manifest = path.parent, read_json(path)
        m = self.manifest
        if m.get('schema') != 1 or m.get('units') != 'common_height_Z0_mm' or m.get('grid_id') != grid_id:
            raise ValueError('Precomputed background geometry/units contract mismatch')
        assets = m.get('assets', {})
        if not assets:
            raise ValueError('Precomputed background needs explicit frozen source assets')
        for name, digest in assets.items():
            source = Path(name) if Path(name).is_absolute() else self.root/name
            if not source.is_file() or sha256(source) != digest:
                raise ValueError(f'Frozen background asset mismatch: {source}')
        self.identity = fingerprint(m)

    @lru_cache(maxsize=2)
    def _month(self, month):
        row = self.manifest['months'][month]
        path = self.root/row['file']
        if sha256(path) != row['sha256']:
            raise ValueError('Precomputed background checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            return np.array(z['hour_ids']), np.array(z['background'])

    def __call__(self, hours, frames):
        result = []
        for hour, month in zip(hours, pd.to_datetime(hours, unit='h').strftime('%Y-%m')):
            h, fields = self._month(month)
            i = int(np.searchsorted(h, hour))
            if i >= len(h) or h[i] != hour:
                raise ValueError('Requested hour absent from frozen background')
            result.append(fields[i])
        return np.stack(result)


def _cycles(hours):
    d = pd.to_datetime(hours, unit='h')
    day = 2*np.pi*d.hour.to_numpy()/24
    year = 2*np.pi*(d.dayofyear.to_numpy()-1+d.hour.to_numpy()/24)/365.25
    return np.stack((np.sin(day), np.cos(day), np.sin(year), np.cos(year)), -1).astype('f4')


class FullFieldData:
    """Read-only batch API plus explicit atomic monthly cache preparation.

    A training view reads only B2016–2018 labels; a development view reads only
    C2019 labels. Both may construct A-only conditions. D labels are always
    forbidden. The opposite view's qualified-hour property is available through
    a separate PackedStore view, never by weakening a store's purpose gate.
    """

    def __init__(self, config, purpose='train', *, store=None, background_provider=None):
        self.config = FieldDataConfig(**config) if isinstance(config, dict) else config
        if purpose not in ('train', 'development', 'context'):
            raise ValueError('Full generative data supports train/development/context only')
        self.purpose = purpose
        self.store = store or PackedStore(self.config.packed_root, purpose=purpose)
        if self.store.purpose != purpose:
            raise ValueError('Injected PackedStore purpose must match the data view')
        self.shape = tuple(self.store.shape)
        h, w = self.shape
        if h % 4 or w % 4 or min(h, w) < 4:
            raise ValueError('Grid must align to the global 4x4 coarse lattice')
        if self.store.manifest.get('registered') and self.shape != (144, 96):
            raise ValueError('Registered global grid must be 144x96')
        if self.store.stats.get('height_datum') != 'EGM96':
            raise ValueError('Expected EGM96 orthometric heights')
        if background_provider is None:
            if not isinstance(self.config.background_assets, (str, Path)):
                raise ValueError('An explicit frozen background provider or manifest is required')
            background_provider = PrecomputedBackgroundProvider(self.config.background_assets, grid_id=self.store.grid_id)
        if not getattr(background_provider, 'identity', None):
            raise ValueError('Frozen background provider must expose a stable identity')
        if self.store.manifest.get('registered') and not isinstance(
                background_provider, (VerifiedBackgroundProvider, PrecomputedBackgroundProvider)):
            raise ValueError('Registered backgrounds require verified external assets')
        self.background_provider = background_provider
        self._background_identity = background_provider.identity
        if isinstance(self.config.background_assets, dict):
            expected = {str(Path(p).resolve()): digest for p, digest in self.config.background_assets.items()}
            if expected != getattr(background_provider, 'assets', None):
                raise ValueError('Configured background assets differ from provider assets')
        self.aids, self.bids = self.store.ids('A'), self.store.ids('B')
        self.input_geometry = np.c_[self.store.xy[self.aids], self.store.stations.height.to_numpy()[self.aids]].astype('f4')
        g = self.store.grid
        self.grid_geometry = np.stack((g['x'], g['y'], g['height']), -1).astype('f4')
        self.fine_xy = np.c_[np.asarray(g['x'], 'f8').ravel(), np.asarray(g['y'], 'f8').ravel()]
        self.coarse_shape = (h//4, w//4)
        cy, cx = np.meshgrid(np.arange(h//4), np.arange(w//4), indexing='ij')
        self.coarse_xy = np.asarray(g['origin']) + np.c_[(cx.ravel()+.5)*100000, (cy.ravel()+.5)*100000]
        expected_x = g['origin'][0]+(np.arange(w)+.5)*25000
        expected_y = g['origin'][1]+(np.arange(h)+.5)*25000
        if not np.allclose(g['x'], expected_x[None, :]) or not np.allclose(g['y'], expected_y[:, None]):
            raise ValueError('Projected grid must use global origin and 25km centres')
        self.land = np.asarray(g['region'], bool)
        if not np.isfinite(self.grid_geometry).all() or not np.isfinite(self.input_geometry).all():
            raise ValueError('Nonfinite projected grid or station geometry')
        self._label_near_grid = cKDTree(self.store.xy[self.bids]).query(self.fine_xy)[0] <= 100000
        self._label_near_query = cKDTree(self.store.xy[self.bids]).query(self.store.xy)[0] <= 100000
        self._query_indices, self._query_weights, self._fixed_query = self._queries()
        self._query_geometry = np.c_[self.store.xy, self.store.stations.height.to_numpy()].astype('f4')
        self.input_mean, self.input_std, self._statistics = self._training_statistics()
        lo, span = self.fine_xy.min(0), np.ptp(self.fine_xy, axis=0)
        self._grid_xy_norm = (2*(self.fine_xy-lo)/np.maximum(span, 1)-1).reshape(h, w, 2).astype('f4')
        self._station_xy_norm = (2*(self.input_geometry[:, :2]-lo)/np.maximum(span, 1)-1).astype('f4')
        north, east = np.gradient(np.asarray(g['height'], 'f8'), 25000, 25000)
        self._static = np.stack((g['height']/3000, east/.1, north/.1,
                                 self._grid_xy_norm[..., 0], self._grid_xy_norm[..., 1], self.land)).astype('f4')
        self.metadata = dict(schema='full_generative_37grid_31station_v1',
            cache_layout='monthly_uncompressed_NPY_atomic_pointer_v2', feature_builder_sha256=sha256(__file__),
            grid_shape=list(self.shape), coarse_shape=list(self.coarse_shape),
            grid_spacing_m=25000, coarse_spacing_m=100000, height_datum='EGM96',
            time='nominal_product_hour', frame_offsets=list(FRAME_OFFSETS),
            input_variables=list(INPUT_VARIABLES), input_units=list(INPUT_UNITS),
            grid_channels=[f'frame_{offset}h_{name}' for offset in FRAME_OFFSETS for name in INPUT_VARIABLES]+
                [f'frame_{offset}h_valid_A_fraction' for offset in FRAME_OFFSETS]+
                ['height_div3000', 'east_slope_div0.1', 'north_slope_div0.1', 'domain_x', 'domain_y', 'land',
                 'normalized_frozen_background', 'nearest_A_capped1000km_div150km', 'log1p_A_within150km_divlog175',
                 'sin_day', 'cos_day', 'sin_year', 'cos_year'],
            station_channels=[f'frame_{offset}h_{name}' for offset in FRAME_OFFSETS for name in INPUT_VARIABLES]+
                [f'frame_{offset}h_joint_valid' for offset in FRAME_OFFSETS]+
                ['domain_x', 'domain_y', 'height_div3000', 'sin_day', 'cos_day', 'sin_year', 'cos_year'],
            target_units='surface_ZWD_mm', background_units='common_height_Z0_mm',
            query_weights='pure_bilinear_height_restoration_separate',
            statistics=self._statistics, frozen_background_identity=self._background_identity,
            grid_id=self.store.grid_id, source_hashes=self._source_hashes(),
            support=dict(nearest_input_m=100000, neighbors_radius_m=150000,
                         neighbors_min=3, convex_hull=True, nearest_training_station_m=100000,
                         complete_land_query_stencil=True))
        self.identity = fingerprint(self.metadata)
        self.metadata['condition_identity'] = self.identity
        self.cache_root = Path(self.config.cache_root)/self.identity[:20] if self.config.cache_root else None
        self._audits = {}
        self._monthly_cache = OrderedDict()

    def _source_hashes(self):
        m = self.store.manifest
        # No D values or D file checksums are read. Source identity freezes A only;
        # target releases have separate hashes and cache namespaces below.
        return dict(shared=m.get('shared', {}), A={k: v['sha256'] for k, v in m.get('files', {}).items()
                    if k.startswith('A/') and int(k[2:6]) <= self.config.development_year})

    def _training_statistics(self):
        hours = np.asarray(self.store.hours, dtype='i8')
        dates = pd.to_datetime(hours, unit='h')
        selected = hours[np.isin(dates.year, self.config.train_years)]
        count, mean, m2 = 0, np.zeros(5, 'f8'), np.zeros(5, 'f8')
        # Parallel variance combination is stable and keeps monthly memory bounded.
        for start in range(0, len(selected), 744):
            raw = self.store.context_values(selected[start:start+744]).astype('f8')
            values = raw[np.isfinite(raw).all(-1)]
            n = len(values)
            if n:
                local = values.mean(0)
                delta = local-mean
                m2 += ((values-local)**2).sum(0)+delta**2*count*n/(count+n)
                mean += delta*n/(count+n)
                count += n
        if not count:
            raise ValueError('No jointly valid A2016–2018 observations for normalization')
        std = np.maximum(np.sqrt(m2/count), 1e-6)
        stats = dict(input_mean=mean.tolist(), input_std=std.tolist(), statistics_years=list(self.config.train_years),
                     source='joint_valid_A_only', station_hours=int(count), background_normalization='A_Z0_mean_std',
                     frozen_background_operator_normalization='external_asset_contract')
        return mean, std, stats

    def _queries(self):
        h, w = self.shape
        q = (self.store.xy-self.store.grid['origin'])/25000-.5
        finite = np.isfinite(q).all(-1)
        inside = finite & (q >= 0).all(-1) & (q[:, 0] <= w-1) & (q[:, 1] <= h-1)
        safe = np.nan_to_num(q).clip([0, 0], [w-1, h-1])
        x = np.floor(safe[:, 0]).astype('i8').clip(0, w-2)
        y = np.floor(safe[:, 1]).astype('i8').clip(0, h-2)
        dx, dy = safe[:, 0]-x, safe[:, 1]-y
        indices = np.stack((y*w+x, y*w+x+1, (y+1)*w+x, (y+1)*w+x+1), -1)
        weights = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1).astype('f4')
        fixed = inside & self.land.ravel()[indices].all(-1) & self._label_near_query
        return indices, weights, fixed

    @lru_cache(maxsize=256)
    def _geometry(self, pattern):
        valid = np.frombuffer(pattern, dtype='?')
        xy = np.asarray(self.store.xy[self.aids][valid], 'f8')
        return tuple(input_geometry(xy, queries) for queries in
                     (self.fine_xy, self.coarse_xy, self.store.xy))

    def _support(self, valid):
        fine, _, query = self._geometry(valid.tobytes())
        grid = fine['support'] & self._label_near_grid & self.land.ravel()
        sites = query['support'] & self._fixed_query & grid[self._query_indices].all(-1)
        return grid.reshape(self.shape), sites, fine

    def _hours(self, hours):
        hours = np.asarray(hours, dtype='i8').reshape(-1)
        if not len(hours):
            raise ValueError('An hour batch must be nonempty')
        available = np.asarray(self.store.hours)
        if not np.isin((hours[:, None]-np.array(FRAME_OFFSETS)).ravel(), available).all():
            raise ValueError('Full 18h history and nominal hours must exist (19h inclusive boundary)')
        if (pd.to_datetime(hours, unit='h').year > 2019).any():
            raise PermissionError('D2020 and later hours are excluded from this implementation')
        return hours

    def _build_compact(self, hours):
        h, w = self.shape
        frames = self.store.context_values((hours[:, None]-np.array(FRAME_OFFSETS)).ravel())
        frames = frames.reshape(len(hours), 4, len(self.aids), 5).astype('f4')
        valid = np.isfinite(frames).all(-1)
        normalized = np.where(valid[..., None], (frames-self.input_mean)/self.input_std, 0).astype('f4')
        coarse = np.zeros((len(hours), 20, h//4, w//4), 'f4')
        nearest, count, support = [], [], []
        for i in range(len(hours)):
            mask, _, fine = self._support(valid[i, 0])
            support.append(mask)
            nearest.append(fine['nearest'].reshape(h, w).astype('f4'))
            count.append(fine['count'].reshape(h, w).astype('u1'))
            for j in range(4):
                selected = np.flatnonzero(valid[i, j])
                if not len(selected):
                    continue
                if len(selected) == 1:
                    value = np.broadcast_to(normalized[i, j, selected[0]], (np.prod(self.coarse_shape), 5))
                else:
                    geo = self._geometry(valid[i, j].tobytes())[1]
                    value = (normalized[i, j, selected][geo['indices']]*geo['weights'][..., None]).sum(1)
                    value[~geo['support']] = 0
                coarse[i, j*5:(j+1)*5] = value.T.reshape(5, *self.coarse_shape)
        background = np.asarray(self.background_provider(hours, frames), 'f4')
        if background.shape != (len(hours), 1, h, w) or not np.isfinite(background).all():
            raise ValueError('Frozen background must be finite [B,1,H,W] common-height mm')
        return dict(hour_ids=hours, coarse=coarse, normalized=normalized, frame_valid=valid,
                    background=background, nearest=np.asarray(nearest), count=np.asarray(count), support=np.asarray(support))

    def _load_shard(self, path_string):
        if path_string in self._monthly_cache:
            self._monthly_cache.move_to_end(path_string)
            return self._monthly_cache[path_string]
        path = Path(path_string)
        record = read_json(path.with_suffix('.json'))
        expected = self.identity if path.parent.name == 'conditions' else self._label_identity(path.parent.parent.name)
        if record['identity'] != expected or Path(record['directory']).name != record['directory']:
            raise ValueError(f'Compact monthly cache identity mismatch: {path}')
        directory = path.parent/record['directory']
        arrays = {}
        for key, info in record['arrays'].items():
            if Path(key).name != key:
                raise ValueError('Invalid compact array name')
            source = directory/f'{key}.npy'
            if not source.is_file() or info['sha256'] != sha256(source):
                raise ValueError(f'Compact monthly cache checksum mismatch: {source}')
            arrays[key] = np.load(source, mmap_mode='r', allow_pickle=False)
            if list(arrays[key].shape) != info['shape'] or str(arrays[key].dtype) != info['dtype']:
                raise ValueError('Compact monthly cache array contract mismatch')
        # These are demand-paged mappings, not a resident decoded month cache.
        # The default holds all 36 training and 12 development condition/label
        # shards. Random full epochs do not reopen/rehash/decompress whole months.
        self._monthly_cache[path_string] = arrays
        while len(self._monthly_cache) > self.config.mapped_month_capacity:
            self._monthly_cache.popitem(last=False)
        return arrays

    @staticmethod
    def _has_shard(path):
        return path is not None and path.with_suffix('.json').is_file()

    def _compact(self, hours):
        rows, missing = [], []
        for i, (hour, month) in enumerate(zip(hours, pd.to_datetime(hours, unit='h').strftime('%Y-%m'))):
            path = self.cache_root/'conditions'/f'{month}.npz' if self.cache_root else None
            if self._has_shard(path):
                shard = self._load_shard(str(path))
                pos = int(np.searchsorted(shard['hour_ids'], hour))
                if pos < len(shard['hour_ids']) and shard['hour_ids'][pos] == hour:
                    rows.append({key: value[pos] for key, value in shard.items()})
                    continue
            rows.append(None)
            missing.append(i)
        if missing:
            built = self._build_preparation_month(hours[missing])
            for j, index in enumerate(missing):
                rows[index] = {key: value[j] for key, value in built.items()}
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}

    def condition_batch(self, hour_ids, device='cpu'):
        hours = self._hours(hour_ids)
        data = self._compact(hours)
        b, (h, w) = len(hours), self.shape
        grid = np.zeros((b, 37, h, w), 'f4')
        grid[:, 20:24] = data['frame_valid'].mean(-1)[..., None, None]
        grid[:, 24:30] = self._static[None]
        grid[:, 30] = (data['background'][:, 0]-self.input_mean[0])/self.input_std[0]
        grid[:, 31] = np.minimum(data['nearest'], 1000000)/150000
        grid[:, 32] = np.log1p(data['count'].astype('f4'))/np.log(175)
        cycle = _cycles(hours)
        grid[:, 33:37] = cycle[..., None, None]
        tokens = np.zeros((b, len(self.aids), 31), 'f4')
        tokens[:, :, :20] = data['normalized'].transpose(0, 2, 1, 3).reshape(b, len(self.aids), 20)
        tokens[:, :, 20:24] = data['frame_valid'].transpose(0, 2, 1)
        tokens[:, :, 24:26] = self._station_xy_norm
        tokens[:, :, 26] = self.input_geometry[:, 2]/3000
        tokens[:, :, 27:31] = cycle[:, None]
        def tensor(value, dtype=None):
            return torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)
        condition = tensor(grid)
        condition[:, :20] = F.interpolate(tensor(data['coarse']), size=self.shape, mode='bilinear', align_corners=False)
        return dict(hour_ids=tensor(hours), grid_condition=condition, station_tokens=tensor(tokens),
                    station_mask=tensor(data['frame_valid'].any(1)), grid_geometry=tensor(self.grid_geometry),
                    input_geometry=tensor(self.input_geometry), background=tensor(data['background']),
                    support=tensor(data['support']))

    def _check_role(self, role, hours):
        years = pd.to_datetime(hours, unit='h').year
        allowed = ((self.purpose == 'train' and role == 'B' and np.isin(years, self.config.train_years).all()) or
                   (self.purpose == 'development' and role == 'C' and (years == 2019).all()))
        if not allowed:
            raise PermissionError(f'{self.purpose} full-data view cannot read {role} labels at requested hours')

    def _label_identity(self, role):
        years = self.config.train_years if role == 'B' else (2019,)
        files = {k: v['sha256'] for k, v in self.store.manifest.get('files', {}).items()
                 if k.startswith(role+'/') and int(k[2:6]) in years}
        return fingerprint(dict(role=role, files=files, condition_identity=self.identity))

    def _targets(self, role, hours):
        self._check_role(role, hours)
        values = []
        for hour, month in zip(hours, pd.to_datetime(hours, unit='h').strftime('%Y-%m')):
            path = self.cache_root/'labels'/role/self._label_identity(role)[:16]/f'{month}.npz' if self.cache_root else None
            if self._has_shard(path):
                shard = self._load_shard(str(path))
                i = int(np.searchsorted(shard['hour_ids'], hour))
                if i < len(shard['hour_ids']) and shard['hour_ids'][i] == hour:
                    values.append(shard['values'][i])
                    continue
            values.append(self.store.targets(role, [int(hour)])[0])
        return np.asarray(values, 'f4')

    def batch(self, hour_ids, role='B', device='cpu'):
        hours = self._hours(hour_ids)
        self._check_role(role, hours)
        out = self.condition_batch(hours, device)
        ids = self.store.ids(role)
        values = self._targets(role, hours)
        current = out['station_tokens'][:, :, 20].cpu().numpy().astype(bool)
        support = np.stack([self._support(valid)[1][ids] for valid in current])
        mask = np.isfinite(values) & support
        def tensor(value):
            return torch.as_tensor(np.array(value, copy=True), device=device)
        out.update(query_indices=tensor(self._query_indices[ids]), query_weights=tensor(self._query_weights[ids]),
                   target_geometry=tensor(self._query_geometry[ids]), target=tensor(np.where(mask, values, 0).astype('f4')),
                   target_mask=tensor(mask))
        return out

    def audit(self, role='B'):
        years = self.config.train_years if role == 'B' else (2019,) if role == 'C' else ()
        hours = np.asarray(self.store.hours, dtype='i8')
        hours = hours[np.isin(pd.to_datetime(hours, unit='h').year, years)]
        self._check_role(role, hours)
        label_id = self._label_identity(role)
        binding = dict(schema=1, condition_identity=self.identity, labels_identity=label_id,
                       role=role, nominal_hours_sha256=fingerprint(hours.tolist()))
        if role in self._audits:
            report = self._audits[role]
            if report['condition_identity'] == self.identity and report['labels_identity'] == label_id:
                return report
            self._audits.pop(role)
        audit_path = self.cache_root/'audits'/f'{role}_{label_id[:16]}.json' if self.cache_root else None
        if audit_path is not None and audit_path.is_file():
            cached = read_json(audit_path)
            report = cached.get('report')
            if (cached.get('binding') != binding or not isinstance(report, dict) or
                    cached.get('report_sha256') != fingerprint(report) or
                    report.get('role') != role or report.get('condition_identity') != self.identity or
                    report.get('labels_identity') != label_id or report.get('hour_ids') != hours.tolist()):
                raise ValueError(f'Eligibility audit cache integrity/identity mismatch: {audit_path}')
            self._audits[role] = report
            return report
        reasons = np.full(len(hours), 'eligible', dtype='<U48')
        counts, raw_counts = np.zeros(len(hours), 'i4'), np.zeros(len(hours), 'i4')
        available = np.asarray(self.store.hours)
        historical = np.isin((hours[:, None]-np.array(FRAME_OFFSETS)).ravel(), available).reshape(-1, 4).all(1)
        reasons[~historical] = 'history_unavailable'
        ids = self.store.ids(role)
        for start in range(0, len(hours), 744):
            stop = min(start+744, len(hours))
            chunk = hours[start:stop]
            raw = self.store.context_values(chunk)
            values = self.store.targets(role, chunk)
            for j in range(len(chunk)):
                i = start+j
                finite = np.isfinite(values[j])
                raw_counts[i] = finite.sum()
                counts[i] = (finite & self._support(np.isfinite(raw[j]).all(-1))[1][ids]).sum()
                if historical[i] and counts[i] < 4:
                    reasons[i] = 'fewer_than_four_supported_labels'
        unique, n = np.unique(reasons, return_counts=True)
        eligible = reasons == 'eligible'
        report = dict(role=role, raw_hours=len(hours), qualified_hours=int(eligible.sum()), hour_ids=hours.tolist(),
                      eligible_hours=hours[eligible].tolist(), reason_codes=reasons.tolist(), reason_counts=dict(zip(unique.tolist(), n.tolist())),
                      raw_valid_label_counts=raw_counts.tolist(), eligible_label_counts=counts.tolist(),
                      reference_count=float(np.median(counts[eligible])) if eligible.any() else None,
                      condition_identity=self.identity, labels_identity=label_id,
                      raw_missing_label_hours=int((raw_counts < 4).sum()),
                      current_support_excluded_hours=int((historical & (raw_counts >= 4) & (counts < 4)).sum()))
        if audit_path is not None:
            # This CPU preflight is shared by subsequent independent Slurm stages.
            # The exact hours, masks/counts, exclusions and common reference count
            # are committed together; no observation or geometry pass is needed
            # for an unchanged condition/label release on the next instantiation.
            immutable_json(self.cache_root/'contract.json', self.metadata)
            write_json(audit_path, dict(binding=binding, report_sha256=fingerprint(report), report=report))
        self._audits[role] = report
        return report

    def _other_view(self, purpose):
        if purpose == self.purpose:
            return self
        return FullFieldData(self.config, purpose=purpose, background_provider=self.background_provider)

    @property
    def train_hours(self):
        return np.asarray(self._other_view('train').audit('B')['eligible_hours'], 'i8')

    @property
    def development_hours(self):
        return np.asarray(self._other_view('development').audit('C')['eligible_hours'], 'i8')

    def _clear_cache(self):
        self._monthly_cache.clear()

    def _commit_month(self, path, rows, *, identity):
        path = Path(path)
        if self._has_shard(path):
            old = self._load_shard(str(path))
            missing = ~np.isin(rows['hour_ids'], old['hour_ids'])
            if not missing.any():
                return
            rows = {key: np.concatenate((old[key], value[missing]), axis=0) for key, value in rows.items()}
        order = np.argsort(rows['hour_ids'])
        rows = {key: value[order] for key, value in rows.items()}
        import hashlib
        content_id = fingerprint({key: dict(shape=list(value.shape), dtype=str(value.dtype),
                        values=hashlib.sha256(np.ascontiguousarray(value).view('u1')).hexdigest())
                        for key, value in rows.items()})
        committed = path.with_name(f'{path.stem}.{content_id[:12]}')
        records = {}
        for key, value in rows.items():
            array_path = committed/f'{key}.npy'
            atomic_numpy(array_path, value)
            records[key] = dict(sha256=sha256(array_path), shape=list(value.shape), dtype=str(value.dtype))
        # Immutable uncompressed columns + one atomic month pointer. A crash
        # before pointer publication leaves the previous month fully readable.
        write_json(path.with_suffix('.json'), dict(schema=2, identity=identity, directory=committed.name,
                  hour_count=len(rows['hour_ids']), arrays=records))
        self._clear_cache()

    def _build_preparation_month(self, hours):
        chunks = [self._build_compact(hours[start:start+self.config.prepare_batch_hours])
                  for start in range(0, len(hours), self.config.prepare_batch_hours)]
        return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}

    def prepare(self, hours=None, *, role=None):
        if self.cache_root is None:
            raise ValueError('Explicit cache_root is required to prepare compact data')
        role = role or ('C' if self.purpose == 'development' else 'B')
        if hours is None:
            hours = self.development_hours if role == 'C' else self.train_hours
        hours = np.unique(self._hours(hours))
        self._check_role(role, hours)
        immutable_json(self.cache_root/'contract.json', self.metadata)
        months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
        label_id = self._label_identity(role)
        for month in sorted(set(months)):
            selected = hours[months == month]
            path = self.cache_root/'conditions'/f'{month}.npz'
            old_hours = self._load_shard(str(path))['hour_ids'] if self._has_shard(path) else np.array([], 'i8')
            missing = selected[~np.isin(selected, old_hours)]
            if len(missing):
                self._commit_month(path, self._build_preparation_month(missing), identity=self.identity)
            labels_path = self.cache_root/'labels'/role/label_id[:16]/f'{month}.npz'
            old_hours = self._load_shard(str(labels_path))['hour_ids'] if self._has_shard(labels_path) else np.array([], 'i8')
            missing = selected[~np.isin(selected, old_hours)]
            if len(missing):
                self._commit_month(labels_path, dict(hour_ids=missing, values=self.store.targets(role, missing)), identity=label_id)
        report = dict(prepared_hours=len(hours), condition_shards=len(set(months)), role=role,
                      condition_identity=self.identity, labels_identity=label_id,
                      cache_format='uncompressed_monthly_NPY_memmaps',
                      prepare_batch_hours=self.config.prepare_batch_hours,
                      mapped_month_capacity=self.config.mapped_month_capacity)
        if role in self._audits:
            write_json(self.cache_root/f'eligibility_{role}.json', self._audits[role])
            report['qualified_hours'] = self._audits[role]['qualified_hours']
            report['reason_counts'] = self._audits[role]['reason_counts']
        write_json(self.cache_root/f'preparation_{role}.json', report)
        return report


def prepare_cache(config, hours=None, *, purpose='train', background_provider=None, role=None):
    return FullFieldData(config, purpose=purpose, background_provider=background_provider).prepare(hours, role=role)
