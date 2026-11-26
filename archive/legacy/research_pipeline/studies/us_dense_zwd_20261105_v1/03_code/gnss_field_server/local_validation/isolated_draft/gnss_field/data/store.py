"""Role-separated, monthly immutable arrays; no cache creation on reads."""
from functools import lru_cache
from pathlib import Path
import io

import numpy as np
import pandas as pd

from gnss_field.artifacts import (atomic_bytes, atomic_numpy, fingerprint,
                                  read_json, sha256, write_json)

ROLE_NAMES = {'A': 'input', 'B': 'train', 'C': 'development', 'D': 'retrospective'}
REGISTERED_COUNTS = {'A': 174, 'B': 511, 'C': 110, 'D': 110}


def pack_legacy(source, destination, *, registered=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Packed output must be separate from legacy data')
    source_hashes = {name: sha256(source/name) for name in
                     ('station_roles.csv', 'geometry.npz', 'observations.npy', 'audit.json')}
    if (destination/'manifest.json').exists():
        old = read_json(destination/'manifest.json')
        if old['sources'] != source_hashes or old['registered'] != registered:
            raise ValueError('Source changed; create a new data-pack directory')
        return old
    destination.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(source/'station_roles.csv')
    ids = {key: np.flatnonzero(r.role == role) for key, role in ROLE_NAMES.items()}
    counts = {key: len(value) for key, value in ids.items()}
    if registered and counts != REGISTERED_COUNTS:
        raise ValueError(f'Unexpected registered physical station counts: {counts}')
    a = np.load(source/'observations.npy', mmap_mode='r')
    audit = read_json(source/'audit.json')
    hours = np.arange(audit['start'], audit['stop'], dtype='i8')
    if a.shape != (len(hours), len(r), 5):
        raise ValueError('Observation dimensions do not match the source audit')
    dates = pd.to_datetime(hours, unit='h')
    base = np.isin(dates.year, [2016, 2017])
    inp = np.asarray(a[base][:, ids['A']], dtype='f8')
    valid = np.isfinite(inp).all(-1)
    inp[~valid] = np.nan
    target = np.asarray(a[base][:, ids['B'], 0], dtype='f8')
    s0 = float(np.nanstd(target))
    if not np.isfinite(s0) or s0 < 1e-6:
        raise ValueError('Degenerate training Z0 scale')
    stats = dict(input_mean=np.nanmean(inp, axis=(0, 1)).tolist(),
                 input_std=np.maximum(np.nanstd(inp, axis=(0, 1)), 1e-6).tolist(),
                 z0_std=s0, surface_std=float(np.nanstd(target*np.exp(-r.height.to_numpy()[ids['B']]/3000))),
                 statistics_years=[2016, 2017], input_role='A', label_role='B', height_datum='EGM96')
    if not np.isfinite(np.array(stats['input_mean'])).all():
        raise ValueError('No jointly valid input observations for statistics')
    keep = ['station', 'role', 'cohort', 'x', 'y', 'height', 'physical_group']
    metadata = r[[c for c in keep if c in r]].copy()
    metadata['dense_group'] = [f'{int(np.floor(x/50000))}:{int(np.floor(y/50000))}'
                               for x, y in metadata[['x', 'y']].to_numpy()]
    atomic_bytes(destination/'stations.csv', metadata.to_csv(index=False).encode('utf-8'))
    atomic_bytes(destination/'geometry.npz', (source/'geometry.npz').read_bytes())
    atomic_numpy(destination/'hours.npy', hours)
    write_json(destination/'statistics.json', stats)
    files = {}
    months = {}
    for month in sorted(set(dates.strftime('%Y-%m'))):
        sel = np.flatnonzero(dates.strftime('%Y-%m') == month)
        months[month] = dict(start=int(hours[sel[0]]), stop=int(hours[sel[-1]]+1))
        for role, indices in ids.items():
            values = np.asarray(a[sel][:, indices], dtype='f4')
            if role != 'A':
                values = values[..., 0]*np.exp(-r.height.to_numpy()[indices]/3000).astype('f4')
            name = f'{role}/{month}.npy'
            atomic_numpy(destination/name, values)
            files[name] = dict(sha256=sha256(destination/name), shape=list(values.shape), dtype=str(values.dtype))
    shared = {name: sha256(destination/name) for name in ('stations.csv', 'geometry.npz', 'statistics.json', 'hours.npy')}
    train_files = {name: info for name, info in files.items()
                   if name[0] in ('A', 'B') and name[2:6] in ('2016', '2017', '2018')}
    manifest = dict(schema=1, registered=registered, kind='observational' if registered else 'synthetic_test',
                    sources=source_hashes, shared=shared, files=files, months=months, counts=counts,
                    ids={k: v.tolist() for k, v in ids.items()}, start=int(hours[0]), stop=int(hours[-1]+1),
                    training_hash=fingerprint(dict(shared=shared, files=train_files)),
                    label_units='surface_ZWD_mm', input_first_variable='Z0_mm',
                    time='nominal_product_hour', coarse_source='A_IDW_not_physical_grid')
    manifest['fingerprint'] = fingerprint(manifest)
    write_json(destination/'manifest.json', manifest)
    return manifest


class PackedStore:
    def __init__(self, root, *, purpose='context', retrospective_lock=None):
        self.root = Path(root)
        self.manifest = read_json(self.root/'manifest.json')
        expected = dict(self.manifest)
        digest = expected.pop('fingerprint')
        if fingerprint(expected) != digest:
            raise ValueError('Manifest hash mismatch')
        for name, digest in self.manifest['shared'].items():
            if sha256(self.root/name) != digest:
                raise ValueError(f'Shared data contract mismatch: {name}')
        if purpose not in ('context', 'train', 'development', 'retrospective', 'audit'):
            raise ValueError('Unknown data read purpose')
        if purpose == 'retrospective':
            lock = retrospective_lock
            if (not isinstance(lock, dict) or not lock.get('products') or
                    lock.get('selection_year') != 2019 or lock.get('retrospective_role') != 'D' or
                    lock.get('retrospective_year') != 2020 or
                    lock.get('manifest_sha256') != self.manifest['fingerprint'] or
                    lock.get('lock_sha256') != fingerprint({k: v for k, v in lock.items() if k != 'lock_sha256'})):
                raise PermissionError('Retrospective labels require a dataset-bound, integrity-validated product lock')
        self.purpose = purpose
        self.stats = read_json(self.root/'statistics.json')
        self.stations = pd.read_csv(self.root/'stations.csv')
        self.hours = np.load(self.root/'hours.npy', mmap_mode='r')
        with np.load(self.root/'geometry.npz') as archive:
            self.grid = {name: archive[name] for name in archive.files}
        self.shape = self.grid['height'].shape
        self.grid_id = self.manifest['shared']['geometry.npz']
        self.xy = self.stations[['x', 'y']].to_numpy()
        self.qxy = (self.xy - self.grid['origin'])/25000 - .5
        h, w = self.shape
        self.query_valid = (np.isfinite(self.qxy).all(-1) & (self.qxy >= 0).all(-1)
                            & (self.qxy[:, 0] <= w-1) & (self.qxy[:, 1] <= h-1))
        safe = self.qxy.clip([0, 0], [w-1, h-1]).round().astype(int)
        self.query_valid &= self.grid['region'][safe[:, 1], safe[:, 0]].astype(bool)

    def ids(self, role):
        return np.array(self.manifest['ids'][role], dtype=int)

    @lru_cache(maxsize=16)
    def _shard(self, role, month):
        name = f'{role}/{month}.npy'
        if sha256(self.root/name) != self.manifest['files'][name]['sha256']:
            raise ValueError(f'Array checksum mismatch: {name}')
        return np.load(self.root/name, mmap_mode='r')

    def _read(self, role, hours):
        hours = np.asarray(hours, dtype='i8').reshape(-1)
        if not len(hours) or (hours < self.manifest['start']).any() or (hours >= self.manifest['stop']).any():
            raise ValueError('Requested nominal hours outside data coverage')
        months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
        out = []
        for hour, month in zip(hours, months):
            out.append(np.asarray(self._shard(role, month)[int(hour)-self.manifest['months'][month]['start']]))
        return np.stack(out)

    def context_values(self, hours):
        return self._read('A', hours)

    def targets(self, role, hours):
        years = pd.to_datetime(np.asarray(hours), unit='h').year
        if self.purpose == 'train':
            allowed = role == 'B' and np.isin(years, [2016, 2017, 2018]).all()
        elif self.purpose == 'development':
            allowed = role == 'C' and (years == 2019).all()
        elif self.purpose == 'retrospective':
            allowed = role == 'D' and (years == 2020).all()
        else:
            allowed = self.purpose == 'audit'
        if not allowed:
            raise PermissionError(f'{self.purpose} view cannot read {role} labels at these hours')
        return self._read(role, hours)

    def verify_all(self):
        for name, info in self.manifest['files'].items():
            if sha256(self.root/name) != info['sha256']:
                raise ValueError(f'Corrupt data shard: {name}')
        return dict(files=len(self.manifest['files']), fingerprint=self.manifest['fingerprint'], verified=True)
