"""Training-hour qualification is independent of development/retrospective labels."""
from dataclasses import dataclass
from pathlib import Path
import io

import numpy as np
import pandas as pd

from gnss_field.artifacts import atomic_bytes, fingerprint, read_json, sha256, write_json
from .context import ContextBuilder


@dataclass
class TrainingManifest:
    hours: np.ndarray
    valid: np.ndarray
    station_ids: np.ndarray
    contract: str

    def select(self, years, minimum=1, station_mask=None):
        mask = np.ones(self.valid.shape[1], bool) if station_mask is None else station_mask
        good = self.valid[:, mask].sum(-1) >= minimum
        good &= np.isin(pd.to_datetime(self.hours, unit='h').year, years)
        return self.hours[good]

    def masks(self, hours):
        idx = np.searchsorted(self.hours, hours)
        if (idx >= len(self.hours)).any() or not np.array_equal(self.hours[idx], hours):
            raise ValueError('Hour absent from training manifest')
        return self.valid[idx]

    def weights(self, selected_hours):
        count = self.masks(selected_hours).sum(0)
        n = max(1, (count > 0).sum())
        return np.where(count > 0, len(selected_hours)/(n*np.maximum(count, 1)), 0).astype('f4')

    def audit(self, years, minimum):
        years_array = pd.to_datetime(self.hours, unit='h').year
        counts = self.valid.sum(-1)
        d = pd.to_datetime(self.hours, unit='h')
        rows = []
        for year in years:
            for month in range(1, 13):
                group = (years_array == year) & (d.month == month)
                rows.append(dict(year=year, month=month, before=int(group.sum()),
                                 after=int((group & (counts >= minimum)).sum())))
        return rows


def build_training_manifest(store, destination):
    if store.purpose != 'train':
        raise PermissionError('Training qualification requires the B training view')
    destination = Path(destination)
    signature = fingerprint(dict(training=store.manifest['training_hash'], policy='B_valid_A_support_full_stencil_v2'))
    if destination.exists():
        meta = read_json(destination.with_suffix('.json'))
        if meta['contract'] != signature or sha256(destination) != meta['sha256']:
            raise ValueError('Qualification cache contract mismatch')
        z = np.load(destination)
        return TrainingManifest(z['hours'], z['valid'], z['station_ids'], signature)
    hours = np.asarray(store.hours)
    years = pd.to_datetime(hours, unit='h').year
    hours = hours[np.isin(years, [2016, 2017, 2018]) & (hours >= store.manifest['start']+18)]
    builder = ContextBuilder(store)
    patterns = np.isfinite(store.context_values(hours)).all(-1)
    unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
    supported = []
    ids = store.ids('B')
    for j, pattern in enumerate(unique):
        _, grid = builder._geometry(pattern.tobytes(), 'grid')
        _, station = builder._geometry(pattern.tobytes(), 'stations')
        mask = (grid['support'] & builder.label_near_grid & store.grid['region'].ravel()).reshape(store.shape)
        x = np.floor(store.qxy[:, 0]).astype(int).clip(0, store.shape[1]-2)
        y = np.floor(store.qxy[:, 1]).astype(int).clip(0, store.shape[0]-2)
        ok = (station['support'] & builder.label_near_query & store.query_valid
              & mask[y, x] & mask[y+1, x] & mask[y, x+1] & mask[y+1, x+1])
        supported.append(ok[ids])
        if j % 100 == 0:
            print(f'QUALIFY {j}/{len(unique)} input patterns', flush=True)
    targets = store.targets('B', hours)
    valid = np.asarray(supported)[inverse] & np.isfinite(targets)
    buf = io.BytesIO()
    np.savez_compressed(buf, hours=hours, valid=valid, station_ids=ids)
    atomic_bytes(destination, buf.getvalue())
    write_json(destination.with_suffix('.json'), dict(contract=signature, sha256=sha256(destination),
               hours=len(hours), stations=len(ids), source='real_B_and_A_geometry_only'))
    return TrainingManifest(hours, valid, ids, signature)


def stratified_hours(hours, year, per_cell=2):
    hours = np.asarray(hours)
    d = pd.to_datetime(hours, unit='h')
    chosen = []
    for month in range(1, 13):
        for clock in range(24):
            rows = np.flatnonzero((d.year == year) & (d.month == month) & (d.hour == clock))
            if len(rows):
                positions = np.minimum(len(rows)-1, (len(rows)*np.arange(1, per_cell+1)/(per_cell+1)).astype(int))
                chosen.extend(hours[rows[np.unique(positions)]].tolist())
    return np.array(sorted(set(chosen)), dtype='i8')


def spatial_folds(store, seed=20261117, folds=3):
    groups = store.stations.iloc[store.ids('B')].dense_group.to_numpy()
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    mapping = {group: i % folds for i, group in enumerate(unique)}
    return np.array([mapping[group] for group in groups])
