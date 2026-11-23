"""Input-only conditions. Missing data and outages never consult target masks."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree

from gnss_field.artifacts import fingerprint
from gnss_field.contracts import ContextBatch
from gnss_field.geometry import input_geometry


@dataclass(frozen=True)
class InputScenario:
    retained: tuple[int, ...] | None = None
    persistent: bool = False

    @property
    def identity(self):
        return fingerprint(dict(retained=self.retained, persistent=self.persistent))


def cycles(hours):
    d = pd.to_datetime(np.asarray(hours), unit='h')
    annual = 2*np.pi*(d.dayofyear.to_numpy()-1+d.hour.to_numpy()/24)/365.25
    daily = 2*np.pi*d.hour.to_numpy()/24
    return np.stack([np.sin(annual), np.cos(annual), np.sin(daily), np.cos(daily)], -1).astype('f4')


class ContextBuilder:
    def __init__(self, store, *, use_history=True):
        self.store = store
        self.use_history = use_history
        self.ids = store.ids('A')
        self.xy = store.xy[self.ids]
        self.heights = store.stations.height.to_numpy()[self.ids]
        self.mean = np.array(store.stats['input_mean'])
        self.std = np.array(store.stats['input_std'])
        self.fine_xy = np.c_[store.grid['x'].ravel(), store.grid['y'].ravel()]
        h, w = store.shape
        cy, cx = np.meshgrid(np.arange(h//4), np.arange(w//4), indexing='ij')
        self.coarse_xy = store.grid['origin'] + np.c_[(cx.ravel()+.5)*100000, (cy.ravel()+.5)*100000]
        self.label_near_grid = cKDTree(store.xy[store.ids('B')]).query(self.fine_xy)[0] <= 100000
        self.label_near_query = cKDTree(store.xy[store.ids('B')]).query(store.xy)[0] <= 100000

    def _available(self, values, scenario, apply):
        valid = np.isfinite(values).all(-1)
        if apply and scenario.retained is not None:
            keep = np.zeros(len(self.ids), bool)
            keep[list(scenario.retained)] = True
            valid &= keep
        return valid

    @lru_cache(maxsize=256)
    def _geometry(self, pattern, kind):
        selected = np.flatnonzero(np.frombuffer(pattern, dtype='?'))
        queries = {'grid': self.fine_xy, 'coarse': self.coarse_xy, 'stations': self.store.xy}[kind]
        return selected, input_geometry(self.xy[selected], queries)

    def support(self, hour, scenario=InputScenario()):
        values = self.store.context_values([hour])[0]
        valid = self._available(values, scenario, True)
        _, grid = self._geometry(valid.tobytes(), 'grid')
        _, station = self._geometry(valid.tobytes(), 'stations')
        support_grid = grid['support'] & self.label_near_grid & self.store.grid['region'].ravel()
        support_station = station['support'] & self.label_near_query & self.store.query_valid
        m = support_grid.reshape(self.store.shape)
        q = self.store.qxy
        x = np.floor(q[:, 0]).astype(int).clip(0, self.store.shape[1]-2)
        y = np.floor(q[:, 1]).astype(int).clip(0, self.store.shape[0]-2)
        support_station &= m[y, x] & m[y+1, x] & m[y, x+1] & m[y+1, x+1]
        return support_grid.reshape(self.store.shape), support_station

    def _coarse(self, values, valid):
        selected, geo = self._geometry(valid.tobytes(), 'coarse')
        if not len(selected):
            raw = np.broadcast_to(self.mean, (len(self.coarse_xy), 5)).copy()
        elif len(selected) == 1:
            raw = np.broadcast_to(values[selected[0]], (len(self.coarse_xy), 5)).copy()
        else:
            raw = (np.nan_to_num(values[selected][geo['indices']])*geo['weights'][..., None]).sum(1)
            if len(selected) >= 3:
                raw[~geo['support']] = 0  # registered legacy normal-input background contract
        return raw, geo

    def build_context(self, hours, scenario=InputScenario(), *, device='cpu'):
        hours = np.asarray(hours, dtype='i8').reshape(-1)
        if (hours-18 < self.store.manifest['start']).any():
            raise ValueError('A full 18h historical index must exist')
        h, w = self.store.shape
        hc, wc = h//4, w//4
        station_rows, pads, frames, maps, bases = [], [], [], [], []
        for hour in hours:
            values = self.store.context_values([hour])[0]
            valid = self._available(values, scenario, True)
            _, g = self._geometry(valid.tobytes(), 'grid')
            support = (g['support'] & self.label_near_grid & self.store.grid['region'].ravel()).reshape(h, w)
            m = np.stack([self.store.grid['height']/3000, self.store.grid['x']/400000,
                          self.store.grid['y']/400000, support,
                          np.minimum(g['nearest'].reshape(h, w)/400000, 2.5), g['count'].reshape(h, w)/16])
            token = np.zeros((len(self.ids), 17), dtype='f4')
            token[:, :5] = np.nan_to_num((values-self.mean)/self.std)
            token[:, 5:10] = valid[:, None]
            token[:, 10:13] = np.c_[self.xy/400000, self.heights/3000]
            token[:, 13:] = cycles([hour])[0]
            token[~valid] = 0
            rows = []
            for j, offset in enumerate((0, 6, 12, 18)):
                v = self.store.context_values([hour-offset])[0]
                ok = self._available(v, scenario, offset == 0 or scenario.persistent)
                raw, geo = self._coarse(v, ok)
                frame = np.zeros((15, hc, wc), dtype='f4')
                frame[:5] = ((raw-self.mean)/self.std).T.reshape(5, hc, wc)
                frame[:5, ~geo['support'].reshape(hc, wc)] = 0
                frame[5] = geo['support'].reshape(hc, wc)
                frame[6:9] = m[:3].reshape(3, hc, 4, wc, 4).mean((2, 4))
                frame[9:13] = cycles([hour-offset])[0, :, None, None]
                frame[13:15] = m[4:6].reshape(2, hc, 4, wc, 4).mean((2, 4))
                if not self.use_history and j:
                    frame[:6] = 0
                rows.append(frame)
                if not offset:
                    bases.append(raw[:, 0].reshape(1, hc, wc))
            station_rows.append(token)
            pads.append(~valid)
            frames.append(np.stack(rows))
            maps.append(m)
        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(np.array(x, copy=True), dtype=dtype, device=device)
        coarse_z0 = tensor(bases)
        return ContextBatch(hours=tensor(hours, torch.int64), stations=tensor(station_rows),
                            padding=tensor(pads, torch.bool), coarse=tensor(frames), maps=tensor(maps),
                            baseline=F.interpolate(coarse_z0, (h, w), mode='bilinear', align_corners=False),
                            grid_id=self.store.grid_id, scenario_hash=scenario.identity, coarse_z0=coarse_z0)
