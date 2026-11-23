from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def legacy_fixture(tmp_path):
    p = tmp_path / 'legacy'
    p.mkdir()
    n = 10
    xy = np.array([[200000, 200000], [250000, 200000], [200000, 250000], [250000, 250000],
                   [212500, 212500], [237500, 212500], [212500, 237500], [237500, 237500],
                   [220000, 225000], [230000, 225000]], dtype=float)
    roles = ['input']*4 + ['train']*4 + ['development', 'retrospective']
    frame = pd.DataFrame(dict(station=[f'S{i}' for i in range(n)], role=roles,
                              cohort=['dispersed']*n, x=xy[:, 0], y=xy[:, 1], height=np.arange(n)*20.,
                              physical_group=np.arange(n)))
    frame.to_csv(p/'station_roles.csv', index=False)
    ax = (np.arange(24)+.5)*25000.
    x, y = np.meshgrid(ax, ax)
    np.savez(p/'geometry.npz', height=np.full((24, 24), 100., dtype='f4'), x=x, y=y,
             region=np.ones((24, 24), bool), origin=np.array([0., 0.]), origins=np.array([[0, 0]]))
    values = np.zeros((96, n, 5), dtype='f4')
    values[..., 0] = 100 + np.arange(96)[:, None]*.1 + np.arange(n)[None]*.3
    values[..., 1:] = np.array([280., 950., 270., 20.])
    values[..., 1] += np.arange(96)[:, None]*.001
    np.save(p/'observations.npy', values)
    start = int(np.datetime64('2016-01-01', 'h').astype('i8'))
    (p/'audit.json').write_text(json.dumps(dict(start=start, stop=start+96)))
    return p
