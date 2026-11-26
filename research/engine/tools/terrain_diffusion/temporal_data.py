"""Role-isolated hourly data and causal spatial histories for temporal screening."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .data import ASSETS, read_raw, sha, static_geometry, write_json
from .coast_features import load_coast, distance_to_coast


def anchors_for_period(hours, length, horizon, start, stop):
    hours = np.asarray(hours)
    if np.any(np.diff(hours) != 1):
        raise ValueError('hours must be continuous')
    indices = np.arange(length-1, len(hours)-horizon)
    return indices[(hours[indices-length+1] >= start) & (hours[indices+horizon] < stop)]


def spatial_history(values, hours, anchor, length, context, xy, height, grid):
    if anchor-length+1 < 0:
        raise ValueError('insufficient causal history')
    raw = values[anchor-length+1:anchor+1, context].astype('float32', copy=True)
    valid = np.isfinite(raw)
    raw[:, :, 0] *= np.exp(height[context]/3000)[None]
    normalized = (raw - np.array([0, 280, 900]))/np.array([100, 20, 100])
    normalized = np.where(valid, normalized, 0)
    age = np.zeros(valid.shape[:2], dtype='float32')
    for k in range(length):
        age[k] = np.where(valid[k, :, 0], 0, age[k-1]+1 if k else length)
    distance = cdist(grid, xy[context])
    features = []
    for radius in (50, 150, 300):
        weight = np.exp(-.5*(distance/radius)**2)*(distance <= radius*3)
        den = np.einsum('mc,lcj->lmj', weight, valid)
        avg = np.einsum('mc,lcj->lmj', weight, normalized)/np.maximum(den, 1e-8)
        coverage = den/np.maximum(weight.sum(1)[None, :, None], 1e-8)
        ages = np.einsum('mc,lc->lm', weight, age)/np.maximum(weight.sum(1)[None], 1e-8)
        features.extend([avg, coverage, np.log1p(ages)[..., None]])
    h = hours[anchor-length+1:anchor+1]
    time = np.stack([np.sin(h%24*2*np.pi/24), np.cos(h%24*2*np.pi/24),
                     np.sin(h*2*np.pi/(365.25*24)), np.cos(h*2*np.pi/(365.25*24))], -1)
    features.append(np.broadcast_to(time[:, None], (length, len(grid), 4)))
    return np.concatenate(features, -1).astype('float32')


def prepare(out, workers=4):
    out = Path(out)
    if (out/'contract.json').exists():
        raise FileExistsError('Existing data contract is immutable; use another output')
    out.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(ASSETS/'station_assignment_proposal.csv')
    train = stations[stations.new_role == 'training'].reset_index(drop=True)
    dev = stations[stations.new_role == 'validation'].reset_index(drop=True)
    hours = np.arange(np.datetime64('2016-01-01', 'h').astype('int64'),
                      np.datetime64('2021-01-01', 'h').astype('int64'))
    dev_hours = hours[hours >= np.datetime64('2020-01-01', 'h').astype('int64')]
    values = np.lib.format.open_memmap(out/'values.npy', mode='w+', dtype='float32', shape=(len(hours), len(train), 3))
    dev_values = np.lib.format.open_memmap(out/'dev_values.npy', mode='w+', dtype='float32', shape=(len(dev_hours), len(dev), 3))
    sources = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for role, frame, hh in [('training', train, hours), ('validation', dev, dev_hours)]:
            for i, station in enumerate(frame.station):
                futures[pool.submit(read_raw, station, hh)] = (role, i)
        for n, future in enumerate(as_completed(futures), 1):
            role, i = futures[future]
            array, receipt = future.result()
            (values if role == 'training' else dev_values)[:, i] = array
            sources.append(dict(role=role, **receipt))
            if n % 20 == 0:
                print(json.dumps(dict(stage='raw_qc', done=n, total=len(futures))), flush=True)
    values.flush(); dev_values.flush()
    write_json(out/'raw_qc_receipt.json', dict(status='RAW_READ_COMPLETE',
        sources=sorted(sources,key=lambda s:(s['role'],s['station'])),
        values_sha256=sha(out/'values.npy'), dev_values_sha256=sha(out/'dev_values.npy')))
    np.save(out/'hours.npy', hours); np.save(out/'dev_hours.npy', dev_hours)
    train_static, _ = static_geometry(train)
    dev_static, _ = static_geometry(dev)
    lines, coast = load_coast(out/'coast_source')
    coast_train = distance_to_coast(train[['x','y']].to_numpy()/1000, lines)
    coast_dev = distance_to_coast(dev[['x','y']].to_numpy()/1000, lines)
    np.savez(out/'geometry.npz', xy=train[['x','y']].to_numpy()/1000,
             dev_xy=dev[['x','y']].to_numpy()/1000, height=train.height.to_numpy(),
             dev_height=dev.height.to_numpy(), static=train_static, dev_static=dev_static,
             coast=coast_train, dev_coast=coast_dev, groups=train.physical_group.to_numpy(),
             station_ids=train.station.to_numpy(dtype='U'), dev_ids=dev.station.to_numpy(dtype='U'))
    write_json(out/'contract.json', dict(status='CONTINUOUS_NOMINAL_HOURLY_DEV_CONTRACT',
        train_period='2016-2018 inner fit; 2019 inner selection; 2016-2019 final refit',
        dev_period='2020; validation137 only; development not blind',
        inputs=['ZWD_mm','temperature_K','pressure_hPa','geometry','terrain','coast_distance_km'],
        forbidden=['PWV','IWV','TCWV','TM'], training_stations=len(train), validation_stations=len(dev),
        confirmation_C_D_read=False, time_basis='source nominal product hour; no invented UTC/GPST conversion',
        sources=sorted(sources,key=lambda s:(s['role'],s['station'])), coast=coast,
        cohort='Inherited historical station selection; allowed-variable record QC rebuilt',
        files={p.name:sha(p) for p in out.glob('*') if p.is_file()},
        code_sha256=sha(__file__)))
    print(json.dumps(dict(stage='data_complete',out=str(out))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare(args.out, args.workers)
