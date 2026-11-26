"""Exact finite-label populations and current-condition distance histograms."""
import json
from pathlib import Path
import sys
import numpy as np

from precision_analysis import ENGINE
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.p14_fusion import chronological_blocks
from tools.terrain_diffusion.station_query import hole_schedule


def histogram(values, hours, geo, seed, holes):
    rng = np.random.default_rng(seed+28011)
    stop = int(np.searchsorted(hours, np.datetime64('2020-01-01', 'h').astype('int64')))
    blocks = chronological_blocks(71, stop, 12)
    counts = np.zeros(4, dtype=np.int64); current_missing = 0; empty = 0
    station_counts = np.zeros((len(geo['xy']), 4), dtype=np.int64)
    distances = np.linalg.norm(geo['xy'][:, None]-geo['xy'][None], axis=-1)
    for step in range(3000):
        anchors = blocks[step % len(blocks)]
        if holes[step]:
            center = int(rng.integers(410)); pool = np.flatnonzero(distances[center] > 100)
        else: pool = np.arange(410)
        c = np.sort(rng.choice(pool, 82, False))
        hidden = ~np.isin(geo['groups'], geo['groups'][c])
        target_ok = np.isfinite(values[anchors, :, 0]) & hidden[None]
        current_ok = np.isfinite(values[anchors[:, None], c[None], 0])
        near = np.where(current_ok[:, None], distances[:, c][None], np.inf).min(-1)
        empty += int((target_ok & ~np.isfinite(near)).sum())
        index = np.searchsorted([50, 100, 200], near, side='right')
        for k in range(4):
            membership = (index == k) & target_ok & np.isfinite(near)
            count = membership.sum(0); station_counts[:, k] += count; counts[k] += count.sum()
        current_missing += int((~current_ok).sum())
    valid_stations = station_counts.sum(1) > 0
    macro = (station_counts[valid_stations]/station_counts[valid_stations].sum(1, keepdims=True)).mean(0)
    return dict(seed=seed, hole_blocks=int(holes.sum()), valid_targets_by_distance=counts.tolist(),
        record_fractions=(counts/counts.sum()).tolist(), station_macro_fractions=macro.tolist(),
        no_current_condition_targets=empty, missing_current_condition_records=current_missing)


def main():
    path = ENGINE/'results/20261125_temporal_continuous_data'
    values = np.load(path/'values.npy', mmap_mode='r'); hours = np.load(path/'hours.npy')
    geo = dict(np.load(path/'geometry.npz')); rows = []
    for seed in (42, 17):
        for regime in ('random', 'quarter_hole100'):
            holes = hole_schedule(3000, seed) if regime != 'random' else np.zeros(3000, dtype=bool)
            result = dict(regime=regime, **histogram(values, hours, geo, seed, holes))
            rows.append(result); print(json.dumps(result), flush=True)
    out = ENGINE/'results/20261125_support_distribution_audit'; out.mkdir(exist_ok=True)
    development_distance=[]
    for layout_seed in (8120,8121,8122):
        c=np.sort(np.random.default_rng(layout_seed).choice(410,82,False))
        development_distance.extend(np.linalg.norm(geo['dev_xy'][:,None]-geo['xy'][c][None],axis=-1).min(-1))
    metadata_counts=np.histogram(development_distance,[0,50,100,200,np.inf])[0]
    target_fraction=metadata_counts/metadata_counts.sum()
    train_fraction=np.mean([r['record_fractions'] for r in rows if r['regime']=='quarter_hole100'],axis=0)
    ratio=target_fraction/train_fraction
    from tools.terrain_diffusion.station_weighting import BIN_WEIGHTS
    np.testing.assert_allclose(np.round(ratio,3),BIN_WEIGHTS)
    write_json(out/'results.json', dict(code_sha256=sha(Path(__file__)),
        data_contract_sha256=sha(path/'contract.json'), distance_bins_km=['0-50', '50-100', '100-200', '200+'],
        population='Exact 3000 continuous12h training blocks, hidden physical groups, finite training truth',
        no_model_fitting=True, results=rows, distance_weight_derivation=dict(
            target='137 registered development coordinates x3 condition layouts; geometry only, no truth or validity mask',
            metadata_counts=metadata_counts.tolist(),target_fractions=target_fraction.tolist(),
            training_quarter_hole_fractions=train_fraction.tolist(),ratio=ratio.tolist(),
            rounded_fixed_weights=list(BIN_WEIGHTS))))


if __name__ == '__main__': main()
