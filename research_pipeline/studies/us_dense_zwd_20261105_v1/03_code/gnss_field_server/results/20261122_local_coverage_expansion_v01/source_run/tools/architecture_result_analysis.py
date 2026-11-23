"""Locked-product point diagnostics, preserving the complete C2019 population."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256
from gnss_field.evaluation import point_metrics
from gnss_field.evaluation.bootstrap import _blocks
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.training.observable import load_bundle
from gnss_field.data import PackedStore
from architecture_bottleneck_diagnostics import PROJECT, ROOT, PILOT


def rmse_interval(hours, truth, candidate, baseline, population, *, days=7, groups=None, replicates=2000):
    blocks = _blocks(hours, days)
    active = population.any(0)
    pop = population[:, active]
    errors = [np.where(pop, (pred[:, active]-truth[:, active])**2, 0.) for pred in (candidate, baseline)]
    if any(not np.isfinite(e).all() for e in errors):
        raise ValueError('Nonfinite scored prediction')
    sums = [np.array([e[ix].sum(0) for ix in blocks]) for e in errors]
    counts = np.array([pop[ix].sum(0) for ix in blocks])
    labels = np.asarray(groups)[active] if groups is not None else None
    unique = list(dict.fromkeys(labels)) if labels is not None else None
    rng = np.random.default_rng(20261117)
    draws = []
    for _ in range(replicates*100):
        if len(draws) == replicates:
            break
        selected = rng.integers(0, len(blocks), len(blocks))
        denominator = counts[selected].sum(0)
        if not (denominator > 0).all():
            continue
        difference = np.sqrt(sums[0][selected].sum(0)/denominator)-np.sqrt(sums[1][selected].sum(0)/denominator)
        weights = np.ones(len(difference))
        if unique is not None:
            chosen = rng.integers(0, len(unique), len(unique))
            multiplicity = np.bincount(chosen, minlength=len(unique))
            weights = np.array([multiplicity[unique.index(label)] for label in labels])
        draws.append(np.average(difference, weights=weights))
    if len(draws) != replicates:
        raise ValueError('Could not obtain complete paired bootstrap populations')
    observed = point_metrics(truth, candidate, population)['station_macro_rmse_mm']-point_metrics(truth, baseline, population)['station_macro_rmse_mm']
    return dict(difference_mm=observed, ci95_mm=np.quantile(draws, [.025, .975]).tolist(),
        days=days, replicates=replicates, stations=int(active.sum()),
        groups=len(unique) if unique is not None else None,
        resampling='Synchronous time blocks'+(' crossed with fixed 50km station groups' if groups is not None else ''))


def predict(path, arrays, expected):
    if sha256(path) != expected:
        raise ValueError('Locked checkpoint differs')
    current = SubspaceMeanShift().eval().requires_grad_(False)
    current.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['ema'])
    with torch.no_grad():
        latent = current.net(torch.tensor(arrays['condition']))
        return arrays['mean']+(latent@torch.tensor(arrays['projection']).T).numpy()*arrays['amplitude'][:, None]


def main():
    torch.set_num_threads(2)
    c = dict(np.load(PILOT/'server_evaluation/data.npz'))
    meta = read_json(PILOT/'server_evaluation/manifest.json')
    if meta['role'] != 'C' or meta['year'] != 2019 or sha256(PILOT/'server_evaluation/data.npz') != meta['data_sha256']:
        raise ValueError('Only frozen exposed C2019 is accepted')
    b, _ = load_bundle(PILOT/'server_bundle')
    old_path = PROJECT/'results/20261120_postcoverage_analysis_v01/artifacts/mean/selected.pt'
    new_path = ROOT/'mean_selected.pt'
    immutable_json(ROOT/'point_diagnostics.lock.json', dict(source_sha256=sha256(__file__),
        C_manifest_sha256=sha256(PILOT/'server_evaluation/manifest.json'),
        old_checkpoint_sha256=sha256(old_path), new_checkpoint_sha256=sha256(new_path),
        point_products_locked=True, C_used_for_diagnostics_only=True, D2020=False))
    old = predict(old_path, c, 'ae04939fc897e1c6c4a2a22e68b5d7581d060d926402eb480cbbf1990b2e7b31')
    new = predict(new_path, c, 'd1096574109d89bdcc2b318f6d2d3f764095279fa9f99a07412633bbb6725a6c')
    archived = read_json(PROJECT/'results/20261120_mean_convergence_v01/server_results.json')['C2019_metrics']['point']
    metrics = point_metrics(c['truth'], new, c['valid'])
    assert abs(metrics['station_macro_rmse_mm']-archived['station_macro_rmse_mm']) < .001
    station = pd.read_csv(PILOT/'server_evaluation/stations.csv')
    intervals = {str(days): {name: rmse_interval(c['hours'], c['truth'], new, old, c['valid'], days=days,
        groups=station.dense_group.to_numpy() if name == 'crossed' else None) for name in ('time', 'crossed')} for days in (7, 14)}
    table = station.copy()
    table['scored_hours'] = c['valid'].sum(0)
    for name, pred in [('old', old), ('new', new)]:
        count = np.maximum(c['valid'].sum(0), 1)
        table[name+'_RMSE_mm'] = np.sqrt(np.where(c['valid'], (pred-c['truth'])**2, 0.).sum(0)/count)
        table[name+'_bias_mm'] = np.where(c['valid'], pred-c['truth'], 0.).sum(0)/count
    table.to_csv(ROOT/'C_station_diagnostics.csv', index=False, encoding='utf-8-sig')
    store = PackedStore(PROJECT/'../../02_data/processed/gnss_field_server_v1', purpose='train')
    new_b = predict(new_path, b, sha256(new_path))
    ix = b['training_indices']; pop = b['valid'][ix]
    slope = np.exp(-store.stations.height.to_numpy()[b['station_ids']]/3000)
    weight = pop/np.maximum(pop.sum(0), 1)[None]
    delta = float(np.sum(weight*slope*(np.where(pop, b['truth'][ix]-new_b[ix], 0.)))/np.sum(weight*slope**2))
    offset = new+delta*np.exp(-station.height.to_numpy()/3000)[None]
    dates = pd.to_datetime(c['hours'], unit='h')
    masks = {name: np.broadcast_to(np.isin(dates.month, months)[:, None], c['valid'].shape) for name, months in
        [('DJF',[12,1,2]), ('MAM',[3,4,5]), ('JJA',[6,7,8]), ('SON',[9,10,11])]}
    masks.update({name:np.broadcast_to(flag[None], c['valid'].shape) for name, flag in
        [('height_lt500',station.height.to_numpy()<500), ('height_500_1500',(station.height.to_numpy()>=500)&(station.height.to_numpy()<1500)), ('height_ge1500',station.height.to_numpy()>=1500)]})
    strata = {name:{arm:point_metrics(c['truth'], pred, c['valid']&mask) for arm,pred in [('old',old),('new',new)]} for name,mask in masks.items()}
    result = dict(current=metrics, old=point_metrics(c['truth'],old,c['valid']), paired_intervals=intervals,
        B_training_only_common_height_offset=dict(delta_Z0_mm=delta, C_metrics=point_metrics(c['truth'],offset,c['valid']),
            interpretation='One scalar fitted only on qualified B training labels; fixed Z0-grid offset, not C-fitted'),
        strata=strata, D2020=False, selection_unchanged=True)
    write_json(ROOT/'point_diagnostics.json', result)
    np.savez_compressed(ROOT/'C_locked_points.npz', hours=c['hours'], old=old, new=new, truth=c['truth'], population=c['valid'])
    print('C_PAIRED_INTERVALS',intervals,flush=True)
    print('B_ONLY_OFFSET',delta,result['B_training_only_common_height_offset']['C_metrics']['station_macro_rmse_mm'],flush=True)


if __name__ == '__main__':
    main()
