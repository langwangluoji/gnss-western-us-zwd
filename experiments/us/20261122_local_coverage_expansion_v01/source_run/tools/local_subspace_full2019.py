"""Frozen-checkpoint annual development audit, without recomputing M0 grids."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
import numpy as np
import torch
from torch.nn import functional as F
from local_subspace_data import ROOT, PilotData, transform_condition
from local_probe_diagnostics import PROJECT, emit, save_npz
from local_conditional_gaussian import wet_feature
from local_probability_experiment import joint_series, joint_intervals
from local_student_field import macro
from gnss_field.data import ContextBuilder
from gnss_field.data.context import cycles
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256
from gnss_field.evaluation import point_metrics, crps_ensemble
from gnss_field.models.subspace import SubspaceEDM
from gnss_field.training.prediction import safe_query_geometry

ANNUAL = ROOT/'full2019'
FROZEN = PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'
IDW = PROJECT/'results/20261119_frozen_m0_audit/monthly'


def conditions(data, hours):
    path = ANNUAL/'A_conditions.npz'
    contract = dict(pca=sha256(ROOT/'condition_pca.npz'), data=data.data.contract, source=sha256(__file__))
    if path.exists():
        meta = read_json(path.with_suffix('.json'))
        assert meta['contract'] == contract and meta['sha256'] == sha256(path)
        return np.load(path)['condition']
    builder = ContextBuilder(data.data.development)
    rows = []
    started = time.monotonic()
    for start in range(0, len(hours), 8):
        selected = hours[start:start+8]
        context = builder.build_context(selected)
        features = torch.cat([context.stations[:, :, :10].flatten(1),
            F.adaptive_avg_pool2d(context.coarse.flatten(1, 2), (4, 4)).flatten(1),
            torch.tensor(cycles(selected))], -1).numpy()
        rows.append(transform_condition(features, data.pca))
        if start % 720 == 0:
            emit(f'Annual A-only condition cache {start}/{len(hours)}; {time.monotonic()-started:.1f}s')
    result = np.concatenate(rows)
    subset = data.arrays(2019)
    error = abs(result[np.searchsorted(hours, subset['hours'])]-subset['condition'].cpu().numpy()).max()
    assert error <= 1e-4, error
    save_npz(path, hours=hours, condition=result)
    write_json(path.with_suffix('.json'), dict(contract=contract, sha256=sha256(path), maximum_subset_PCA_error=float(error)))
    return result


def main():
    torch.set_num_threads(2)
    ANNUAL.mkdir(parents=True, exist_ok=True)
    previous = sha256(ROOT/'condition_pca.npz')
    data = PilotData()
    assert sha256(ROOT/'condition_pca.npz') == previous
    selection = read_json(ROOT/'selection.lock.json')
    files = sorted(FROZEN.glob('forecast_2019-*.npz'))
    assert len(files) == 12
    immutable_json(ANNUAL/'protocol.lock.json', dict(source=sha256(__file__), selection=sha256(ROOT/'selection.lock.json'),
        checkpoint=sha256(selection['checkpoint']), foundation=data.contract,
        input_sha256={p.name: sha256(p) for p in files}, members=128, seed=20261119,
        purpose='Post-selection annual attribution audit, not automatic promotion or new C selection',
        gaussian='same subspace decomposition and keyed streams; initial Gaussian is exact', D2020='sealed'))
    c = data.arrays(2019)
    records = [dict(np.load(p)) for p in files]
    hours = np.concatenate([r['hours'] for r in records])
    assert len(hours) == 8760
    feature = conditions(data, hours)
    truth = np.concatenate([r['truth'] for r in records])
    m0 = np.concatenate([r['prediction'] for r in records])
    idw = np.concatenate([np.load(IDW/f'{p.stem[-7:]}.npz')['idw'] for p in files])
    valid = np.concatenate([r['normal_support_S0'] & r['population_F'] for r in records])
    full = np.concatenate([r['population_F'] for r in records])
    assert valid.sum() == 642258 and valid.any(0).sum() == 76
    wet, present = wet_feature(data.data.development.context_values(hours))
    a = data.amplitude
    x = np.where(present, (wet-a['A_wetness_center'])/a['A_wetness_std'], 0.)
    arrays = dict(c, hours=hours, condition=torch.tensor(feature, device=data.device),
        mean=torch.tensor(m0+data.weight*(idw-m0), device=data.device),
        amplitude=torch.tensor(np.exp(a['a']+a['b']*x), dtype=torch.float32, device=data.device))
    model = SubspaceEDM().to(data.device).eval()
    model.load_state_dict(torch.load(selection['checkpoint'], map_location=data.device, weights_only=False)['ema'])
    metadata = c['store'].stations.iloc[c['station_ids']].reset_index(drop=True)
    series = {k: [] for k in ('EDM', 'Gaussian')}
    means, crps = {k: [] for k in series}, {k: [] for k in series}
    intervals = {k: {level: [[], []] for level in (.8, .9, .95)} for k in series}
    elapsed = time.monotonic()
    start = 0
    for month, record in enumerate(records):
        stop = start+len(record['hours'])
        path = ANNUAL/files[month].name
        if path.exists():
            saved = dict(np.load(path))
            samples, reference = saved['samples'], saved['reference']
        else:
            samples, reference = [], []
            with torch.no_grad():
                for row in range(start, stop, 8):
                    idx = np.arange(row, min(row+8, stop))
                    samples.append(data.samples(model, arrays, idx, 128, seed=20261119).cpu().numpy())
                    reference.append(data.samples(model, arrays, idx, 128, seed=20261119, gaussian=True).cpu().numpy())
            samples, reference = np.concatenate(samples), np.concatenate(reference)
            save_npz(path, hours=record['hours'], station_ids=c['station_ids'], samples=samples, reference=reference,
                     analytic_mean=arrays['mean'][start:stop].cpu().numpy(), population=valid[start:stop])
        for name, members in [('EDM', samples), ('Gaussian', reference)]:
            pop = valid[start:stop]
            if not np.isfinite(members.transpose(0, 2, 1)[full[start:stop]]).all():
                raise ValueError('Nonfinite full forecast; no sample may be dropped')
            scores, names = joint_series(truth[start:stop], members, pop, metadata)
            series[name].append(scores)
            means[name].append(members.mean(1))
            values = np.zeros(pop.shape)
            values[pop] = crps_ensemble(truth[start:stop][pop], members.transpose(0, 2, 1)[pop])
            crps[name].append(values)
            for level in (.8, .9, .95):
                low, high = np.quantile(members, [(1-level)/2, (1+level)/2], axis=1)
                intervals[name][level][0].append((truth[start:stop] >= low) & (truth[start:stop] <= high))
                intervals[name][level][1].append(high-low)
        emit(f'Annual paired EDM {files[month].stem[-7:]} complete; sampling/scoring elapsed {time.monotonic()-elapsed:.1f}s')
        start = stop
    series = {k: np.concatenate(v) for k, v in series.items()}
    means = {k: np.concatenate(v) for k, v in means.items()}
    active = np.isfinite(series['EDM'][..., 0]).any(0)
    series = {k: v[:, active] for k, v in series.items()}
    metrics = {}
    for name in series:
        metrics[name] = dict(point=point_metrics(truth, means[name], valid),
            ES_mm=float(np.nanmean(np.nanmean(series[name][..., 0], 0))),
            VS_mm=float(np.nanmean(np.nanmean(series[name][..., 1], 0))),
            station_macro_CRPS_mm=macro(np.concatenate(crps[name]), valid),
            intervals={str(round(level*100)): dict(coverage=macro(np.concatenate(cov), valid), width_mm=macro(np.concatenate(width), valid))
                       for level, (cov, width) in intervals[name].items()})
    difference = joint_intervals(hours, series['EDM'], series['Gaussian'])
    drift = means['EDM']-means['Gaussian']
    save_npz(ANNUAL/'joint_scores.npz', hours=hours, **series)
    save_npz(ANNUAL/'annual_means.npz', hours=hours, truth=truth, population=valid, analytic_mean=arrays['mean'].cpu().numpy(), **means)
    write_json(ANNUAL/'results.json', dict(metrics=metrics, joint_intervals=difference, selection=selection,
        analytic_point=point_metrics(truth, arrays['mean'].cpu().numpy(), valid), n_stations=76, n_station_hours=642258,
        paired_mean_drift_RMS_mm=float(np.sqrt(np.mean(drift[valid]**2))), n_groups=int(active.sum()),
        note='EDM changes mean and shape; raw ES gain is not attributed to nonGaussian dependence', D2020='sealed'))
    emit(f'Annual complete: {metrics}; differences {difference}')


if __name__ == '__main__':
    main()
