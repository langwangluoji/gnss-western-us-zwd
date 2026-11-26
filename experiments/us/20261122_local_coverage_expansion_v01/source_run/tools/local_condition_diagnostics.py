"""Post-fit calibration diagnostics; no fitting to development observations."""
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import norm
from local_probe_diagnostics import PROJECT, ProbeData
from local_joint_field import local_bases, observation_factors
from gnss_field.artifacts import read_json, write_json, sha256
from gnss_field.evaluation import point_metrics


def macro(value, population):
    count = population.sum(0)
    active = count > 0
    return float((np.where(population, value, 0).sum(0)[active]/count[active]).mean())


def main():
    torch.set_num_threads(2)
    root = PROJECT/'results/20261119_local_method_research_v02'
    fused = PROJECT/'results/20261119_local_fused_probability_v02'
    data = ProbeData(device='cpu')
    b = data.arrays(2018)
    raw = data.store.context_values(b['hours'])
    values = np.nanmedian(np.where(np.isfinite(raw).all(-1), raw[..., 0], np.nan), 1)
    threshold = np.quantile(values, [.25, .75])
    yearly = dict(np.load(root/'full2019_blend/forecast_2019.npz'))
    ids, hours = yearly['station_ids'], yearly['hours']
    truth, mean, population = yearly['truth'], yearly['blend'], yearly['population_S0']
    raw = data.development.context_values(hours)
    valid_a = np.isfinite(raw).all(-1)
    wet = np.nanmedian(np.where(valid_a, raw[..., 0], np.nan), 1)
    assert np.isfinite(wet[population.any(1)]).all()
    parameters = read_json(fused/'B_only_parameters.lock.json')['model']['std']
    factors = observation_factors(data.development, ids, local_bases(data.store))
    sigma = np.sqrt(sum(s*s*np.sum(h*h, axis=1) for s, h in zip(parameters, factors))+parameters[-1]**2)
    residual = np.where(population, truth-mean, 0.)
    z = residual/sigma
    crps = sigma*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    month = pd.to_datetime(hours, unit='h').month
    distance = cdist(data.development.xy[ids], data.development.xy[data.development.ids('A')])/1000.
    nearest = []
    for start in range(0, len(hours), 128):
        nearest.append(np.where(valid_a[start:start+128, None], distance[None], np.inf).min(-1))
    nearest = np.concatenate(nearest)
    height = data.development.stations.height.to_numpy()[ids]
    masks = {name: np.broadcast_to(np.isin(month, values)[:, None], population.shape) for name, values in
             [('DJF', [12, 1, 2]), ('MAM', [3, 4, 5]), ('JJA', [6, 7, 8]), ('SON', [9, 10, 11])]}
    masks.update(A_Z0_low=np.broadcast_to((wet < threshold[0])[:, None], population.shape),
                 A_Z0_middle=np.broadcast_to(((wet >= threshold[0]) & (wet <= threshold[1]))[:, None], population.shape),
                 A_Z0_high=np.broadcast_to((wet > threshold[1])[:, None], population.shape),
                 nearest_A_0_50=nearest < 50, nearest_A_50_75=(nearest >= 50) & (nearest < 75), nearest_A_75_100=nearest >= 75,
                 height_0_500=np.broadcast_to(height < 500, population.shape),
                 height_500_1500=np.broadcast_to((height >= 500) & (height < 1500), population.shape),
                 height_above1500=np.broadcast_to(height >= 1500, population.shape))
    results = {}
    for name, mask in masks.items():
        pop = population & mask
        metrics = point_metrics(truth, mean, pop)
        if not pop.any():
            results[name] = metrics
            continue
        metrics['crps_mm'] = macro(crps, pop)
        metrics['intervals'] = {str(round(level*100)): dict(coverage=macro(abs(residual) <= norm.ppf((1+level)/2)*sigma, pop),
                    width_mm=macro(np.broadcast_to(2*norm.ppf((1+level)/2)*sigma, pop.shape), pop)) for level in (.8, .9, .95)}
        results[name] = metrics
    write_json(root/'calibration_conditions.json', dict(source=sha256(__file__), parameters=sha256(fused/'B_only_parameters.lock.json'),
        reference_mean=sha256(root/'full2019_blend/forecast_2019.npz'), wetness_thresholds_B2018_Z0_mm=threshold,
        interpretation='Post-fit descriptive strata; current A-station median common-height Z0 defines wetness; no C fitting',
        results=results))
    print({k: dict(n=v['n_station_hours'], rmse=v['station_macro_rmse_mm'],
                  coverage90=v.get('intervals', {}).get('90', {}).get('coverage')) for k, v in results.items()}, flush=True)


if __name__ == '__main__':
    main()
