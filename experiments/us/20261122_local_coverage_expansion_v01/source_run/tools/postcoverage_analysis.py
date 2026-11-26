"""Frozen-product diagnostics: B optimization, independent C mean integration and ES attribution."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import read_json, write_json, immutable_json, sha256
from gnss_field.models.subspace import SubspaceEDM, station_members
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.models.diffusion import keyed_noise
from gnss_field.objectives.scores import energy_score
from gnss_field.evaluation import point_metrics, crps_ensemble
from gnss_field.training.observable import load_bundle
from gnss_field.runtime import configure

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261120_postcoverage_analysis_v01'
PILOT = PROJECT/'results/20261119_local_edm_subspace_v01'


def latent_mean_shift(model, condition, hours, projection, amplitude, *, members=128, seed=20261121):
    """A forecast-independent paired latent pool; labels and nuisance noise never enter."""
    noise = keyed_noise(hours, members, seed, (model.rank,), device=condition.device)
    changes = (model.sample_noise(condition, noise)-noise)@projection.T
    changes = changes*amplitude[:, None, None]
    return changes.mean(1), changes[:, :members//2].mean(1), changes.std(1)/members**.5


def load_evaluation(directory):
    directory = Path(directory)
    meta = read_json(directory/'manifest.json')
    if meta['role'] != 'C' or meta['year'] != 2019:
        raise PermissionError('Postcoverage analysis only accepts exposed C2019, never sealed D2020')
    if sha256(directory/'data.npz') != meta['data_sha256'] or sha256(directory/'stations.csv') != meta['stations_sha256']:
        raise ValueError('Evaluation bytes differ from frozen inputs')
    arrays = dict(np.load(directory/'data.npz'))
    assert len(arrays['hours']) == 8760
    assert (pd.to_datetime(arrays['hours'], unit='h').year == 2019).all()
    return arrays, meta, pd.read_csv(directory/'stations.csv')


def tensors(arrays, device):
    return {k: torch.tensor(arrays[k], device=device) for k in
        ('condition', 'mean', 'amplitude', 'projection', 'remainder')}


def noises(arrays, values, indices, members=128, seed=20261119):
    hours = arrays['hours'][indices]
    device = values['mean'].device
    z = keyed_noise(hours, members, seed, (64,), device=device)
    remainder = keyed_noise(hours, members, seed+1000, (arrays['remainder'].shape[1],), device=device)
    error = keyed_noise(hours, members, seed+2000, (905,), device=device)[..., arrays['station_ids']]
    return z, remainder, error


def members_for(model, arrays, values, indices, z, remainder, error, tau):
    latent = z if model is None else model.sample_noise(values['condition'][indices], z)
    return station_members(values['mean'][indices], latent, remainder, error,
        values['projection'], values['remainder'], values['amplitude'][indices], tau)


def models(device, key='ema'):
    expected = read_json(ROOT/'expected.json')['checkpoints']
    output = {}
    for arm, cls in [('mean', SubspaceMeanShift), ('edm', SubspaceEDM)]:
        checkpoint = ROOT/'artifacts'/arm/'selected.pt'
        if sha256(checkpoint) != expected[arm]:
            raise ValueError('Selected source checkpoint differs')
        current = cls().to(device).eval().requires_grad_(False)
        current.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)[key])
        output[arm] = current
    return output


def B_diagnostics(device):
    a, meta = load_bundle(PILOT/'server_bundle')
    t = tensors(a, device)
    truth = torch.tensor(a['truth'], device=device)
    valid = torch.tensor(a['valid'], device=device)
    fitted = {key: models(device, key) for key in ('ema', 'model')}
    result = {}
    with torch.no_grad():
        for arm in ('mean', 'edm'):
            result[arm] = {}
            for version in ('ema', 'model'):
                total = 0.
                indices = a['development_indices']
                for start in range(0, len(indices), 8):
                    selected = indices[start:start+8]
                    noise = noises(a, t, selected, members=16)
                    samples = members_for(fitted[version][arm], a, t, selected, *noise, meta['tau_surface_mm'])
                    total += float(energy_score(samples, truth[selected], valid[selected]))*len(selected)
                result[arm][version+'_ES_mm'] = total/len(indices)
                print('B diagnostic', arm, version, result[arm][version+'_ES_mm'], flush=True)
            result[arm]['EMA_minus_raw_ES_mm'] = result[arm]['ema_ES_mm']-result[arm]['model_ES_mm']
            archived = read_json(ROOT/'artifacts'/arm/'selection.lock.json')['value']
            assert abs(result[arm]['ema_ES_mm']-archived) <= .001
        # Diagnostic only: pooled marginal spread fit uses B training labels, never C.
        mean = (t['mean']+(fitted['ema']['mean'].net(t['condition'])@t['projection'].T)*t['amplitude'][:, None]).cpu().numpy()
    variance = (a['projection']**2).sum(1)+(a['remainder']**2).sum(1)+meta['tau_surface_mm']**2
    std = a['amplitude'][:, None]*variance[None]**.5
    train = a['training_indices']; development = a['development_indices']
    standardized = (a['truth']-mean)/std
    lam = float(np.sqrt(np.mean(standardized[train][a['valid'][train]]**2)))
    result['B_only_marginal_scale_diagnostic'] = dict(fitted_scale=lam,
        training_role='B2018 registered 6288 hours only', used_for_C_product=False,
        development_standardized_RMS=float(np.sqrt(np.mean(standardized[development][a['valid'][development]]**2))))
    write_json(ROOT/'B_diagnostics.json', result)
    return result


def annual(device):
    from local_probability_experiment import joint_series
    a, meta, metadata = load_evaluation(PILOT/'server_evaluation')
    t = tensors(a, device)
    fitted = models(device)
    names = ('Gaussian', 'mean', 'EDM', 'Gaussian_plus_EDM_shift', 'shape_only_EDM')
    series = {name: [] for name in names}
    crps = {name: [] for name in names}
    shifts, half_shifts, errors, observed_means, analytic_means = [], [], [], [], []
    intervals = {arm: {level: [[], []] for level in (.8, .9, .95)} for arm in ('mean', 'EDM')}
    with torch.no_grad():
        for start in range(0, len(a['hours']), 8):
            stop = min(start+8, len(a['hours'])); idx = np.arange(start, stop)
            shift, half, se = latent_mean_shift(fitted['edm'], t['condition'][idx], a['hours'][idx],
                t['projection'], t['amplitude'][idx])
            shift, half, se = (v.cpu().numpy() for v in (shift, half, se))
            shifts.append(shift); half_shifts.append(half); errors.append(se)
            noise = noises(a, t, idx)
            gaussian = members_for(None, a, t, idx, *noise, meta['tau_surface_mm']).cpu().numpy()
            edm = members_for(fitted['edm'], a, t, idx, *noise, meta['tau_surface_mm']).cpu().numpy()
            mean = members_for(fitted['mean'], a, t, idx, *noise, meta['tau_surface_mm']).cpu().numpy()
            analytic = t['mean'][idx]+(fitted['mean'].net(t['condition'][idx])@t['projection'].T)*t['amplitude'][idx, None]
            analytic_means.append(analytic.cpu().numpy()); observed_means.append(edm.mean(1))
            pop, truth = a['valid'][idx], a['truth'][idx]
            products = dict(Gaussian=gaussian, mean=mean, EDM=edm,
                Gaussian_plus_EDM_shift=gaussian+shift[:, None], shape_only_EDM=edm-shift[:, None])
            for name, samples in products.items():
                if not np.isfinite(samples.transpose(0, 2, 1)[a['population_F'][idx]]).all():
                    raise ValueError('Nonfinite forecasts cannot be removed')
                values, group_names = joint_series(truth, samples, pop, metadata)
                series[name].append(values)
                values = np.zeros(pop.shape)
                values[pop] = crps_ensemble(truth[pop], samples.transpose(0, 2, 1)[pop])
                crps[name].append(values)
                if name in intervals:
                    for level in intervals[name]:
                        low, high = np.quantile(samples, [(1-level)/2, (1+level)/2], axis=1)
                        intervals[name][level][0].append((truth >= low)&(truth <= high))
                        intervals[name][level][1].append(high-low)
            if start % 720 == 0:
                print('C2019 independent attribution', start, '/', len(a['hours']), flush=True)
    shift, half, se, observed, mean_point = (np.concatenate(v) for v in
        (shifts, half_shifts, errors, observed_means, analytic_means))
    series = {name: np.concatenate(values) for name, values in series.items()}
    active = np.isfinite(series['Gaussian'][..., 0]).any(0)
    series = {name: values[:, active] for name, values in series.items()}
    pop = a['valid']; count = pop.sum(0); live = count > 0
    def macro(value):
        return float((np.where(pop, value, 0).sum(0)[live]/count[live]).mean())
    results = {name: dict(ES_mm=float(np.nanmean(values[..., 0], axis=0).mean()),
        VS_mm=float(np.nanmean(values[..., 1], axis=0).mean()),
        station_macro_CRPS_mm=macro(np.concatenate(crps[name]))) for name, values in series.items()}
    for arm in ('mean', 'edm'):
        name = 'EDM' if arm == 'edm' else arm
        archived = read_json(ROOT/'artifacts'/arm/'C2019_metrics.json')
        for current, saved in [('ES_mm','group_macro_ES_mm'),('VS_mm','group_macro_VS_mm'),('station_macro_CRPS_mm','station_macro_CRPS_mm')]:
            assert abs(results[name][current]-archived[saved]) <= .001, (arm,current)
    audit_path = PILOT/'review/review_edm.py'
    spec = importlib.util.spec_from_file_location('postcoverage_crossed_audit', audit_path)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    comparisons = [('EDM','mean'), ('EDM','Gaussian_plus_EDM_shift'), ('shape_only_EDM','Gaussian'), ('mean','Gaussian')]
    paired = {first+'_minus_'+second: audit.crossed_intervals(a['hours'], series[first], series[second]) for first, second in comparisons}
    result = dict(probability=results,paired=paired,
        mean_analytic_point=point_metrics(a['truth'],mean_point,pop),
        EDM_integrated_mean_point=point_metrics(a['truth'],a['mean']+shift,pop),
        EDM_half_pool_mean_point=point_metrics(a['truth'],a['mean']+half,pop),
        EDM_observation_ensemble_mean_point=point_metrics(a['truth'],observed,pop),
        mean_pool_SE_RMS_mm=float(np.sqrt(np.mean(se[pop]**2))),
        mean_pool_SE_max_mm=float(se[pop].max()),
        intervals={arm: {str(round(level*100)):dict(coverage=macro(np.concatenate(values[0])),width_mm=macro(np.concatenate(values[1])))
            for level,values in levels.items()} for arm,levels in intervals.items()},
        evaluation_population=dict(stations=int(live.sum()),station_hours=int(pop.sum()),groups=int(active.sum())),
        mean_pool_seed=20261121,mean_pool_members=128,forecast_seed=20261119,forecast_members=128,
        D2020=False,no_new_training=True,no_C_reselection=True)
    write_json(ROOT/'annual_results.json',result)
    np.savez_compressed(ROOT/'annual_scores.npz',hours=a['hours'],**series)
    np.savez_compressed(ROOT/'integrated_means.npz',hours=a['hours'],shift=shift,half_shift=half,SE=se,mean_point=mean_point)
    print('ANNUAL_RESULT',result,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B-only',action='store_true')
    args = parser.parse_args()
    configure(type('Runtime',(),{'cpus':2})(),42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    immutable_json(ROOT/'analysis.lock.json',dict(source=sha256(__file__),
        mean_checkpoint=sha256(ROOT/'artifacts/mean/selected.pt'),edm_checkpoint=sha256(ROOT/'artifacts/edm/selected.pt'),
        B_manifest=sha256(PILOT/'server_bundle/manifest.json'),C_manifest=sha256(PILOT/'server_evaluation/manifest.json'),
        mean_pool_seed=20261121,mean_pool_members=128,evaluation_members=128,no_new_training=True,D2020=False))
    B_diagnostics(device)
    if not args.B_only:
        annual(device)


if __name__ == '__main__':
    main()
