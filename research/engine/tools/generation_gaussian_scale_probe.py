"""CPU-only scalar controls fit exclusively to allowed B training observations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from scipy.stats import norm
import torch

from grid_generation_metrics import score_ensemble
from gnss_field.models.diffusion import keyed_noise
from gnss_field.artifacts import sha256


def main():
    root = Path(__file__).resolve().parents[1]/'results/20261122_generation_postrun_research_v01'
    source = root/'downloaded'
    bundle = Path(__file__).resolve().parents[1]/'results/20261119_local_edm_subspace_v01/server_bundle'
    manifest = json.loads((source/'metadata/bundle_manifest.json').read_text(encoding='utf-8'))
    assert sha256(bundle/'data.npz') == manifest['data_sha256']
    a = dict(np.load(bundle/'data.npz'))
    contract = json.loads((source/'data.lock.json').read_text(encoding='utf-8'))
    design = json.loads((source/'metadata/design.json').read_text(encoding='utf-8'))
    stations = pd.read_csv(source/'metadata/stations.csv')
    assignment = np.asarray(design['B_fold_assignment'])
    ix = np.searchsorted(a['hours'], contract['train_hours'])
    assert len(ix) == 5472 and np.array_equal(a['hours'][ix], contract['train_hours'])
    spatial = np.sum(a['projection'].astype('f8')**2, axis=1) + np.sum(a['remainder'].astype('f8')**2, axis=1)
    tau = float(manifest['tau_surface_mm'])
    output = dict(exploratory=True, target_fitting='training B only, no outer labels', folds={})
    torch.set_num_threads(2)
    for fold in (0, 1):
        original = dict(np.load(source/f'fold_{fold}/gaussian_outer.npz'))
        oi = np.searchsorted(a['hours'], original['hours'])
        assert np.max(abs(a['mean'][oi]-original['mean'])) < .001
        assert np.array_equal(original['population'], a['valid'][oi] & (assignment == fold)[None])
        train_mask = a['valid'][ix] & (assignment != fold)[None]
        counts = train_mask.sum(0); use = counts > 0
        training_error = np.where(train_mask[:, use], a['truth'][ix][:, use]-a['mean'][ix][:, use], 0.).astype('f8')
        amp = a['amplitude'][ix].astype('f8')[:, None]
        weights = train_mask[:, use] / counts[use][None] / use.sum()
        xy = stations.iloc[original['station_ids']][['x', 'y']].to_numpy()
        obs_noise = a['amplitude'][oi, None, None]*tau*keyed_noise(original['hours'], 32, 20263124,
            shape=(905,))[:, :, original['station_ids']].numpy()
        rows = {}
        for kind in ('field_only', 'total_error'):
            def sigma(alpha, s=spatial[use], g=amp):
                return g*np.sqrt(alpha**2*s+tau**2) if kind == 'field_only' else alpha*g*np.sqrt(s+tau**2)

            def loss(alpha):
                std = sigma(alpha); z = training_error/std
                crps = std*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
                return float((crps*weights).sum())

            fit = minimize_scalar(loss, bounds=(.1, 1.8), method='bounded', options={'xatol': 1e-4})
            assert fit.success and np.isfinite(fit.fun)
            alpha = float(fit.x)
            residual = original['samples']-original['mean'][:, None]
            samples = (original['mean'][:, None]+alpha*(residual-obs_noise)+obs_noise
                       if kind == 'field_only' else original['mean'][:, None]+alpha*residual)
            metrics = score_ensemble(original['truth'], samples, original['mean'], original['population'], xy)
            mask = original['population']; count = mask.sum(0); keep = count > 0
            error = np.where(mask, original['truth']-original['mean'], 0.)
            sd = sigma(alpha, s=spatial[None], g=a['amplitude'][oi, None])
            z = error/sd
            analytic = sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
            macro = lambda x: float((np.where(mask, x, 0.).sum(0)[keep]/count[keep]).mean())
            rows[kind] = dict(alpha=alpha, training_CRPS=fit.fun, baseline_training_CRPS=loss(1.),
                fitted_station_hours=int(train_mask.sum()), train_stations=int(use.sum()), metrics=metrics,
                analytic_gaussian_CRPS=macro(analytic), analytic_coverage90=macro(abs(error)<=norm.ppf(.95)*sd),
                analytic_width90=macro(2*norm.ppf(.95)*sd))
            print('SCALAR_CONTROL', fold, kind, alpha, metrics['group_ES_mm'], metrics['CRPS_mm'], flush=True)
        output['folds'][str(fold)] = rows
        (root/'gaussian_scale_control.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
