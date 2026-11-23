"""Exploratory B-only representation diagnostics; never a deployable forecast."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256
from gnss_field.data import PackedStore
from gnss_field.data.manifests import spatial_folds
from gnss_field.training.observable import load_bundle
from gnss_field.evaluation import point_metrics

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261120_architecture_research_v01'
PILOT = PROJECT/'results/20261119_local_edm_subspace_v01'


def oracle_prediction(design, truth, mean, valid, fitted, amplitude, tau):
    """Contemporaneous fitted-B observations are allowed ONLY in this diagnostic."""
    prediction = np.array(mean, dtype='f8', copy=True)
    patterns, assignment = np.unique(valid[:, fitted], axis=0, return_inverse=True)
    for j, pattern in enumerate(patterns):
        if not pattern.any():
            continue
        hours = np.flatnonzero(assignment == j)
        used = np.flatnonzero(fitted)[pattern]
        h = np.asarray(design, dtype='f8')[used]
        residual = (truth[np.ix_(hours, used)]-mean[np.ix_(hours, used)])/amplitude[hours, None]
        if not np.isfinite(residual).all():
            raise ValueError('Nonfinite fitted-B residual')
        operator = cho_solve(cho_factor(h.T@h+tau*tau*np.eye(h.shape[1]), lower=True), h.T)
        prediction[hours] += amplitude[hours, None]*(residual@operator.T)@design.T
    return prediction


def main():
    ROOT.mkdir(exist_ok=True)
    a, meta = load_bundle(PILOT/'server_bundle')
    store = PackedStore(PROJECT/'../../02_data/processed/gnss_field_server_v1', purpose='train')
    assignment = spatial_folds(store, 20261117)
    np.testing.assert_array_equal(a['station_ids'], store.ids('B'))
    immutable_json(ROOT/'diagnostic.lock.json', dict(source_sha256=sha256(__file__),
        B_manifest_sha256=sha256(PILOT/'server_bundle/manifest.json'),
        protocol='B-only fixed three physical-group folds, fixed 288 development anchors; diagnostic uses fitted-B current labels',
        rank_diagnostic=[32, 64, 128], label_use='Oracle diagnostic, not A-only performance or an attainable error bound',
        no_C_selection=True, D2020=False, new_server_jobs=False))
    pca = dict(np.load(PILOT/'condition_pca.npz'))
    energy = np.cumsum(pca['singular_values']**2)/np.sum(pca['singular_values']**2)
    spatial = dict(np.load(PILOT/'server_bundle/spatial.npz'))
    right = np.load(PROJECT/'results/20261119_local_method_research_v02/observable_subspace.npz')['right_vectors']
    np.testing.assert_allclose(right[:64], spatial['v'], atol=1e-7)
    h = a['projection'].astype('f8')@spatial['v'].astype('f8')+a['remainder'].astype('f8')
    np.testing.assert_allclose(h@right[:64].T, a['projection'], atol=2e-5)
    ix = a['development_indices']
    truth, mean, valid, g = [a[name][ix] for name in ('truth', 'mean', 'valid', 'amplitude')]
    results = {}
    for rank in (32, 64, 128):
        matrix = h@right[:rank].T
        combined = np.array(mean, dtype='f8')
        folds = []
        for fold in range(3):
            fitted = assignment != fold
            prediction = oracle_prediction(matrix, truth, mean, valid, fitted, g, meta['tau_surface_mm'])
            population = valid & ~fitted[None]
            combined[:, ~fitted] = prediction[:, ~fitted]
            folds.append(dict(fold=fold, oracle=point_metrics(truth, prediction, population),
                baseline=point_metrics(truth, mean, population)))
        results[str(rank)] = dict(folds=folds, pooled_held_group=point_metrics(truth, combined, valid))
        print('B_ASSISTED_ORACLE', rank, results[str(rank)]['pooled_held_group']['station_macro_rmse_mm'], flush=True)
    result = dict(PCA_input_variance={str(rank):float(energy[rank-1]) for rank in (16, 32, 64, 128)},
        B_baseline=point_metrics(truth, mean, valid), oracle=results,
        interpretation='Oracle consumes current fitted-B labels; tests representation/transfer only, not deployable A-only skill',
        B_hours=len(ix), D2020=False)
    write_json(ROOT/'bottleneck_diagnostics.json', result)
    print('DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
