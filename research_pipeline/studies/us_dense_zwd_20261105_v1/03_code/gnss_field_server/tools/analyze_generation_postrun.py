"""Read-only outer-pilot diagnostics from the verified downloaded generation run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generation_postrun_diagnostics import matched_decomposition, rank_histogram, fair_crps
from grid_generation_metrics import score_ensemble
from gnss_field.evaluation import point_metrics


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def analyze(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    stations = pd.read_csv(source / 'metadata/stations.csv')
    output = dict(cohort='B2018 outer96 pilot; exposed incremental development',
                  C2019=False, D2020=False, folds={})
    for fold in (0, 1):
        folder = source / f'fold_{fold}'
        gaussian = load_arrays(folder / 'gaussian_outer.npz')
        ids = gaussian['station_ids']
        xy = stations.iloc[ids][['x', 'y']].to_numpy()
        truth, mask = gaussian['truth'], gaussian['population']

        def score(samples, mean):
            result = score_ensemble(truth, samples, mean, mask, xy)
            fair = fair_crps(np.where(mask, truth, 0.), samples)
            counts = mask.sum(0); use = counts > 0
            result['supplemental_fair_CRPS_mm'] = float((np.where(mask, fair, 0.).sum(0)[use]/counts[use]).mean())
            return result

        baseline = score(gaussian['samples'], gaussian['mean'])
        archived = json.loads((folder / 'gaussian_outer.json').read_text(encoding='utf-8'))
        for key in ('station_macro_rmse_mm', 'CRPS_mm', 'group_ES_mm', 'group_VS'):
            if abs(baseline[key] - archived[key]) > 1e-6:
                raise ValueError(f'Baseline metric failed reproduction: {key}')
        rows = dict(gaussian=baseline,
                    gaussian_ranks=rank_histogram(truth, gaussian['samples'], mask), arms={})
        for arm in ('stochastic', 'diffusion', 'gan'):
            products = {}
            for product in ('point', 'joint'):
                prefix = folder / arm / product
                generated = load_arrays(prefix.with_name(f'{product}_outer_predictions.npz'))
                decomposition = matched_decomposition(gaussian, generated)
                full = score(generated['samples'], generated['mean'])
                archived = json.loads(prefix.with_name(f'{product}_outer.json').read_text(encoding='utf-8'))['metrics']
                for key in ('station_macro_rmse_mm', 'CRPS_mm', 'group_ES_mm', 'group_VS'):
                    if abs(full[key] - archived[key]) > 1e-6:
                        raise ValueError(f'Generated metric failed reproduction: {arm}/{product}/{key}')
                shifted = score(decomposition['gaussian_shifted'], generated['mean'])
                recentered = score(decomposition['generated_recentered'], gaussian['mean'])
                prefixes = {}
                for k in (8, 16, 24, 32):
                    if k > generated['samples'].shape[1]:
                        continue
                    mean = gaussian['mean'] + decomposition['correction'][:, :k].mean(1)
                    prefixes[str(k)] = point_metrics(truth, mean, mask)
                shift = decomposition['mean_shift']
                products[product] = dict(original=full, gaussian_at_generated_mean=shifted,
                    generated_at_baseline_mean=recentered,
                    mean_shift_rms_mm=float(np.sqrt(np.mean(shift[mask]**2))),
                    correction_mean_prefixes=prefixes,
                    ranks=rank_histogram(truth, generated['samples'], mask),
                    interpretation='Same finite-member estimated translation; residual score differences can reflect covariance or distribution shape, not proof of non-Gaussian benefit')
            rows['arms'][arm] = products
        output['folds'][str(fold)] = rows
    path = destination / 'outer_diagnostics.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)

