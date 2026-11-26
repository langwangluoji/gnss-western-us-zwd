"""Exploratory held-member centering control, without new training or label fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_generation_postrun import load_arrays
from generation_postrun_diagnostics import independent_centering
from grid_generation_metrics import score_ensemble


def run(source, output):
    source, output = Path(source), Path(output)
    torch.set_num_threads(2)
    stations = pd.read_csv(source/'metadata/stations.csv')
    records = dict(protocol='first16 pilot -> last16 scored, then reverse; conditional on independent pilot',
                   target_fitting=False, C2019=False, D2020=False, folds={})
    for fold in (0, 1):
        folder = source/f'fold_{fold}'
        gaussian = load_arrays(folder/'gaussian_outer.npz')
        xy = stations.iloc[gaussian['station_ids']][['x', 'y']].to_numpy()

        def score(samples, mean):
            return score_ensemble(gaussian['truth'], samples, mean, gaussian['population'], xy)

        rows = {}
        for arm in ('stochastic', 'diffusion', 'gan'):
            generated = load_arrays(folder/arm/'joint_outer_predictions.npz')
            directions = []
            for pilot, evaluation in ((np.arange(16), np.arange(16, 32)),
                                      (np.arange(16, 32), np.arange(16))):
                value = independent_centering(gaussian, generated, pilot, evaluation)
                directions.append(dict(
                    pilot_members=pilot.tolist(), evaluation_members=evaluation.tolist(),
                    baseline=score(value['baseline_samples'], gaussian['mean']),
                    independently_centered=score(value['samples'], value['mean']),
                    pilot_scaled_gaussian=score(value['scaled_gaussian'], gaussian['mean']),
                    scale_mean=float(value['scale'].mean()), scale_min=float(value['scale'].min()),
                    scale_max=float(value['scale'].max())))
            rows[arm] = directions
        records['folds'][str(fold)] = rows
    output.write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)

