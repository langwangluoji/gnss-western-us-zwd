"""Audit only the existing training410 pack; never load validation labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def station_correlations(x, lag):
    left, right = x[:-lag], x[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(0)
    n = np.maximum(count, 1)
    a, b = np.where(valid, left, 0), np.where(valid, right, 0)
    cross = (a * b).sum(0) - a.sum(0) * b.sum(0) / n
    va = (a * a).sum(0) - a.sum(0) ** 2 / n
    vb = (b * b).sum(0) - b.sum(0) ** 2 / n
    good = (count >= 100) & (va > 0) & (vb > 0)
    return float(np.median(cross[good] / np.sqrt(va[good] * vb[good])))


def audit(path):
    with np.load(path, allow_pickle=False) as archive:
        hours = archive['train_hours']
        train = archive['train']
        # Read no val_y, confirmation or C/D arrays.
    unique, first, inverse = np.unique(hours, return_index=True, return_inverse=True)
    values = train.reshape(-1, *train.shape[2:])[first]
    assert np.array_equal(train.reshape(-1, *train.shape[2:]),
                          values[inverse.reshape(-1)], equal_nan=True), 'Conflicting duplicated hours'
    assert np.all(np.diff(unique) == 1), 'This audit expects a continuous hourly axis'
    start = int(np.datetime64('2016-01-01', 'h').astype('int64'))
    outside = int((unique < start).sum())
    values = values[unique >= start]
    unique = unique[unique >= start]
    zwd = values[..., 0].astype('float64')
    increments = np.diff(zwd, axis=0)
    finite = np.isfinite(zwd)
    cumulative = np.vstack([np.zeros((1, finite.shape[1])), finite.cumsum(0)])
    coverage = []
    for length in (12, 24, 48, 72, 168):
        # Six-hour targets must exist in the same axis; availability here is
        # history-only and does not select examples by future label values.
        anchors = np.arange(length - 1, len(unique) - 6)
        counts = cumulative[anchors + 1] - cumulative[anchors + 1 - length]
        coverage.append(dict(history_hours=length, anchors=len(anchors),
            station_windows=int(counts.size),
            completely_observed_fraction=float((counts == length).mean()),
            at_least_80pct_observed_fraction=float((counts >= .8 * length).mean())))
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(status='TRAINING_PACK_DESCRIPTIVE_AUDIT_NOT_MODEL_VALIDATION',
        input_path=path.as_posix(), input_sha256=digest,
        train_shape=list(train.shape), stored_hour_occurrences=int(hours.size),
        unique_hours_before_boundary_filter=int(len(first)),
        duplicate_hour_occurrences=int(hours.size - len(first)),
        duplicated_hour_values_consistent=True,
        hours_before_2016=outside, continuous_axis_gaps=0,
        start=str(np.datetime64(int(unique[0]), 'h')),
        end=str(np.datetime64(int(unique[-1]), 'h')),
        finite_fraction_by_variable=np.isfinite(values).mean((0, 1)).tolist(),
        candidate_history_coverage=coverage,
        lag_correlations=[dict(lag_hours=lag,
            station_median_zwd_corr=station_correlations(zwd, lag),
            station_median_increment_corr=station_correlations(increments, lag))
            for lag in (1, 3, 6, 12, 24, 48, 72, 168)],
        interpretation='Raw ZWD correlation includes seasonality; not OOF residual predictability. '
                       'Overlapping rows deduplicated by first occurrence; old pack unchanged. '
                       'Availability pooled over training stations, not dynamic support layouts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pack)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
