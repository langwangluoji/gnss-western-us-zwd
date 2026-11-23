"""Scientific invariants for the local A-conditioned probe."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_zero_probe_keeps_frozen_mean_without_coefficient_center_offset():
    from local_condition_probe import LinearCoefficientProbe
    head = LinearCoefficientProbe(7, 3)
    features = torch.randn(4, 7)
    base = torch.randn(4, 5)*20+60
    matrix = torch.randn(5, 3)
    torch.testing.assert_close(head.station_prediction(features, base, matrix, 35.), base,
                               rtol=0, atol=0)


def test_probe_loss_handles_missing_targets_and_uses_frozen_spatial_matrix():
    from local_condition_probe import LinearCoefficientProbe, observation_objective
    head = LinearCoefficientProbe(7, 3)
    matrix = torch.randn(5, 3)
    features = torch.randn(4, 7)
    base = torch.ones(4, 5)*60
    target = torch.ones(4, 5)*63
    valid = torch.ones(4, 5, dtype=torch.bool)
    valid[:, -1] = False
    target[:, -1] = float('nan')
    curvature_gram = torch.eye(3).expand(4, -1, -1)
    result = observation_objective(head, features, base, target, valid, matrix,
                                   curvature_gram, 35.)
    result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert matrix.grad is None
    assert any(p.grad.abs().sum() > 0 for p in head.parameters())


def test_probe_split_retains_entire_nominal_calendar_blocks():
    from local_probe_diagnostics import split_head_hours
    hours = np.arange(0, 168*16, 7, dtype='i8')+420000
    train, development = split_head_hours(hours)
    assert len(train) and len(development)
    assert not set((train-hours[0])//168) & set((development-hours[0])//168)
    np.testing.assert_array_equal(np.sort(np.r_[train, development]), hours)
