"""Numerical contract for batched observation-critic pair pooling."""
import copy
import itertools

import pytest
import torch

from gnss_field.generative.architecture import ObservationCritic


def reference_forward(model, values, local_condition, geometry, mask):
    """Pre-vectorization implementation, including the original pair cap/order."""
    clean = torch.where(mask, values, 0.)
    local = torch.where(mask[..., None], local_condition, 0.)
    xyz = geometry / geometry.new_tensor([150000., 150000., 3000.])
    point = model.point(torch.cat((local, clean[..., None], xyz[None].expand(len(values), -1, -1)), -1))
    counts = mask.sum(1, keepdim=True).clamp_min(1)
    mean = torch.where(mask[..., None], point, 0.).sum(1) / counts
    variance = torch.where(mask[..., None], (point - mean[:, None]).square(), 0.).sum(1) / counts
    rows = []
    for row in range(len(values)):
        pooled = []
        for left, right in model._pairs(geometry):
            valid = mask[row, left] & mask[row, right]
            left, right = left[valid][:256], right[valid][:256]
            if not len(left):
                pooled.append(values.new_zeros(64))
                continue
            difference = clean[row, left] - clean[row, right]
            distance = torch.linalg.vector_norm(geometry[left, :2] - geometry[right, :2], dim=-1) / 150000.
            height = (geometry[left, 2] - geometry[right, 2]) / 3000.
            features = torch.cat((local[row, left] - local[row, right], difference[:, None], difference.abs()[:, None], distance[:, None], height[:, None]), -1)
            pooled.append(model.pair(features).mean(0))
        rows.append(torch.cat(pooled))
    return model.output(torch.cat((mean, variance, torch.stack(rows)), -1)).squeeze(-1)


def example(dtype=torch.float64, stations=80):
    torch.manual_seed(871)
    geometry = torch.rand(stations, 3, dtype=dtype) * torch.tensor([150000., 150000., 3000.], dtype=dtype)
    values = torch.randn(5, stations, dtype=dtype)
    local = torch.randn(5, stations, 32, dtype=dtype)
    mask = torch.rand(5, stations) > .3
    mask[0] = False
    mask[1] = False
    if stations:
        mask[1, 0] = True
    mask[2] = True
    return values, local, geometry, mask


def test_pair_network_runs_once_for_heterogeneous_batch():
    model = ObservationCritic().double()
    calls = []
    handle = model.pair.register_forward_hook(lambda *args: calls.append(1))
    model(*example())
    handle.remove()
    assert len(calls) == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scores_first_derivatives_and_gp_second_derivatives_match_reference(dtype):
    values, local, geometry, mask = example(dtype)
    model = ObservationCritic().to(dtype=dtype)
    reference = copy.deepcopy(model)

    def derivatives(network, forward):
        x = values.clone().requires_grad_()
        condition = local.clone().requires_grad_()
        score = forward(network, x, condition, geometry, mask)
        first = torch.autograd.grad(score.sum(), (x, condition), create_graph=True)
        penalty = (first[0].norm(2, dim=1) - 1).square().mean()
        targets = (x, condition, *network.parameters())
        second = torch.autograd.grad(penalty, targets, allow_unused=True)
        return score, first, tuple(torch.zeros_like(p) if g is None else g for p, g in zip(targets, second))

    actual = derivatives(model, lambda m, *args: m(*args))
    expected = derivatives(reference, reference_forward)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-11
    for actual_group, expected_group in zip(actual, expected):
        if isinstance(actual_group, torch.Tensor):
            actual_group, expected_group = (actual_group,), (expected_group,)
        for a, e in zip(actual_group, expected_group):
            torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
    assert actual[1][0][~mask].count_nonzero() == 0


@pytest.mark.parametrize('stations', [0, 1, 2, 19, 80])
def test_empty_singleton_padded_nonfinite_and_permuted_sets_match_reference(stations):
    values, local, geometry, mask = example(stations=stations)
    values[~mask] = float('nan')
    local[~mask] = float('inf')
    model = ObservationCritic().double()
    score = model(values, local, geometry, mask)
    torch.testing.assert_close(score, reference_forward(model, values, local, geometry, mask), atol=1e-12, rtol=1e-11)
    permutation = torch.randperm(stations)
    torch.testing.assert_close(score, model(values[:, permutation], local[:, permutation], geometry[permutation], mask[:, permutation]), atol=1e-12, rtol=1e-11)
    padding = 3
    padded_geometry = torch.cat((geometry, torch.randn(padding, 3, dtype=geometry.dtype) * 10000))
    torch.testing.assert_close(score, model(torch.nn.functional.pad(values, (0, padding), value=float('nan')),
                                          torch.nn.functional.pad(local, (0, 0, 0, padding), value=float('inf')),
                                          padded_geometry, torch.nn.functional.pad(mask, (0, padding))), atol=1e-12, rtol=1e-11)


def test_pair_geometry_bins_keep_lexicographic_order_and_boundaries():
    geometry = torch.tensor([[200000., 0., 0.], [0., 0., 1.], [25000., 0., 0.],
                             [100000., 0., 0.], [50000., 0., 0.], [0., 0., 0.]])
    order = sorted(range(len(geometry)), key=lambda i: tuple(geometry[i].tolist()))
    pairs = list(itertools.combinations(order, 2))
    for (left, right), (low, high) in zip(ObservationCritic._pairs(geometry), [(0, 25000), (25000, 50000), (50000, 100000), (100000, 200000)]):
        expected = [(a, b) for a, b in pairs if low <= abs(float(geometry[a, 0] - geometry[b, 0])) < high]
        assert list(zip(left.tolist(), right.tolist())) == expected


@pytest.mark.parametrize('which', ['values', 'local', 'geometry'])
def test_valid_nonfinite_inputs_still_raise(which):
    values, local, geometry, mask = example(stations=19)
    if which == 'values':
        values[2, 0] = float('nan')
    elif which == 'local':
        local[2, 0, 0] = float('inf')
    else:
        geometry[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        ObservationCritic().double()(values, local, geometry, mask)
