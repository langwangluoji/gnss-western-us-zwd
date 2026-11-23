"""Contracts that distinguish local information from the output subspace."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_cache_rejects_wrong_hour_or_mutated_field(tmp_path):
    from local_context_data import SparseMeanCache
    cache = SparseMeanCache(tmp_path, [11, 13], (2, 3), {'mean': 'locked'})
    cache.put(11, np.ones((2, 3)))
    np.testing.assert_array_equal(cache.get(11), np.ones((2, 3)))
    assert not cache.contains(13)
    with pytest.raises(ValueError):
        cache.put(12, np.ones((2, 3)))
    with pytest.raises(ValueError):
        cache.put(13, np.full((2, 3), np.nan))
    with pytest.raises(ValueError):
        SparseMeanCache(tmp_path, [11, 13], (2, 3), {'mean': 'changed'})


def test_joint_mask_drops_entire_station_and_distinguishes_zero_from_missing():
    from local_context_features import kernel_aggregate
    values = np.array([[0., 1., 2., 3., 4.], [99., np.nan, 9., 9., 9.]])
    kernel = np.ones((2, 2))
    result, density = kernel_aggregate(values, kernel)
    np.testing.assert_allclose(result, [[0, 1, 2, 3, 4]]*2)
    np.testing.assert_allclose(density, [1, 1])
    empty, rho = kernel_aggregate(values*np.nan, kernel)
    assert np.isfinite(empty).all() and not empty.any() and not rho.any()
    perm, prho = kernel_aggregate(values[::-1], kernel[:, ::-1])
    np.testing.assert_array_equal(perm, result)
    np.testing.assert_array_equal(prho, density)


def test_query_runs_nonlinearity_at_vertices_and_preserves_projection_gradient():
    from local_context_features import VertexQuery, LocalGridHead, project_operator
    query = VertexQuery((3, 3), [[.25, .5]], [0.], np.arange(9))
    features = torch.randn(2, 9, 99)
    model = LocalGridHead()
    assert torch.equal(query(model(features)), torch.zeros(2, 1))
    torch.nn.init.normal_(model.net[-1].weight)
    field = model(features)
    from gnss_field.geometry import surface_query
    expected = surface_query(field.reshape(2, 1, 3, 3), [[.25, .5]], [0.])[:, 0]
    torch.testing.assert_close(query(field), expected)
    basis = np.c_[np.ones(9), np.arange(9)]
    land = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], bool)
    operator = torch.tensor(project_operator(basis, land), dtype=torch.float64)
    x = torch.arange(9, dtype=torch.float64).square().requires_grad_(True)
    coefficients = x[land] @ operator.T
    gradient = torch.autograd.grad(coefficients.sum(), x)[0]
    assert (gradient[land].abs()>0).all() and gradient[~land].abs().sum()==0
    np.testing.assert_allclose(operator.numpy() @ basis[land], np.eye(2), atol=1e-12)


def test_feature_normalizer_does_not_read_inner_or_outer_values():
    from local_context_features import fit_normalizer
    features = np.zeros((5, 3, 2), dtype='f4')
    features[:2, :, 0] = np.array([1., 3.])[:, None]
    features[:2, :, 1] = 10
    center, scale = fit_normalizer(features, [0, 1])
    features[2:] = 1e20
    other = fit_normalizer(features, [0, 1])
    np.testing.assert_array_equal(center, [2, 10])
    np.testing.assert_allclose(scale, [1, 1])
    np.testing.assert_array_equal(other[0], center)
    np.testing.assert_array_equal(other[1], scale)


def test_local_feature_schema_uses_common_height_innovation_and_four_frames():
    from local_context_features import LocalFeatureBuilder
    x, y = np.meshgrid(np.arange(3)*25000., np.arange(3)*25000.)
    maker = LocalFeatureBuilder(np.c_[x.ravel(), y.ravel()], [[0., 0.]], [3000.],
        np.full((3, 3), 3000.), np.ones((3, 3), bool), np.arange(9),
        dict(input_mean=[0.]*5, input_std=[1.]*5, z0_std=10.), [[0., 0.]], [True])
    frames = np.ones((4, 1, 5))*20
    field = np.ones((3, 3))*10
    result = maker.build(frames, field, np.arange(32), np.ones(4))
    assert result.shape==(9, 99) and np.isfinite(result).all()
    # A Z0=20 and M0 Z0=10 => standardized innovation 1, regardless of H=3000.
    np.testing.assert_allclose(result[:, 48], 1.)
    np.testing.assert_allclose(result[:, 50], 1.)
    np.testing.assert_allclose(result[:, 58], 1.)
    np.testing.assert_allclose(result[:, 59], 2.)
    np.testing.assert_allclose(result[:, 67:], np.broadcast_to(np.arange(32), (9, 32)))
    frames[0, 0, 1] = np.nan
    missing = maker.build(frames, field, np.zeros(32), np.ones(4))
    assert not missing[:, :6].any() and not missing[:, 48:52].any()
    assert missing[:, 12].sum()>0  # previous 6h frame remains available
    empty = maker.build(frames*np.nan, field, np.zeros(32), np.ones(4))
    assert np.isfinite(empty).all() and not empty[:, :52].any()


def test_three_arms_share_zero_mean_and_all_missing_fallback():
    from local_context_train import ThreeArmMean
    arrays = dict(projection=np.ones((1, 2), 'f4'), operator=np.ones((2, 3), 'f4')/3,
        land=np.ones(3, bool), query_indices=np.array([[0, 1, 2, 0]]),
        query_weights=np.array([[.25]*4], 'f4'))
    for arm in ('PCA32_R64', 'local99_R64_projection', 'local99_grid'):
        model = ThreeArmMean(arm, arrays, 10.)
        x, c = torch.randn(2, 3, 99), torch.randn(2, 32)
        amp, gate = torch.ones(2), torch.tensor([True, False])
        assert torch.equal(model(x, c, amp, gate), torch.zeros(2, 1))
        with torch.no_grad():
            model.head.net[-1].bias.fill_(1.)
        value = model(x, c, amp, gate)
        assert value[0].abs().sum()>0 and value[1].abs().sum()==0
        value.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_ES_rejects_nonfinite_and_cannot_read_held_group_labels():
    from local_context_train import observed_loss
    samples = torch.ones(2, 4, 3, requires_grad=True)
    truth = torch.tensor([[3., 9., 2.], [4., 1., 2.]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    fit = torch.tensor([True, False, True])
    a = observed_loss(samples, truth, valid, fit)
    truth[:, 1] = float('nan')
    b = observed_loss(samples, truth, valid, fit)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    samples.data[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        observed_loss(samples, truth, valid, fit)


def test_linear_control_fits_only_valid_observations():
    from local_context_controls import ridge_statistics
    x = np.array([[[1., 0.], [0., 1.], [1., 1.], [100., 100.]]])
    y = np.array([[2., 3., 5., float('nan')]])
    pop = np.array([[True, True, True, False]])
    normal, rhs, n = ridge_statistics(x, y, pop)
    coefficient = np.linalg.solve(normal, rhs)
    np.testing.assert_allclose(coefficient, [2., 3.], atol=1e-12)
    assert n==3
    y[0, 3]=1e15
    second = ridge_statistics(x, y, pop)
    np.testing.assert_array_equal(normal, second[0])
    np.testing.assert_array_equal(rhs, second[1])


def test_zero_candidate_survives_partial_initial_save_and_controls_patience(tmp_path):
    from local_context_train import initialize_zero
    from gnss_field.contracts import ScienceConfig
    from gnss_field.training.engine import TrainEngine, StageSpec, atomic_torch
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    science = ScienceConfig(warmup=1, startup_evaluation=0, evaluate_every=100,
        log_every=100, minimum_refinement_steps=300, patience=3)
    engine = TrainEngine(model, tmp_path, np.arange(8),
        StageSpec('test_refine', 1000, 1000, 8, 8, 42, {}), science)
    # Reproduce interruption after writing initial.pt and before zero metrics.
    atomic_torch(tmp_path/'initial.pt', engine.state())
    def evaluator(current, step):
        return {'joint': dict(value=float(current.weight.detach().abs().sum()))}
    initialize_zero(engine, evaluator)
    assert (tmp_path/'initial_metrics.json').exists()
    assert engine.products['joint'][0]['step']==0
    def objective(hours, current, step):
        return {'loss': (current(torch.ones(len(hours), 1))-1).square().mean()}
    engine.run(objective, evaluator=evaluator)
    assert engine.step==300  # all trained candidates worse than the zero incumbent
    assert engine.products['joint'][0]['value']==0
    import json
    steps = [json.loads(row)['step'] for row in (tmp_path/'development.jsonl').read_text().splitlines()]
    assert steps==[100, 200, 300]


def test_monitor_shows_recorded_rmse_without_changing_ES_selection():
    from local_context_monitor import display_status
    payload = dict(runs=[dict(name='fold_0/local99_grid', status=dict(stage='local_context_refine',
        products={'joint':[dict(value=5., step=100)]}), training=[],
        development=[dict(step=100, metrics={'joint':dict(value=5., rmse_mm=8.)})]),
        dict(name='engineering_resume/continuous', status={}, training=[], development=[])])
    shown = display_status(payload)
    assert len(shown['runs'])==1
    assert shown['runs'][0]['development'][0]['metrics']['point']['value']==8.
    assert 'point' not in payload['runs'][0]['development'][0]['metrics']
    assert 'point' not in shown['runs'][0]['status']['products']


def test_monitor_ignores_replaceable_status_files_and_reads_append_logs(tmp_path):
    from local_context_monitor import collect_local_status
    folder = tmp_path/'fold_0'/'local99_grid'
    folder.mkdir(parents=True)
    (folder/'status.json').write_bytes(b'\xff invalid and not read')
    (folder/'training.jsonl').write_text('{"step":100,"updated":7,"loss":5}\n')
    result = collect_local_status(tmp_path)
    assert result['runs'][0]['status']['step']==100
    assert result['runs'][0]['status']['updated']==7
    assert result['runs'][0]['training'][0]['loss']==5


def test_server_promotion_requires_controls_fold_consistency_and_interval():
    from local_context_report import promotion_gate
    scores = dict(candidate=9.5, PCA32_R64=10., linear_control=10., zero=10.1)
    intervals = {'PCA32_R64':[-.8,-.1], 'linear_control':[-.7,-.1]}
    folds = [dict(candidate=9.5,PCA32_R64=10.,linear_control=10.)]*2
    assert promotion_gate(scores, folds, intervals)['passed']
    bad = [dict(candidate=9.5,PCA32_R64=10.,linear_control=10.),
           dict(candidate=10.1,PCA32_R64=10.,linear_control=10.)]
    assert not promotion_gate(scores, bad, intervals)['passed']
    assert not promotion_gate(scores, folds, {**intervals,'linear_control':[-.7,.1]})['passed']
