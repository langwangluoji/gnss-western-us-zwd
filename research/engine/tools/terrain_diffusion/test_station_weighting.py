import torch


def test_condition_distance_weights_use_current_observations_and_exact_boundaries():
    from . import station_weighting
    q = torch.tensor([[0., 0.], [50., 0.], [100., 0.], [200., 0.]])
    c = torch.tensor([[0., 0.], [200., 0.]])
    valid = torch.tensor([[True, False], [False, True], [False, False]])
    actual = station_weighting.distance_weights(q, c, valid)
    expected = torch.tensor([[.278, 2.297, 1.922, .879], [.879, 1.922, 1.922, .278],
                             [.879, .879, .879, .879]])
    torch.testing.assert_close(actual, expected)


def test_uniform_weighted_loss_recovers_original_objective():
    from . import station_weighting
    p = torch.tensor([[2., 4.], [3., 6.]], requires_grad=True)
    loss = station_weighting.weighted_loss(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool), torch.ones_like(p))
    torch.testing.assert_close(loss, torch.tensor(16.5))
    loss.backward(); assert torch.isfinite(p.grad).all()


def test_distance_weights_apply_to_level_and_adjacent_hour_loss():
    from . import station_weighting
    p = torch.tensor([[2., 4.], [3., 6.]], requires_grad=True)
    loss = station_weighting.weighted_loss(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool), torch.tensor([[1., 3.], [1., 3.]]))
    torch.testing.assert_close(loss, torch.tensor(21.45))
    loss.backward(); assert torch.isfinite(p.grad).all()


def test_masked_targets_and_missing_adjacent_pairs_do_not_enter_loss():
    from . import station_weighting
    p = torch.tensor([[2., 1000.], [1000., 6.]], requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    loss = station_weighting.weighted_loss(p, torch.zeros_like(p), mask, torch.tensor([[1., 3.], [1., 3.]]))
    torch.testing.assert_close(loss, torch.tensor(28.))
    loss.backward(); assert torch.equal(p.grad[~mask], torch.zeros(2))
