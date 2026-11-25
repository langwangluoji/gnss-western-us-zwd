import torch


def test_spatial_anomaly_removes_height_scaled_common_field():
    from . import station_anomaly
    height = torch.tensor([0., 700., 1900., 2300.])
    common = torch.tensor([80., 140., 95.])[:, None]
    values = common * torch.exp(-height[None]/3000)
    xy = torch.tensor([[0., 0.], [60., 0.], [0., 90.], [120., 50.]])
    residual, valid = station_anomaly.leave_group_out_anomaly(values, height, xy, torch.arange(4))
    assert valid.all()
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=3e-6, rtol=0)


def test_spatial_anomaly_excludes_same_physical_group_and_missing_observations():
    from . import station_anomaly
    values = torch.tensor([[100., 100., 50.], [float('nan'), 100., float('nan')]])
    height = torch.zeros(3); xy = torch.tensor([[0., 0.], [0., 0.], [50., 0.]])
    residual, valid = station_anomaly.leave_group_out_anomaly(values, height, xy, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(residual[0], torch.tensor([5., 5., -5.]))
    assert valid[0].all() and not valid[1].any()
    assert torch.isfinite(residual).all()
    assert torch.equal(residual[1], torch.zeros(3))


def test_spatial_anomaly_is_station_permutation_equivariant_and_hour_local():
    from . import station_anomaly
    torch.manual_seed(81)
    values = torch.rand(8, 5)*80+20
    height = torch.rand(5)*1500; xy = torch.rand(5, 2)*200; groups = torch.arange(5)
    a, am = station_anomaly.leave_group_out_anomaly(values, height, xy, groups)
    order = torch.tensor([3, 0, 4, 1, 2])
    b, bm = station_anomaly.leave_group_out_anomaly(values[:, order], height[order], xy[order], groups[order])
    torch.testing.assert_close(b, a[:, order]); assert torch.equal(bm, am[:, order])
    changed = values.clone(); changed[4:] += 1000
    c, cm = station_anomaly.leave_group_out_anomaly(changed, height, xy, groups)
    torch.testing.assert_close(c[:4], a[:4], atol=0, rtol=0)
    assert torch.equal(cm[:4], am[:4])
