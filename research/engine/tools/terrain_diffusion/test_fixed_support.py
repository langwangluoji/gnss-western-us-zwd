import torch
from .fixed_support import height_transport, masked_multiscale_summary, FixedSupportModel


def test_masked_multiscale_summary_marks_empty_support_finite():
    query_xy = torch.tensor([[0.0, 0.0], [100.0, 0.0]])
    query_h = torch.tensor([100.0, 200.0])
    cond_xy = torch.tensor([[1000.0, 0.0]])
    cond_h = torch.tensor([500.0])
    values = torch.tensor([[10.0]])
    mask = torch.tensor([[False]])
    out = masked_multiscale_summary(query_xy, query_h, cond_xy, cond_h, values, mask)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, 2] == 0)  # first-scale log support count


def test_height_transport_round_trip_is_explicit():
    zwd = torch.tensor([10.0, 20.0])
    cond_h = torch.tensor([0.0, 1000.0])
    query_h = torch.tensor([500.0, 500.0])
    out = height_transport(zwd, cond_h, query_h, scale_km=3.0)
    expected = zwd * torch.exp((cond_h - query_h) / 3000.0)
    assert torch.allclose(out, expected)


def test_model_support_branch_is_query_set_invariant_and_finite():
    torch.manual_seed(0)
    model = FixedSupportModel(point_dim=113, summary_dim=30, width=8)
    point = torch.randn(4, 113)
    support = torch.randn(1, 30, 8, 8)
    uv_a = torch.tensor([[[[-0.5, -0.5], [0.5, 0.5]]]])
    uv_b = torch.tensor([[[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]]])
    point_b = torch.cat([point[:1], torch.randn(1, 113), point[1:2]], 0)
    out_a = model(point[:2], support, uv_a)
    out_b = model(point_b, support, uv_b)
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[2], atol=1e-6)
