import pytest
import torch


@pytest.mark.parametrize('kind', ['none', 'mlp', 'gru', 'lstm', 'transformer'])
def test_query_subset_and_reordering_do_not_change_predictions(kind):
    from .temporal_encoders import TemporalFoundation
    torch.manual_seed(7)
    model = TemporalFoundation(kind, length=12, channels=25, width=16, hidden=32).eval()
    x = torch.randn(1, 12, 64, 25)
    uv = torch.rand(1, 9, 2)*2-1
    static = torch.randn(1, 9, 9)
    with torch.no_grad():
        whole = model(x, uv, static)
        subset = model(x, uv[:, [7, 1]], static[:, [7, 1]])
    torch.testing.assert_close(whole[:, :, [7, 1]], subset, atol=2e-6, rtol=1e-5)
    assert whole.shape == (1, 6, 9)
    assert torch.isfinite(whole).all()


@pytest.mark.parametrize('kind', ['gru', 'lstm', 'transformer'])
def test_temporal_states_cannot_access_later_tokens(kind):
    from .temporal_encoders import TemporalFoundation
    torch.manual_seed(8)
    model = TemporalFoundation(kind, length=12, channels=25, width=16, hidden=32).eval()
    x = torch.randn(2, 12, 3, 16)
    other = x.clone()
    other[:, 7:] += 100
    with torch.no_grad():
        a = model.encode_time(x)
        b = model.encode_time(other)
    torch.testing.assert_close(a[:, :7], b[:, :7], atol=1e-6, rtol=1e-5)


def test_forecast_head_has_finite_gradients():
    from .temporal_encoders import TemporalFoundation
    model = TemporalFoundation('gru', length=12, channels=25, width=16, hidden=32)
    output = model(torch.randn(2, 12, 64, 25), torch.zeros(2, 3, 2), torch.zeros(2, 3, 9))
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_delta_model_retains_exact_spatial_persistence():
    from .temporal_encoders import TemporalFoundation
    model = TemporalFoundation('none', length=12, width=16, hidden=32).eval()
    background = torch.tensor([[.5, 1.25]])
    with torch.no_grad():
        output = model(torch.randn(1, 12, 64, 25), torch.zeros(1, 2, 2),
                       torch.zeros(1, 2, 9), background=background)
    torch.testing.assert_close(output, background[:, None].expand(1, 6, 2))
