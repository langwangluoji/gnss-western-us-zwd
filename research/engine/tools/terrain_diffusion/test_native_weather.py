import numpy as np
import pytest
import torch

from .native_weather import graph_basis, full_spectrum, weather_fields, history_features, validate_origin


def test_full_spectrum_preserves_all_information_and_energy():
    xy=torch.tensor([[0.,0.],[35.,4.],[120.,-20.],[400.,10.],[600.,-50.]])
    eigen,U=graph_basis(xy,torch.zeros(5))
    x=torch.randn(3,5,7)
    z=full_spectrum(x,U)
    torch.testing.assert_close(full_spectrum(z,U,inverse=True),x,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(z.square().sum(),x.square().sum(),atol=2e-5,rtol=2e-6)
    assert eigen.min()>-1e-5 and eigen.max()<2.001


def test_weather_has_separate_variable_masks_and_no_missing_value_bias():
    raw=torch.tensor([[[100.,280.,float('nan')],[100.,float('nan'),900.]]])
    x=weather_fields(raw,torch.zeros(1,2),torch.zeros(2,2),torch.zeros(1),torch.zeros(2))
    assert x.shape==(1,1,18)
    # Each radius: normalized means ZWD=1,T=0,P=0, then variable coverage.
    for k in (0,6,12):
        torch.testing.assert_close(x[0,0,k:k+3],torch.tensor([1.,0.,0.]))
        torch.testing.assert_close(x[0,0,k+3:k+6],torch.tensor([1.,.5,.5]))


def test_history_features_are_causal_and_keep_lags():
    T=50;Q=2
    innovation=torch.randn(T,Q,27);background=torch.randn(T,Q)
    weather=torch.randn(T,Q,18);hours=torch.arange(T);static=torch.randn(Q,9)
    nearest=torch.ones(2,Q)*50;positions=torch.tensor([30,31])
    a=history_features(innovation,background,weather,hours,positions,static,nearest)
    innovation[32:]=1e9;background[32:]=1e9;weather[32:]=1e9
    b=history_features(innovation,background,weather,hours,positions,static,nearest)
    torch.testing.assert_close(a,b)
    assert a.shape==(2,Q,111) and torch.isfinite(a).all()
    with pytest.raises(ValueError):
        history_features(innovation,background,weather,hours,torch.tensor([23]),static,nearest[:1])


def test_forecast_contract_rejects_future_dynamic_information_and_split_crossing():
    validate_origin(np.arange(100,112),np.arange(100,112),0,120,72,6)
    with pytest.raises(ValueError):validate_origin(np.arange(100,112),np.arange(101,113),0,120,72,6)
    with pytest.raises(ValueError):validate_origin(np.arange(100,112),np.arange(100,112),0,115,72,6)
    with pytest.raises(ValueError):validate_origin(np.arange(50,62),np.arange(50,62),0,120,72,6)
