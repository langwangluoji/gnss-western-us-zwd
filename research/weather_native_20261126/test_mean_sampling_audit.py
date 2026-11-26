import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).with_name('mean_sampling_audit.py')
spec=importlib.util.spec_from_file_location('sampling_audit',path);module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_fair_mean_mse_removes_iid_sampling_variance_without_clipping():
    assert hasattr(module,'fair_mean_mse'),'conditional mean MSE estimator missing'
    # Four equiprobable IID draws from {-1,5}: population mean2, variance9.
    samples=np.array([[-1.,-1.,5.,5.],[-1.,5.,-1.,5.]])
    actual=module.fair_mean_mse(samples,np.zeros(4))
    np.testing.assert_array_equal(actual,np.array([1.,-5.,-5.,25.]))
    assert actual.mean()==4
    assert np.mean(samples.mean(0)**2)==8.5
    with pytest.raises(ValueError):module.fair_mean_mse(samples[:1],np.zeros(4))
    result=module.fair_mean_mse(samples,np.array([0.,np.nan,0.,0.]))
    assert np.isnan(result[1])
