import importlib
import importlib.util
import numpy as np


def test_randomized_finite_ensemble_rank_interval_has_exact_80_percent_reference():
    assert importlib.util.find_spec('inducing_calibration_audit') is not None,'finite-member calibration audit missing'
    m=importlib.import_module('inducing_calibration_audit')
    # Enumerate each possible truth rank among 33 exchangeable distinct values.
    truth=np.arange(33,dtype='float64');samples=np.stack([np.delete(truth,j) for j in range(33)],axis=1)
    scores=m.rank_coverage(samples,truth)
    np.testing.assert_allclose(scores['rank3_30'].mean(),27/33,rtol=0,atol=1e-15)
    np.testing.assert_allclose(scores['rank4_29'].mean(),25/33,rtol=0,atol=1e-15)
    np.testing.assert_allclose(scores['randomized80'].mean(),.8,rtol=0,atol=1e-15)
    truth[0]=np.nan
    assert np.isnan(m.rank_coverage(samples,truth)['randomized80'][0])
