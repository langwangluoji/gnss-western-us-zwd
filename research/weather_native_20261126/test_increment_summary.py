import importlib
import importlib.util
import numpy as np
import pytest


def function():
    assert importlib.util.find_spec('increment_summary') is not None, 'increment paired RMSE estimator is missing'
    return importlib.import_module('increment_summary').rmse_comparison


def record(sums,counts):
    return dict(monthly_score_sum=sums,monthly_valid_count=counts)


def test_primary_is_mean_seed_root_pooled_mse_and_shared_month_interval():
    f=function()
    # Within each seed all months have fixed MSE; across seeds mean RMSE != pooled RMSE.
    result=f([(record([4,8],[1,2]),record([9,18],[1,2])),
        (record([16,32],[1,2]),record([25,50],[1,2]))])
    assert result['candidate']==3 and result['reference']==4 and result['delta']==-1
    np.testing.assert_allclose(result['delta_ci95'],[-1,-1])


def test_counts_weight_each_month_and_changed_populations_rejected():
    f=function();a=record([1,900],[1,100]);b=record([4,1600],[1,100])
    result=f([(a,b)],draws=50)
    assert result['candidate']==pytest.approx((901/101)**.5)
    with pytest.raises(ValueError,match='population'):
        f([(a,record([4,1600],[1,99]))])
