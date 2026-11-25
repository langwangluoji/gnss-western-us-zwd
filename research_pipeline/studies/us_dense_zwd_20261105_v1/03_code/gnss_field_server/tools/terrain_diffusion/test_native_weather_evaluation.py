import numpy as np
from .native_weather_evaluation import fair_crps,point_metrics,gaussian_crps


def test_fair_crps_uses_distinct_iid_member_pairs_and_masks_truth():
    samples=np.array([[-1.,0.,2.],[1.,2.,4.]])
    truth=np.array([0.,1.,np.nan])
    value=fair_crps(samples,truth)
    np.testing.assert_allclose(value[:2],[0.,0.])
    assert np.isnan(value[2])
    np.testing.assert_allclose(fair_crps(samples[::-1],truth),value,equal_nan=True)


def test_macro_preserves_per_layout_station_population_and_time_boundaries():
    truth=np.zeros((2,2,3,2,7));prediction=truth.copy()
    prediction[0,...]=1;prediction[1,...]=3
    result=point_metrics(prediction,truth)
    assert result['horizon_macro_rmse_mm']==[2.]*7
    assert abs(result['horizon_micro_rmse_mm'][0]-np.sqrt(5))<1e-12
    assert result['current_hourly_increment_macro_rmse_mm']==0
    # Calendar gaps are separate month axis; no increments may connect them.
    prediction[:,1]+=10;truth[:,1]+=10
    assert point_metrics(prediction,truth)['current_hourly_increment_macro_rmse_mm']==0


def test_gaussian_crps_known_normal_and_zero_scale_limit():
    score=gaussian_crps(np.array([0.,2.]),np.array([1.,1e-10]),np.array([0.,1.]))
    np.testing.assert_allclose(score,[.23369497725510913,1.],atol=1e-8)
