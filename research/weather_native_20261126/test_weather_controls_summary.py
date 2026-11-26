import numpy as np
import pytest

from weather_controls_summary import monthly_record, paired_record, fair_mean_mse, event_records


def test_monthly_pooling_uses_observation_counts_and_same_seed_weight():
    a=np.array([[1.,np.nan],[3.,5.],[7.,9.]])
    row=monthly_record(a,0)
    assert row['monthly_valid_count']==[1,2,2]
    assert row['value']==5
    reference=monthly_record(a+2,0)
    result=paired_record([(row,reference),(row,reference)],draws=100)
    assert result['delta']==pytest.approx(-2)
    np.testing.assert_allclose(result['delta_ci95'],[-2,-2])


def test_pairing_rejects_different_observation_population():
    a=monthly_record(np.ones((3,2)),0)
    b=monthly_record(np.array([[1.,np.nan],[1.,1.],[1.,1.]]),0)
    with pytest.raises(ValueError,match='population'):
        paired_record([(a,b)])


def test_fair_mean_mse_removes_unbiased_member_variance_and_preserves_nan():
    samples=np.array([[-1.,0.],[1.,2.]])
    truth=np.array([2.,np.nan])
    result=fair_mean_mse(samples,truth)
    assert result[0]==pytest.approx(3.)
    assert np.isnan(result[1])
    with pytest.raises(ValueError,match='two'):
        fair_mean_mse(samples[:1],truth)


def test_event_scores_exclude_missing_targets_and_use_all_valid_origins():
    samples=np.zeros((2,1,2,1,1,7));samples[0,...,6]=-10;samples[1,...,6]=30
    truth=np.zeros((1,2,1,1,7));truth[:,0,...,6]=25;truth[:,1,...,6]=np.nan
    records=event_records(samples,truth,{'thresholds_mm':[20],'sides':['increase','decrease']})
    for side in ['increase','decrease']:
        for metric in ['brier','tail']:
            record=records[f'event20_{side}_{metric}']
            assert record['monthly_valid_count']==[1,0]
            assert record['value']==pytest.approx(0.)
