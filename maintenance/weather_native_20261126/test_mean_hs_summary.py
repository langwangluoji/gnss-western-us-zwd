import numpy as np
import pytest
from mean_hs_summary import risk_interaction, safe_comparison


def record(value):
    return dict(value=value,monthly_score_sum=[value*2,value*4],monthly_valid_count=[2,4])


def test_hs_interaction_uses_same_observations_and_signed_mse_difference():
    row=risk_interaction(record(1),record(1.2),record(1.3),record(1.5))
    np.testing.assert_allclose(row['monthly_score_sum'],[0,0],atol=1e-14)
    assert row['value']==pytest.approx(0)
    row=risk_interaction(record(.8),record(1.2),record(1.3),record(1.5))
    assert row['value']==pytest.approx(-.2)


def test_hs_interaction_rejects_changed_valid_population():
    a=record(1);b={**a,'monthly_valid_count':[2,3]}
    with pytest.raises(ValueError,match='population'):
        risk_interaction(a,a,a,b)


def test_sparse_strata_keep_descriptive_values_without_invalid_month_intervals():
    a=dict(monthly_score_sum=[1]+[0]*11,monthly_valid_count=[1]+[0]*11)
    b=dict(monthly_score_sum=[2]+[0]*11,monthly_valid_count=[1]+[0]*11)
    result=safe_comparison([(a,b),(a,b)])
    assert result['candidate']==1 and result['reference']==2
    assert result['delta']==-1 and result['delta_ci95'] is None
    assert result['positive_month_counts']==[1,1]
    with pytest.raises(ValueError,match='population'):
        safe_comparison([(a,{**b,'monthly_valid_count':[2]+[0]*11})])
