import numpy as np
from . import features


def test_dry_height_pressure_transport_and_missing_fallback():
    raw=np.array([[[100,280,1000],[120,280,1000]],[[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]],dtype='f4')
    result,valid=features.reference_pressure(raw,np.array([[0.,0.],[10.,0.]]),np.array([0.,0.]),np.array([[0.,0.]]),np.array([1000.]),np.array([[900.],[901.]]))
    np.testing.assert_allclose(result[:,0],[885.129,901.],atol=.01)
    assert valid[:,0].tolist()==[True,False]


def test_vmf_feature_units_anomaly_and_missing_mask():
    # VMF3 zhd/zwd in m; field and final features use mm before scaling.
    zhd=np.array([[2.3],[np.nan]]);zwd=np.array([[.1],[np.nan]])
    dry=np.array([[2.28],[2.28]]);field=np.array([[110.],[110.]])
    result=features.background_features(zhd,zwd,dry,field)
    np.testing.assert_allclose(result[0,0],[1.,1.2,.5,1.],atol=1e-6)
    np.testing.assert_array_equal(result[1,0],np.zeros(4))


def test_missing_epoch_is_not_filled_from_a_future_valid_time():
    requested=np.array([0,6,12]);available=np.array([0,12]);values=np.array([[1.],[3.]])
    result=features.exact_epochs(requested,available,values)
    assert np.isnan(result[1]).all()
    np.testing.assert_array_equal(result[[0,2]],values)
