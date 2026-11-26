import numpy as np
from directed_v2_error_audit import station_statistics


def test_station_rmse_preserves_layout_then_station_macro_contract():
    error=np.ones((3,1,2,2,7),dtype='float64')*np.arange(1,4)[:,None,None,None,None]
    error[:,:,:,1]=np.nan
    result=station_statistics(error)
    # Average the three layout-specific station roots; pooling first gives sqrt(14/3).
    np.testing.assert_allclose(result['rmse'][0],2,atol=0,rtol=0)
    assert np.isnan(result['rmse'][1]).all()
    np.testing.assert_array_equal(result['count'][0],np.full(7,6))
    np.testing.assert_array_equal(result['count'][1],np.zeros(7))
