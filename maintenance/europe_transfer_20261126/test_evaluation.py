import numpy as np
import pytest
from . import evaluate


def test_rmse_uses_same_station_epoch_keys_for_all_models():
    target=np.array([[10.,20.],[30.,40.]])
    a=np.array([[11.,np.nan],[32.,43.]])
    b=np.array([[12.,24.],[np.nan,44.]])
    result=evaluate.common_rmse(target,{'a':a,'b':b},np.ones(2,dtype=bool))
    assert result['records']==2
    assert result['models']['a']['micro_rmse_mm']==pytest.approx(np.sqrt(5))
    assert result['models']['b']['micro_rmse_mm']==pytest.approx(np.sqrt(10))


def test_train_station_cannot_enter_reported_spatial_holdout():
    target=np.array([[0.,0.]])
    result=evaluate.common_rmse(target,{'a':np.array([[100.,3.]])},np.array([False,True]))
    assert result['records']==1
    assert result['models']['a']['micro_rmse_mm']==3
