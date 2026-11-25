import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('mean_analysis',Path(__file__).with_name('analyze_means.py'))
module=importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():spec.loader.exec_module(module)


def test_macro_and_paired_month_interval_keep_station_weighting():
    assert hasattr(module,'macro'), 'independent analysis is missing'
    error=np.ones((2,3,4,2,7));error[1]*=3;error[:,:,:,1]*=2
    np.testing.assert_allclose(module.macro(error),[3.]*7)
    assert module.month_interval(error,error,draws=20)==[0.,0.]
    a=np.ones((2,3,4,2,7));b=a*2
    np.testing.assert_allclose(module.month_interval(a,b,draws=20),[-1.,-1.])


def test_station_pair_error_cancels_common_bias_and_masks_missing():
    assert hasattr(module,'pair_errors'), 'station pair diagnostic is missing'
    error=np.ones((1,1,3,3,7))*5;xy=np.array([[0.,0.],[40.,0.],[200.,0.]])
    result=module.pair_errors(error,xy)
    assert result['0-50']['pair_count']==1
    assert result['0-50']['current_rmse_mm']==0
    error[...,1,0]+=2
    assert module.pair_errors(error,xy)['0-50']['current_rmse_mm']==2


def test_old_loss_curve_remains_readable_without_inventing_new_diagnostics():
    assert hasattr(module,'curve_values'), 'curve schema compatibility is missing'
    old=[dict(step=1,loss=2.,current_correction_rms_mm=0.)]
    assert module.curve_values(old,'query_loss')==[2.]
    assert np.isnan(module.curve_values(old,'future6_correction_rms_mm')[0])


def test_month_interval_resamples_actual_valid_hours_when_coverage_differs():
    data_rng=np.random.default_rng(101)
    a=data_rng.normal(size=(1,12,5,3,1));b=data_rng.normal(size=a.shape)
    a[:,0:6,1:]=np.nan
    rng=np.random.default_rng(261126);expected=[]
    for _ in range(150):
        index=rng.integers(12,size=12)
        expected.append(float(module.macro(a[:,index])[0]-module.macro(b[:,index])[0]))
    np.testing.assert_allclose(module.month_interval(a,b,draws=150),np.quantile(expected,[.025,.975]))


def test_report_json_marks_empty_groups_and_open_bounds_as_null():
    assert hasattr(module,'plain'), 'strict public JSON export is missing'
    assert module.plain(dict(a=[np.nan,np.inf,-np.inf,1.]))==dict(a=[None,None,None,1.])
