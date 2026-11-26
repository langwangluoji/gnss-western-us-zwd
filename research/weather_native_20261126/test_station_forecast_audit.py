import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('station_forecast_audit.py')
spec=importlib.util.spec_from_file_location('station_forecast_audit',path)
module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_increment_audit_keeps_physical_units_and_variable_masks():
    assert hasattr(module,'increment_scores'), 'physical increment audit is missing'
    truth=np.ones((2,4,6,3))*[10.,2.,2.];prediction=truth.copy()
    truth[0,0,0,1]=np.nan
    result=module.increment_scores(prediction,truth)
    np.testing.assert_allclose(result['prediction_rmse_by_lead_variable'],np.zeros((6,3)))
    np.testing.assert_allclose(result['persistence_rmse_by_lead_variable'],np.ones((6,3))*[10.,2.,2.])
    assert result['valid_count_by_lead_variable'][0]==[8,7,8]
