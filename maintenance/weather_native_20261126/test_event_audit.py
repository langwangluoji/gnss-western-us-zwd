import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('event_audit.py')
spec=importlib.util.spec_from_file_location('event_audit',path);module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_fair_brier_removes_member_noise_with_missing_truth():
    assert hasattr(module,'fair_brier'),'IID-corrected Brier score missing'
    events=np.array([[0.,1.,0.],[1.,1.,1.]])
    actual=module.fair_brier(events,np.array([1.,0.,np.nan]))
    np.testing.assert_array_equal(actual[:2],[0.,1.]);assert np.isnan(actual[2])


def test_tail_weighted_crps_scores_censored_increment_distributions():
    assert hasattr(module,'tail_crps'),'tail-weighted CRPS missing'
    x=np.array([[0.,0.],[2.,-2.]])
    up=module.tail_crps(x,np.array([3.,-3.]),1,'increase')
    down=module.tail_crps(x,np.array([3.,-3.]),1,'decrease')
    np.testing.assert_array_equal(up,[1.,0.]);np.testing.assert_array_equal(down,[0.,1.])
