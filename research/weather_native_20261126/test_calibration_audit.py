import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('calibration_audit.py')
spec=importlib.util.spec_from_file_location('calibration_audit',path)
module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_order_interval_reports_its_exact_finite_rank_reference():
    assert hasattr(module,'rank_interval'), 'finite-member calibration audit is missing'
    samples=np.arange(32.)[:,None]
    lower,upper,reference,ranks=module.rank_interval(samples)
    np.testing.assert_array_equal(lower,[2.]);np.testing.assert_array_equal(upper,[29.])
    assert ranks==[3,30] and reference==27/33
