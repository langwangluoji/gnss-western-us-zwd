from pathlib import Path
import importlib.util
import numpy as np


def test_k16_randomized_rank_interval_has_80_percent_exchangeable_coverage():
    path=Path(__file__).with_name('diagnostics.py')
    assert path.exists(),'fusion supplementary diagnostics missing'
    spec=importlib.util.spec_from_file_location('fusion_diagnostics_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(123);draw=rng.standard_normal((17,80000))
    assert abs(np.nanmean(m.randomized_rank80(draw[:16],draw[16]))-.8)<.005


def test_k32_rank_interval_omits_missing_truth_and_has_80_percent_reference():
    path=Path(__file__).with_name('diagnostics.py')
    assert path.exists(),'fusion supplementary diagnostics missing'
    spec=importlib.util.spec_from_file_location('fusion_diagnostics_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rng=np.random.default_rng(456);draw=rng.standard_normal((33,80000));truth=draw[32].copy();truth[:100]=np.nan
    value=m.randomized_rank80(draw[:32],truth)
    assert np.isnan(value[:100]).all()
    assert abs(np.nanmean(value)-.8)<.005
