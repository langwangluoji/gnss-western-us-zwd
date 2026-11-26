import importlib
import importlib.util
import numpy as np


def test_residual_energy_reduction_is_distinct_from_centered_predictive_r2():
    assert importlib.util.find_spec('innovation_panel_analysis') is not None, 'panel residual-energy analysis is missing'
    scores=importlib.import_module('innovation_panel_analysis').relative_scores
    r=scores(np.array([1.,0.]),np.array([5.,4.]),np.array([1.,2.]))
    np.testing.assert_allclose(r['relative_residual_energy_reduction'],[.8,1.])
    assert r['predictive_r2'][0]==.75 and np.isnan(r['predictive_r2'][1])
