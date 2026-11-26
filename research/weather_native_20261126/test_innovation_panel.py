import importlib
import importlib.util
import numpy as np
import pytest
from innovation_panel_campaign import schedule


def test_schedule_is_label_independent_fixed_months_and_chooses_first_duplicate():
    hours=np.array([int(np.datetime64(f'{y}-{m:02d}-08T01','h').astype('int64')) for y in range(2016,2020) for m in range(1,13)])
    a=hours[:,None]+np.arange(12);a=np.r_[a,a[:3]]
    rows=schedule(a)
    assert len(rows)==48 and [r['index'] for r in rows]==list(range(48))
    a[0,1]+=1
    with pytest.raises(ValueError,match='consecutive'):schedule(a)


def test_probe_standardization_and_coefficients_never_use2019_labels_or_features():
    assert importlib.util.find_spec('innovation_panel_summary') is not None, 'panel probe fitter is missing'
    f=importlib.import_module('innovation_panel_summary').fit_probe
    rng=np.random.default_rng(7);x=rng.normal(size=(8,5,2));y=rng.normal(size=(8,5,7));w=np.ones((8,5))
    years=np.array([2016,2016,2017,2017,2018,2018,2019,2019])
    before=f(x,y,w,years)
    x[years==2019]=1e6;y[years==2019]=1e8
    after=f(x,y,w,years)
    for key in ['center','scale','coefficient']:np.testing.assert_array_equal(before[key],after[key])
