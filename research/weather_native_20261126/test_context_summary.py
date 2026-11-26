import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('context_summary.py')
spec=importlib.util.spec_from_file_location('context_summary',path)
module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_score_interval_preserves_valid_counts_under_month_resampling():
    assert hasattr(module,'score_interval'), 'paired probability summary is missing'
    rng=np.random.default_rng(39);a=rng.normal(size=(1,12,5,3,7));b=rng.normal(size=a.shape)
    a[:,:6,1:]=np.nan
    rng=np.random.default_rng(261126);expected=[]
    for _ in range(150):
        index=rng.integers(12,size=12)
        expected.append(np.nanmean(a[:,index,...,0])-np.nanmean(b[:,index,...,0]))
    np.testing.assert_allclose(module.score_interval(a,b,0,150),np.quantile(expected,[.025,.975]))
