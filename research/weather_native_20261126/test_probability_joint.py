import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('probability_joint_audit.py')
spec=importlib.util.spec_from_file_location('joint_audit',path)
module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_fair_energy_uses_distinct_iid_pairs_and_physical_dimension_scaling():
    assert hasattr(module,'fair_energy'), 'joint probability score is missing'
    samples=np.array([[[-1.,-1.]],[[1.,1.]]]);truth=np.zeros((1,2))
    np.testing.assert_allclose(module.fair_energy(samples,truth),[0.],atol=1e-12)
    np.testing.assert_allclose(module.fair_energy(samples[::-1],truth),[0.],atol=1e-12)
    samples[:]=1;truth[:,1]=np.nan
    np.testing.assert_allclose(module.fair_energy(samples,truth),[1.])


def test_joint_energy_matches_direct_member_sum_with_missing_coordinates():
    rng=np.random.default_rng(37);x=rng.normal(size=(5,3,7));y=rng.normal(size=(3,7));y[0,1:4]=np.nan
    expected=[]
    for origin in range(3):
        keep=np.isfinite(y[origin]);a=x[:,origin,keep];b=y[origin,keep]
        first=np.mean([np.linalg.norm(v-b) for v in a])
        second=sum(np.linalg.norm(a[i]-a[j]) for i in range(5) for j in range(5) if i!=j)/(2*5*4)
        expected.append((first-second)/np.sqrt(keep.sum()))
    np.testing.assert_allclose(module.fair_energy(x,y),expected,atol=1e-12)
