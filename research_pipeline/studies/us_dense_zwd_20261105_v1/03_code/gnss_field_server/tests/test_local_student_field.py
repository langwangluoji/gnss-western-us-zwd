"""Student marginal scores and global scale-mixture controls."""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import t
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def test_student_crps_matches_numerical_cdf_integral():
    from local_student_field import student_crps
    for x in (-3., .7, 4.):
        cdf = lambda y: t.cdf(y/2., df=7.)
        expected = quad(lambda y: cdf(y)**2, -np.inf, x)[0]+quad(lambda y: (1-cdf(y))**2, x, np.inf)[0]
        np.testing.assert_allclose(student_crps(x, 2., 7.), expected, atol=1e-8)


def test_student_common_radial_scale_is_member_prefix_invariant():
    from local_student_field import radial_scales
    a = radial_scales([422000, 422006], 128, 7., .9, 42)
    b = radial_scales([422006], 64, 7., .9, 42)
    np.testing.assert_array_equal(a[1, :64], b[0])
