import numpy as np
import pytest

from independent_pair_centering import independent_pair_center


def test_exact_finite_distribution_mean_and_full_covariance():
    states=np.array([[0.,2.],[1.,-1.],[4.,3.]])
    left=np.repeat(states,len(states),axis=0);right=np.tile(states,(len(states),1))
    base=np.array([10.,20.]);samples=np.concatenate([left,right])
    result=independent_pair_center(samples,base)
    np.testing.assert_allclose(result.mean(0),base,atol=1e-14)
    np.testing.assert_allclose(np.cov(result,rowvar=False,bias=True),np.cov(states,rowvar=False,bias=True),atol=1e-14)
    reverse=independent_pair_center(np.concatenate([right,left]),base)
    np.testing.assert_allclose(result+reverse,np.broadcast_to(2*base,result.shape),atol=1e-14)


def test_pairs_are_disjoint_and_query_subset_is_consistent():
    samples=np.arange(32*5*7).reshape(32,5,7).astype('float32');base=np.full((5,7),100.)
    result=independent_pair_center(samples,base)
    assert result.shape==(16,5,7)
    np.testing.assert_allclose(result,base+(samples[:16].astype('float64')-samples[16:])/np.sqrt(2))
    np.testing.assert_array_equal(result[:,[0,3]],independent_pair_center(samples[:,[0,3]],base[[0,3]]))


def test_invalid_member_population_and_nonfinite_inputs_fail():
    for count in [1,2,3,5]:
        with pytest.raises(ValueError,match='even'):
            independent_pair_center(np.zeros((count,3)),np.zeros(3))
    with pytest.raises(ValueError,match='shape'):
        independent_pair_center(np.zeros((4,3)),np.zeros(4))
    with pytest.raises(ValueError,match='finite'):
        independent_pair_center(np.full((4,3),np.nan),np.zeros(3))
