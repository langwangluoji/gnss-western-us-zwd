import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path=Path(__file__).with_name('fusion_contract.py')
    assert path.exists(),'new fusion source/moment contract is missing'
    spec=importlib.util.spec_from_file_location('fusion_contract_under_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_directed_identity_and_cache_are_required_for_every_embedding():
    m=module();train=dict(model_kind='transformer',mean_variant='directed_v2_joint',weights='last',
        mean_weights_sha256='joint',mean_protocol_sha256='p',complete=True,cache_manifest_sha256='train')
    dev={**train,'cache_manifest_sha256':'dev'}
    m.validate_embedding_identity(dev,train,'dev','joint','p')
    for key,value in [('mean_variant','physics_aux'),('weights','ema'),('mean_weights_sha256','otherseed'),
                      ('mean_protocol_sha256','old'),('cache_manifest_sha256','train'),('complete',False)]:
        with pytest.raises(ValueError):m.validate_embedding_identity({**dev,key:value},train,'dev','joint','p')


def test_normalization_uses_only_finite_training_statistics():
    m=module();s=dict(role='training_only',bias_mm=[0.]*7,scale_mm=[10.]*7,count=[100]*7)
    m.require_statistics(s)
    for bad in ({**s,'role':'descriptive_development_not_fit'},{**s,'scale_mm':[0.]*7},
                {**s,'bias_mm':[float('nan')]*7},{**s,'count':[0]*7}):
        with pytest.raises(ValueError):m.require_statistics(bad)


def test_pairing_uses_disjoint_pairs_and_cancels_any_source_location():
    m=module();x=np.arange(32*2*3*7,dtype='float64').reshape(32,2,3,7);mu=np.ones((2,3,7))*2
    y=m.centered_pairs(x,mu)
    np.testing.assert_allclose(y,mu+(x[0::2]-x[1::2])/np.sqrt(2),rtol=0,atol=0)
    np.testing.assert_allclose(m.centered_pairs(x+50,mu),y,rtol=0,atol=0)
    with pytest.raises(ValueError):m.centered_pairs(x[:31],mu)
    with pytest.raises(ValueError):m.centered_pairs(x,mu[...,0])


def test_independent_pairing_retains_complete_covariance_and_target_mean():
    m=module();rng=np.random.default_rng(17);factor=np.array([[1.,0,0],[.5,2,0],[-.2,.3,1.]])
    white=rng.standard_normal((160000,3));x=white@factor.T+np.array([5,10,-3]);mu=np.array([1.,2.,3.])
    y=m.centered_pairs(x,mu)
    np.testing.assert_allclose(y.mean(0),mu,atol=.02,rtol=0)
    np.testing.assert_allclose(np.cov(y.T),factor@factor.T,atol=.045,rtol=0)


def test_completed_artifact_lineage_cannot_reuse_old_probability_sources():
    m=module();expected=dict(seed=42,mean='joint',code={'model':'frozen'},cache='training')
    m.require_equal_receipt(expected,expected)
    for key,value in [('seed',17),('mean','TF3000'),('cache','development'),('code',{})]:
        with pytest.raises(ValueError):m.require_equal_receipt({**expected,key:value},expected)


def test_complete_trajectory_clock_counts_real_labels_not_nominal_blocks():
    m=module();y=np.zeros((3,2,3,7));y[1]=np.nan;y[2,0,0,3]=np.nan
    assert m.trajectory_clock(y)==dict(supervised_updates=2,complete_trajectories=11)


def test_reused_evaluation_must_keep_registered_solver_and_spec():
    m=module();m.require_recipe(dict(spec_sha256='spec'),dict(spec_sha256='spec'))
    with pytest.raises(ValueError):m.require_recipe(dict(spec_sha256='changed'),dict(spec_sha256='spec'))
    expected=dict(solver='heun0',sampling_steps=20)
    for actual in (dict(solver='ddim',sampling_steps=20),dict(solver='heun0',sampling_steps=40)):
        with pytest.raises(ValueError):m.require_recipe(actual,expected)
