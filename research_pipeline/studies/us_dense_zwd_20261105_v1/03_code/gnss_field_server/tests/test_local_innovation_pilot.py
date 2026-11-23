"""Contracts that prevent label leakage, query changes and false covariance fits."""
import numpy as np
import pytest
import torch

from local_innovation_components import (InnovationHead, aggregate_innovation,
    nested_split, query_nodes, station_macro_mse, gaussian_records,
    gaussian_objective, fit_gaussian, epoch_batches)


def test_innovation_ignores_missing_A_and_preserves_station_permutation():
    k = np.array([[1., 2., 0.], [0., 0., 0.]])
    a = np.array([4., np.nan, 7.]); base = np.array([1., 1., 2.])
    valid = np.array([True, False, True])
    actual = aggregate_innovation(a, base, k, valid)
    np.testing.assert_array_equal(actual, [3., 0.])
    order = [2, 0, 1]
    np.testing.assert_array_equal(actual, aggregate_innovation(a[order], base[order], k[:, order], valid[order]))


def test_spatial_weather_split_keeps_physical_and_dense_groups_and_buffers():
    h = np.arange(np.datetime64('2018-01-01', 'h').astype(int), np.datetime64('2019-01-01', 'h').astype(int), 6)
    fit = np.ones(8, bool)
    split = nested_split(h, fit, np.array([0, 0, 1, 2, 3, 4, 5, 6]),
                         np.array(['a', 'b', 'c', 'c', 'd', 'e', 'f', 'g']), seed=42)
    assert len(split['week_starts']) == 4
    for key in ('physical', 'dense'):
        labels = np.array([0, 0, 1, 2, 3, 4, 5, 6]) if key == 'physical' else np.array(['a','b','c','c','d','e','f','g'])
        assert not set(labels[split['fit']]) & set(labels[split['inner']])
    for start in split['week_starts']:
        assert not np.any((split['train_hours'] >= start-18) & (split['train_hours'] < start+168+18))
    assert not np.intersect1d(split['train_hours'], split['inner_hours']).size


@pytest.mark.parametrize('arm', ['constant', 'gated', 'free'])
def test_zero_head_and_compact_corner_query_match_full_grid_with_gradients(arm):
    torch.manual_seed(42)
    model = InnovationHead(arm, scale=5.).double()
    f = torch.randn(2, 9, 67, dtype=torch.float64)
    j = torch.randn(2, 9, 2, dtype=torch.float64)
    qi = torch.tensor([[0,1,3,4], [4,5,7,8]])
    qw = torch.tensor([[.25,.25,.25,.25], [.1,.2,.3,.4]], dtype=torch.float64)
    assert torch.equal(model(f,j), torch.zeros(2,9,dtype=torch.float64))
    with torch.no_grad():
        for p in model.parameters(): p.add_(.02)
    nodes, inverse = torch.unique(qi, sorted=True, return_inverse=True)
    full = query_nodes(model(f,j), qi, qw)
    compact = query_nodes(model(f[:,nodes], j[:,nodes]), inverse, qw)
    torch.testing.assert_close(full, compact, rtol=1e-12, atol=1e-12)
    full.sum().backward()
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


def test_macro_loss_has_fixed_station_weights_and_rejects_nonfinite_active_values():
    p = torch.tensor([[2.,4.],[4.,10.]])
    y = torch.tensor([[0.,0.],[0.,float('nan')]])
    m = torch.tensor([[True,True],[True,False]])
    # Means of squared errors: station0=(4+16)/2, station1=16; macro=13.
    assert station_macro_mse(p,y,m,torch.tensor([2,1]),2).item() == 13.
    y[0,0] = float('nan')
    with pytest.raises(ValueError): station_macro_mse(p,y,m,torch.tensor([2,1]),2)


def test_epoch_batches_include_tail_once_without_crossing_epoch():
    batches = epoch_batches(np.arange(11), 8, 42, 2)
    assert [len(x) for x in batches] == [8,3,8,3]
    for off in (0,2): np.testing.assert_array_equal(np.sort(np.concatenate(batches[off:off+2])), np.arange(11))


def test_gaussian_composite_nll_and_gradient_match_direct_likelihood():
    e = np.array([[1.,2.],[2.,3.]])
    h = np.array([[1.,0.],[.5,1.]])
    mask = np.ones_like(e, bool)
    rec = gaussian_records(e,mask,np.ones(2),h@h.T,np.array([0,0]))
    theta = np.log([.7,.9]); tau = 1.2
    loss, grad = gaussian_objective(theta,rec,tau)
    cov = .7**2*(h@h.T)+(.9*tau)**2*np.eye(2)
    exact = np.mean([.5*(np.linalg.slogdet(cov)[1]+r@np.linalg.solve(cov,r)+2*np.log(2*np.pi))/2 for r in e])
    assert abs(loss-exact)<1e-12
    for i in range(2):
        eps=np.zeros(2); eps[i]=1e-5
        numeric=(gaussian_objective(theta+eps,rec,tau)[0]-gaussian_objective(theta-eps,rec,tau)[0])/2e-5
        assert abs(numeric-grad[i])<1e-7


def test_gaussian_fit_recovers_synthetic_field_and_nugget_scales():
    cov=np.ones((3,3)); rng=np.random.default_rng(42)
    e=.7*rng.normal(size=(12000,1))+.9*rng.normal(size=(12000,3))
    rec=gaussian_records(e,np.ones_like(e,bool),np.ones(len(e)),cov,np.zeros(3,int))
    fit=fit_gaussian(rec,1.,'two_scales')
    np.testing.assert_allclose([fit['field'],fit['nugget']],[.7,.9],atol=.025)


def test_gaussian_missing_patterns_use_submatrix_and_nan_predictions_fail():
    e=np.array([[1.,np.nan,2.],[2.,3.,4.]])
    mask=np.array([[1,0,1],[1,1,1]],bool)
    records=gaussian_records(e,mask,np.ones(2),np.eye(3),np.zeros(3,int))
    loss,_=gaussian_objective(np.zeros(2),records,1.)
    assert np.isfinite(loss)
    e[1,1]=np.nan
    with pytest.raises(ValueError): gaussian_records(e,mask,np.ones(2),np.eye(3),np.zeros(3,int))


def test_bootstrap_reweights_same_group_objective_instead_of_changing_groups():
    from local_innovation_report import bootstrap_records
    error=np.arange(24,dtype=float).reshape(4,6)/3
    mask=np.ones_like(error,bool); groups=np.array([0,0,0,1,1,1])
    cov=np.eye(6); amp=np.ones(4)
    original=gaussian_records(error,mask,amp,cov,groups)
    combined=bootstrap_records(error,mask,amp,cov,groups,{0:1,1:1})
    x=np.log([.6,.9])
    before=gaussian_objective(x,original,1.); after=gaussian_objective(x,combined,1.)
    assert abs(before[0]-after[0])<1e-12
    np.testing.assert_allclose(before[1],after[1],rtol=1e-12)
    doubled=bootstrap_records(error,mask,amp,cov,groups,{0:2,1:0})
    alone=gaussian_records(error[:,:3],mask[:,:3],amp,cov[:3,:3],np.zeros(3,int))
    assert abs(gaussian_objective(x,doubled,1.)[0]-gaussian_objective(x,alone,1.)[0])<1e-12


def test_bootstrap_optimizer_verifies_stationarity_without_silent_drop():
    import json
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/gaussian_scale_stationary_case.json').read_text())
    for r in fixture['records']:
        r['eigen']=np.array(r['eigen']);r['squared']=np.array(r['squared'])
    fit=fit_gaussian(fixture['records'],fixture['tau'],'two_scales',gradient_tolerance=1e-7)
    assert fit['success'] and fit['gradient_norm']<1e-7
    np.testing.assert_allclose([fit['field'],fit['nugget']],[.7044,.8455],atol=.002)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual local GPU resume path')
def test_real_training_runner_100_plus_100_matches_200(tmp_path,monkeypatch):
    import local_innovation_pilot as runner
    from gnss_field.training.observable import _same
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'experiment.lock.json').write_text('{}')
    rng=np.random.default_rng(42)
    d=object.__new__(runner.Pilot)
    d.a=dict(hours=np.arange(8),truth=rng.normal(size=(8,4)).astype('f4'),valid=np.ones((8,4),bool))
    d.settings={'grid_scale':5.}
    d.t={k:torch.tensor(v,device='cuda') for k,v in dict(features=rng.normal(size=(8,4,67)).astype('f4'),
        innovation=rng.normal(size=(8,4,2)).astype('f4'),query_indices=np.arange(4)[:,None],
        query_weights=np.ones((4,1),'f4'),mean=np.zeros((8,4),'f4'),truth=d.a['truth'],valid=d.a['valid']).items()}
    d.splits=[dict(train_hours=np.arange(8),inner_hours=np.arange(8),fit=np.array([1,1,0,0],bool),inner=np.array([0,0,1,1],bool))]
    batches=[np.arange(8)]*200
    whole=runner.run_neural(d,'gated',0,tmp_path/'whole',batches,engineering=True)
    runner.run_neural(d,'gated',0,tmp_path/'split',batches,stop_after=100,engineering=True)
    split=runner.run_neural(d,'gated',0,tmp_path/'split',batches,engineering=True)
    for key in ('step','model','ema','optimizer','rng'): _same(whole[key],split[key])
    (tmp_path/'PAUSE').touch()
    paused=runner.run_neural(d,'gated',0,tmp_path/'paused',batches,engineering=True)
    assert paused['step']==0
    import json
    assert json.loads((tmp_path/'paused/status.json').read_text())['status']=='PAUSED'
