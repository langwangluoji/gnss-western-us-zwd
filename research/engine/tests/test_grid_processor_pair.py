import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


@pytest.mark.parametrize('arm', ['mlp69', 'resunet69'])
def test_zero_initial_field_query_and_all_missing_gate(arm):
    from gnss_field.models.grid_processor import GridProcessor, query_field
    torch.manual_seed(42)
    model = GridProcessor(arm)
    x = torch.randn(2, 69, 24, 16)
    field = model(x)
    assert field.shape == (2, 1, 24, 16) and not field.any()
    with torch.no_grad():
        model.output.bias.fill_(2.)
    field = model(x, torch.tensor([True, False]))
    indices = torch.tensor([[0, 1, 16, 17]])
    weights = torch.tensor([[.25]*4])*np.exp(-1.)
    values = query_field(field, indices, weights)
    torch.testing.assert_close(values[:, 0], torch.tensor([2*np.exp(-1.), 0.], dtype=torch.float32))
    assert torch.isfinite(values).all()


def test_unet_has_spatial_receptive_field_and_mlp_is_pointwise():
    from gnss_field.models.grid_processor import GridProcessor
    for arm in ['mlp69', 'resunet69']:
        torch.manual_seed(42)
        m = GridProcessor(arm)
        torch.nn.init.normal_(m.output.weight, std=.03)
        x = torch.randn(1, 69, 24, 16, requires_grad=True)
        g = torch.autograd.grad(m(x)[0, 0, 12, 8], x)[0]
        outside = g.clone()
        outside[:, :, 12, 8] = 0
        assert bool(outside.abs().sum()>0) == (arm == 'resunet69')
        assert torch.isfinite(g).all()


def test_same_schema_four_frames_missing_and_station_permutation():
    from grid_pair_data import feature69
    from local_context_features import LocalFeatureBuilder
    xx, yy = np.meshgrid(np.arange(3)*25000., np.arange(3)*25000.)
    gridxy = np.c_[xx.ravel(), yy.ravel()]
    args = dict(terrain=np.full((3,3),100.), land=np.ones((3,3), bool), nodes=np.arange(9),
                stats=dict(input_mean=[0.]*5, input_std=[1.]*5, z0_std=10.))
    frames = np.full((4, 2, 5), 20.)
    frames[0, 1, 2] = np.nan
    builder = LocalFeatureBuilder(gridxy, gridxy[[0,4]], [100., 200.],
        **args, a_qxy=np.array([[0.,0.],[1.,1.]]), a_query_valid=[True,True])
    field = np.full((3,3),10.)
    result = feature69(builder, frames, field, np.ones(4), (3,3))
    assert result.shape == (69,3,3) and np.isfinite(result).all()
    np.testing.assert_allclose(result[48], 1.)
    np.testing.assert_allclose(result[67,0], [-1,0,1])
    other = LocalFeatureBuilder(gridxy, gridxy[[4,0]], [200.,100.],
        **args, a_qxy=np.array([[1.,1.],[0.,0.]]), a_query_valid=[True,True])
    np.testing.assert_allclose(feature69(other, frames[:,::-1], field, np.ones(4),(3,3)), result, atol=1e-6)
    empty = feature69(builder, frames*np.nan, field, np.ones(4), (3,3))
    assert np.isfinite(empty).all() and not empty[:52].any()


def test_full_pool_split_excludes_whole_weeks_and_history_buffer():
    from grid_pair_data import exclude_weeks
    hours = np.arange(1000,2000)
    train = exclude_weeks(hours, [1200,1600], 18)
    assert not np.any((train>=1182)&(train<1386))
    assert not np.any((train>=1582)&(train<1786))
    assert len(train)==592
    start=np.datetime_as_string(np.datetime64('1970-01-01T00','h')+np.timedelta64(1200,'h'))
    np.testing.assert_array_equal(exclude_weeks(hours,[start]),exclude_weeks(hours,[1200]))


def test_grid_query_matches_publication_operator_and_gradients():
    from gnss_field.models.grid_processor import query_field
    from gnss_field.geometry import surface_query
    from local_context_features import VertexQuery
    xy, heights = np.array([[.2,.7],[1.3,1.4]]), np.array([300.,1300.])
    q = VertexQuery((3,4), xy, heights, np.arange(12))
    x = torch.randn(2,1,3,4,requires_grad=True)
    actual=query_field(x,q.indices,q.weights)
    expected=surface_query(x,xy,heights)[:,0]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad!=0).sum()>0
    with pytest.raises(ValueError):
        query_field(x*float('nan'),q.indices,q.weights)


def test_pair_schedule_respects_real_hours_and_zero_startup_exclusion():
    from grid_pair_train import scientific_settings
    s, updates = scientific_settings(5472)
    assert updates==2736 and s.startup_evaluation==200
    assert s.evaluate_every==342 and s.minimum_refinement_steps==1368
    assert s.ema==.99 and s.patience==3
    with pytest.raises(ValueError):
        scientific_settings(5473)


def test_monitor_keeps_ES_selection_and_labels_rmse_as_diagnostic():
    from grid_pair_monitor import monitor_payload
    p = dict(runs=[dict(name='fold_0/resunet69',status={},training=[],
        development=[dict(step=200,startup=True,metrics={'joint':dict(value=5.,rmse_mm=8.)})])])
    q=monitor_payload(p)
    assert q['runs'][0]['development'][0]['metrics']['point']['value']==8.
    assert 'point' not in p['runs'][0]['development'][0]['metrics']


def test_release_rejects_mutated_or_missing_dependencies(tmp_path):
    from grid_pair_release import verify_grid_release
    from gnss_field.artifacts import write_json,sha256
    f=tmp_path/'source.py'; f.write_text('checked')
    write_json(tmp_path/'grid_release.json',dict(files={'source.py':sha256(f)}))
    assert verify_grid_release(tmp_path)
    f.write_text('modified')
    with pytest.raises(ValueError):
        verify_grid_release(tmp_path)


def test_bundle_validation_rejects_label_corruption_and_role_mismatch(tmp_path):
    from grid_pair_data import verify_bundle
    from gnss_field.artifacts import write_json,sha256
    f=tmp_path/'data.npz'; np.savez(f,station_ids=[1,2],hours=[10])
    m=dict(data_sha256=sha256(f),frozen_contract={'foundation':{'data':'original'}})
    write_json(tmp_path/'manifest.json',m)
    verify_bundle(tmp_path,'original',np.array([1,2]))
    with pytest.raises(ValueError):
        verify_bundle(tmp_path,'other',np.array([1,2]))
    with pytest.raises(ValueError):
        verify_bundle(tmp_path,'original',np.array([2,1]))
    f.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_bundle(tmp_path,'original',np.array([1,2]))


def test_budget_includes_engineering_and_full_outer_evaluations():
    from grid_pair_train import planned_seconds
    total=planned_seconds(.1,.2,10.)
    expected=(10944+410)*.1+8760*.2+(4*10+4*2016/96)*10.
    assert total==pytest.approx(expected)


def test_pause_during_evaluation_keeps_pending_work(tmp_path):
    from grid_pair_train import check_evaluation_pause
    from gnss_field.training.engine import PauseRequested
    data=type('Data',(),dict(root=tmp_path,deadline=None))()
    check_evaluation_pause(data)
    (tmp_path/'PAUSE').touch()
    with pytest.raises(PauseRequested):
        check_evaluation_pause(data)


def test_monitor_reads_last_complete_remote_snapshot(tmp_path):
    import json
    from grid_pair_monitor import read_remote_status
    p=tmp_path/'ssh.log'
    snapshot=dict(generated_at=10,runs=[],full2019=[],campaign_queue={'state':'PENDING'})
    p.write_text('Other SSH output\nGRID_SNAPSHOT '+json.dumps(snapshot)+'\nGRID_SNAPSHOT {"partial"')
    result=read_remote_status(p)
    assert result['campaign_queue']['state']=='PENDING'
    assert result['ssh_sync']['last_success']==10


def test_cache_rows_reject_mutation_and_missing_committed_component(tmp_path):
    from grid_pair_data import row_digest,verify_row,GridStudy
    a=[np.ones((2,3),dtype='f4'),np.ones(3,dtype='f4'),np.zeros(4,dtype='f4')]
    digest=row_digest(a)
    verify_row(a,digest)
    a[0][0,0]=2
    with pytest.raises(ValueError): verify_row(a,digest)
    with pytest.raises(ValueError): GridStudy._memmap(tmp_path/'missing.npy',(2,3),required=True)


def test_completed_benchmark_is_reused_without_retiming(tmp_path):
    from grid_pair_train import cached_benchmark
    from gnss_field.artifacts import write_json
    d=type('Data',(),dict(root=tmp_path))()
    write_json(tmp_path/'benchmark.json',dict(contract='locked',updates=10,seconds=1.))
    assert cached_benchmark(d,'locked')['seconds']==1.
    with pytest.raises(ValueError): cached_benchmark(d,'changed')


def test_allocation_accounting_caps_resume_and_refuses_unknown_charge():
    from grid_pair_submit import remaining_allocation
    attempts=[{'job_id':'1'},{'job_id':'2'}]
    assert remaining_allocation(attempts,{'1':{'elapsed':200},'2':{'elapsed':300}},600)==100
    with pytest.raises(ValueError): remaining_allocation(attempts,{'1':{'elapsed':200}},600)


def test_status_distinguishes_optimizer_end_outer_evaluation_and_slurm_failure():
    from grid_pair_monitor import stage_status
    assert stage_status('COMPLETE',False,'RUNNING')=='OUTER_EVALUATION'
    assert stage_status('PAUSED',False,'COMPLETED')=='PAUSED'
    assert stage_status('RUNNING',False,'FAILED')=='FAILED'


def test_generated_page_contains_actual_startup_element_and_script():
    from grid_pair_monitor import build_page
    page=build_page()
    assert 'id="startup_check"' in page and "getElementById('startup_check')" in page
