"""Temporal separation and feature/publication contracts for expanded local training."""
import numpy as np
import pytest
from local_coverage_data import coverage_schedule, compact67


def test_schedule_uses_full_calendar_for_holdouts_and_preserves_training_allowlist():
    begin=np.datetime64('2018-01-01','h').astype('i8')
    end=np.datetime64('2019-01-01','h').astype('i8')
    hours=np.arange(begin,end)
    outer=[int(np.datetime64(f'2018-{m:02d}-15','h').astype('i8')) for m in range(1,13)]
    permitted=np.ones(len(hours),bool)
    for s in outer:permitted&=(hours<s-18)|(hours>=s+168+18)
    # Old exploratory split removed early February and November. New validation
    # can use those B2018 calendar blocks, but training stays in the allowlist.
    for m in (2,11):
        s=np.datetime64(f'2018-{m:02d}-01','h').astype('i8')
        permitted&=(hours<s)|(hours>=s+240)
    split=coverage_schedule(hours,hours[permitted],outer)
    assert len(split['train_hours'])==576 and len(split['inner_hours'])==48
    assert len(set(split['train_hours']))==576 and len(split['inner_week_starts'])==12
    assert np.isin(split['train_hours'],hours[permitted]).all()
    for s in split['inner_week_starts']:
        assert not np.any((split['train_hours']>=s-18)&(split['train_hours']<s+168+18))
    for s in outer:
        assert not np.any((split['inner_hours']>=s-18)&(split['inner_hours']<s+168+18))
    months=split['train_hours'].astype('datetime64[h]').astype('datetime64[M]')
    np.testing.assert_array_equal(np.unique(months,return_counts=True)[1],np.full(12,48))


def test_schedule_fails_instead_of_silently_reducing_month_coverage():
    h=np.arange(np.datetime64('2018-01-01','h').astype('i8'),np.datetime64('2019-01-01','h').astype('i8'))
    with pytest.raises(ValueError):coverage_schedule(h,h[:100],[])


def test_compact_features_equal_full_grid_before_nonlinearity_even_with_missing_A():
    from local_context_features import LocalFeatureBuilder
    from grid_pair_data import feature69
    shape=(4,4);yy,xx=np.indices(shape);xy=np.c_[xx.ravel(),yy.ravel()]*25000.
    aq=np.array([[.5,.5],[1.5,1.5],[2.5,2.5]])
    axy=aq*25000;terrain=(xx+yy)*20.;land=np.ones(shape,bool)
    stats=dict(input_mean=np.zeros(5),input_std=np.ones(5),z0_std=50.)
    nodes=np.array([0,1,4,5,10,11,14,15])
    args=(xy,axy,np.array([10.,20.,30.]),terrain,land)
    full=LocalFeatureBuilder(*args,np.arange(16),stats,aq,np.ones(3,bool))
    compact=LocalFeatureBuilder(*args,nodes,stats,aq,np.ones(3,bool))
    frames=np.arange(60,dtype='f4').reshape(4,3,5)/10;frames[:,1]=np.nan
    m0=np.ones(shape,dtype='f4')*3;cycle=np.array([.1,.2,.3,.4])
    expected=feature69(full,frames,m0,cycle,shape)[:67].reshape(67,-1).T[nodes]
    actual=compact67(compact,frames,m0,cycle)
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)


def test_corrector_affine_readout_is_applied_at_vertices_with_height_scaled_bias():
    import torch
    from local_coverage_train import Corrector
    model=Corrector(torch.zeros(69),torch.ones(69),scale=10.)
    x=torch.randn(2,4,69)
    assert torch.equal(model(x),torch.zeros(2,4))
    beta=np.zeros(33);beta[-1]=.2;model.set_readout(beta)
    qw=torch.tensor([[.125,.125,.125,.125]])  # d=.5 for four equal geometric weights
    pred=(model(x)[:,torch.tensor([[0,1,2,3]])]*qw).sum(-1)
    torch.testing.assert_close(pred,torch.ones(2,1))
    pred.sum().backward()
    assert model.output.bias.grad.item()==10.


def test_coverage_model_normalizer_is_checkpointed_and_all_predictions_finite():
    import torch
    from local_coverage_train import Corrector
    center=torch.arange(69,dtype=torch.float32)/10;std=torch.ones(69)*2
    a=Corrector(center,std,scale=5.)
    beta=np.ones(33)*.01;a.set_readout(beta)
    b=Corrector(torch.zeros(69),torch.ones(69),scale=5.);b.load_state_dict(a.state_dict())
    x=torch.randn(3,8,69)
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)


@pytest.mark.parametrize('interrupted_step',[150,200])
def test_engineering_can_resume_second_segment(tmp_path,monkeypatch,interrupted_step):
    import torch
    import local_coverage_train as runner
    from gnss_field.contracts import ScienceConfig
    from gnss_field.training.engine import TrainEngine,StageSpec,atomic_torch
    def factory(data,fold,directory,budget):
        torch.manual_seed(42)
        model=torch.nn.Linear(1,1)
        science=ScienceConfig(learning_rate=1e-3,minimum_lr=1e-3,warmup=0,log_every=50)
        engine=TrainEngine(model,directory,np.arange(16),StageSpec('engineering',budget,budget,8,8,42,{}),science)
        def objective(hours,model,step):
            x=torch.tensor(hours[:,None],dtype=torch.float32)/16
            return {'loss':((model(x)-x)**2).mean()}
        return engine,objective,None
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'make_engine',factory)
    engine,obj,_=factory(None,0,tmp_path/'engineering/split',200)
    engine.run(obj,stop_after=100)
    atomic_torch(tmp_path/'engineering/split_100.pt',engine.state())
    engine,obj,_=factory(None,0,tmp_path/'engineering/split',200)
    engine.run(obj,stop_after=interrupted_step)
    try:runner.engineering(None)
    except runner.PauseRequested as exc:pytest.fail(str(exc))
    assert runner.read_json(tmp_path/'engineering/verification.json')['bitwise']


def test_preparation_provenance_rejects_changed_helper_and_packed_identity(tmp_path):
    from local_coverage_provenance import dependency_snapshot,lock_dependencies
    from gnss_field.artifacts import write_json
    project=tmp_path/'project';(project/'tools').mkdir(parents=True)
    (project/'src/gnss_field').mkdir(parents=True)
    (project/'src/gnss_field/__init__.py').write_text('')
    helper=project/'tools/local_context_features.py';helper.write_text('VERSION=1')
    packed=tmp_path/'packed';packed.mkdir();write_json(packed/'manifest.json',{'fingerprint':'one'})
    root=tmp_path/'newrun';root.mkdir()
    original=dependency_snapshot(project,{'data_root':packed})
    lock_dependencies(root,original);lock_dependencies(root,original)
    helper.write_text('VERSION=2')
    with pytest.raises(ValueError,match='Immutable artifact differs'):
        lock_dependencies(root,dependency_snapshot(project,{'data_root':packed}))
    helper.write_text('VERSION=1');write_json(packed/'manifest.json',{'fingerprint':'two'})
    with pytest.raises(ValueError,match='Immutable artifact differs'):
        lock_dependencies(root,dependency_snapshot(project,{'data_root':packed}))
    legacy=tmp_path/'legacy';legacy.mkdir();write_json(legacy/'experiment.lock.json',{})
    with pytest.raises(ValueError,match='legacy'):
        lock_dependencies(legacy,original)


def test_pair_diagnostics_use_same_population_and_restore_common_height():
    from local_coverage_report import pair_diagnostics
    xy=np.c_[[0.,10000.,40000.,150000.],np.zeros(4)]
    height=np.array([0.,80.,300.,80.]);truth=np.zeros((2,4))
    mask=np.ones((2,4),bool);mask[1,1]=False
    prediction=np.broadcast_to(5*np.exp(-height/3000),(2,4))
    result=pair_diagnostics(truth,{'base':truth,'constant_Z0':prediction},mask,xy,height)
    short=result['0-25']['common_height']
    assert short['n_pairs']==1 and short['n_pair_hours']==1
    assert short['methods']['constant_Z0']['pair_macro_rmse_mm']<1e-12
    assert result['0-25']['same_height_surface']['methods']['constant_Z0']['pair_macro_rmse_mm']>0
    assert result['25-50']['common_height']['n_pairs']==2
    assert result['25-50']['same_height_surface']['n_pairs']==0
    assert result['50-100']['common_height']['methods']['base']['pair_macro_rmse_mm'] is None
    bad=prediction.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        pair_diagnostics(truth,{'bad':bad},mask,xy,height)
