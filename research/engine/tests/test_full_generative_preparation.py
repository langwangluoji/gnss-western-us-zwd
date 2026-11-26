import numpy as np
import pytest
from types import SimpleNamespace


def hour(text):
    return int(np.datetime64(text, 'h').astype('i8'))


def test_month_plan_exclusive_ownership_and_role_boundaries():
    from gnss_field.generative.preparation import preparation_plan
    plan = preparation_plan([hour('2016-01-02'), hour('2016-01-01'), hour('2016-02-01')],
                            [hour('2019-01-01')])
    assert [(p['role'], p['month']) for p in plan] == [('B','2016-01'),('B','2016-02'),('C','2019-01')]
    assert len({p['month'] for p in plan}) == len(plan)
    assert plan[0]['hours'] == sorted(plan[0]['hours'])
    with pytest.raises(PermissionError):
        preparation_plan([hour('2020-01-01')], [])
    with pytest.raises(PermissionError):
        preparation_plan([], [hour('2018-01-01')])


def test_dispatch_fail_closed_and_never_aggregates_partial_month(tmp_path):
    from gnss_field.generative.preparation import prepare_parallel
    class Data:
        identity = 'condition'; metadata = {}
        cache_root = tmp_path
        config = SimpleNamespace(cache_root=str(tmp_path))
        train_hours = [hour('2016-01-01')]
        development_hours = [hour('2019-01-01')]
        def audit(self,role): return {'hour_ids':self.development_hours}
        def _label_identity(self, role): return role+'labels'
        def _other_view(self, purpose): return self
        def prepare(self, *args, **kwargs): pytest.fail('must not aggregate failed workers')
    def fail(plan, **kwargs):
        raise RuntimeError('worker died')
    with pytest.raises(RuntimeError, match='worker died'):
        prepare_parallel({'runtime':{'device':'cuda:0'}}, data=Data(), _dispatch=fail)
    assert not (tmp_path/'preparation_parallel.json').exists()


def test_parent_aggregates_both_complete_populations_and_bounds_workers(tmp_path):
    from gnss_field.generative.preparation import prepare_parallel
    seen = []
    class Data:
        identity = 'condition'; metadata = {}; cache_root = tmp_path
        config = SimpleNamespace(cache_root=str(tmp_path))
        train_hours = [hour('2016-01-01'),hour('2016-02-01')]
        development_hours = [hour('2019-01-01')]
        def audit(self,role): return {'hour_ids':self.development_hours}
        def _label_identity(self, role): return role+'labels'
        def _other_view(self, purpose): return self
        def _clear_cache(self): pass
        def prepare(self, hours, role):
            seen.append((role,list(hours))); return {'prepared_hours':len(hours)}
    def dispatch(plan, **kwargs):
        assert kwargs['workers'] == 4
        return [{**p, 'condition_identity':'condition','labels_identity':p['role']+'labels','resolved_cache_root':str(tmp_path.resolve()),
                 'worker_pid':7 if p['role']=='B' else 8,
                 'cuda_peak_allocated_bytes':10 if p['role']=='B' else 30,
                 'cuda_peak_reserved_bytes':20 if p['role']=='B' else 40} for p in plan]
    result = prepare_parallel({'runtime':{'device':'cuda:0'}}, data=Data(), workers=4, _dispatch=dispatch)
    assert [x[0] for x in seen] == ['B','C']
    assert result['prepared_hours'] == 3
    assert result['workers'] == 4
    assert result['worker_peak_allocated_sum_bytes'] == 40
    assert result['worker_peak_reserved_sum_bytes'] == 60
    for workers in (0,5,True):
        with pytest.raises(ValueError):
            prepare_parallel({'runtime':{'device':'cuda:0'}}, data=Data(), workers=workers, _dispatch=dispatch)

def test_owner_mismatch_rejected_before_aggregate(tmp_path):
    from gnss_field.generative.preparation import prepare_parallel
    class Data:
        identity='same'; metadata={}; cache_root=tmp_path
        config=SimpleNamespace(cache_root=str(tmp_path))
        train_hours=[hour('2016-01-01')]; development_hours=[]
        def _other_view(self,purpose): return self
        def audit(self,role): return {'hour_ids':self.development_hours}
        def _label_identity(self,role): return role
        def prepare(self,*args,**kwargs): pytest.fail('invalid identity must not aggregate')
    def wrong(plan,**kwargs):
        return [{**plan[0], 'condition_identity':'wrong','labels_identity':'B','resolved_cache_root':str(tmp_path.resolve())}]
    with pytest.raises(RuntimeError,match='ownership or source identity'):
        prepare_parallel({'runtime':{'device':'cuda:0'}},data=Data(),_dispatch=wrong)


def test_dead_process_aborts_spawn_pool_without_completion(monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from gnss_field.generative import preparation as p
    seen=[]
    class Process:
        def is_alive(self): return True
        def terminate(self): seen.append('terminated')
    class Pool:
        _processes={1:Process()}
        def __init__(self,**kwargs):
            assert kwargs['max_workers']==4
            assert kwargs['mp_context'].get_start_method()=='spawn'
        def submit(self,*args):
            f=Future(); f.set_exception(BrokenProcessPool('worker died')); return f
        def shutdown(self,**kwargs): seen.append('shutdown')
    monkeypatch.setattr(p,'ProcessPoolExecutor',Pool)
    with pytest.raises(BrokenProcessPool):
        p._dispatch_months([{}],config={},workers=4,timeout_seconds=1)
    assert seen==['terminated','shutdown']


def test_worker_uses_injected_config_root_and_complete_C_calendar(tmp_path):
    from gnss_field.generative.preparation import prepare_parallel
    requested = []
    class Data:
        identity='condition'; metadata={}
        config=SimpleNamespace(cache_root=str(tmp_path/'benchmark_cache'))
        cache_root=tmp_path/'benchmark_cache'/'condition_identity'
        train_hours=[hour('2016-01-01')]
        development_hours=[hour('2019-01-01')]
        def audit(self,role): return {'hour_ids':[hour('2019-01-01'),hour('2019-01-02')]}
        def _other_view(self,purpose): return self
        def _label_identity(self,role): return role
        def _clear_cache(self): pass
        def prepare(self,hours,role):
            requested.append((role,list(hours)));return {}
    config={'runtime':{'device':'cuda:0'}, 'data':{'cache_root':'formal_cache',
             'background_source':{'frozen':'assets'}}}
    def dispatch(plan,**kwargs):
        assert kwargs['config']['data']['cache_root']==str(Data.config.cache_root)
        assert kwargs['config']['data']['background_source']=={'frozen':'assets'}
        return [{**p,'condition_identity':'condition','labels_identity':p['role'],
                 'resolved_cache_root':str(Data.cache_root.resolve())} for p in plan]
    prepare_parallel(config,data=Data(),_dispatch=dispatch)
    assert config['data']['cache_root']=='formal_cache'
    assert requested[-1]==('C',[hour('2019-01-01'),hour('2019-01-02')])


def test_worker_resolved_cache_mismatch_rejected(tmp_path):
    from gnss_field.generative.preparation import prepare_parallel
    class Data:
        identity='condition'; metadata={}; cache_root=tmp_path/'correct'
        config=SimpleNamespace(cache_root=str(tmp_path/'correct'))
        train_hours=[hour('2016-01-01')]; development_hours=[]
        def audit(self,role): return {'hour_ids':[]}
        def _other_view(self,purpose): return self
        def _label_identity(self,role): return role
        def prepare(self,*args,**kwargs): pytest.fail('wrong cache must not aggregate')
    def dispatch(plan,**kwargs):
        return [{**p,'condition_identity':'condition','labels_identity':p['role'],
                 'resolved_cache_root':str(tmp_path/'wrong')} for p in plan]
    with pytest.raises(RuntimeError,match='ownership or source identity'):
        prepare_parallel({'runtime':{'device':'cuda:0'}},data=Data(),_dispatch=dispatch)


def test_timeout_finishes_running_month_without_forced_termination(monkeypatch):
    from concurrent.futures import Future
    from gnss_field.generative import preparation as p
    seen=[]
    class Process:
        def is_alive(self): return True
        def terminate(self): pytest.fail('timeout must allow active month boundary')
    class Pool:
        _processes={1:Process()}
        def __init__(self,**kwargs): pass
        def submit(self,*args): return Future()
        def shutdown(self,**kwargs): seen.append(kwargs)
    monkeypatch.setattr(p,'ProcessPoolExecutor',Pool)
    with pytest.raises(p.PreparationPaused):
        p._dispatch_months([{}],config={},workers=4,timeout_seconds=.001)
    assert seen==[{'wait':True,'cancel_futures':True}]


def test_parent_signal_is_cooperative_and_prior_handler_restored(tmp_path,monkeypatch):
    from gnss_field.generative import preparation as p
    handlers=[]
    previous=object()
    monkeypatch.setattr(p.signal,'SIGUSR1',123456,raising=False)
    def install(signum,handler):
        handlers.append(handler); return previous
    monkeypatch.setattr(p.signal,'signal',install)
    class Data:
        identity='condition'; metadata={}; cache_root=tmp_path
        config=SimpleNamespace(cache_root=str(tmp_path))
        train_hours=[hour('2016-01-01')]
        def audit(self,role): return {'hour_ids':[]}
        def _other_view(self,purpose): return self
        def prepare(self,*args,**kwargs): pytest.fail('signal must not publish aggregate')
    def dispatch(plan,**kwargs):
        assert not kwargs['stop_requested']()
        handlers[0](123456,None)
        assert kwargs['stop_requested']()
        raise p.PreparationPaused('signal requested pause')
    with pytest.raises(p.PreparationPaused):
        p.prepare_parallel({'runtime':{'device':'cuda:0'}},data=Data(),_dispatch=dispatch)
    assert handlers[-1] is previous

def test_parallel_preparation_prices_real_interval_union_and_startup_once():
    from gnss_field.generative.preparation import interval_union_seconds, price_preparation
    assert interval_union_seconds([(1,5),(2,3),(4,8),(10,12)])==9
    assert interval_union_seconds([])==0
    with pytest.raises(ValueError): interval_union_seconds([(2,1)])
    report=dict(startup_bound_seconds=20.,task_union_seconds=12.,verification_seconds=3.,
                missing_condition_hours=128,prepared_hours=140)
    costs=price_preparation(report,total_hours=35022)
    assert costs['preparation_seconds']==pytest.approx(35022*(12/128+3/140))
    assert costs['preparation_startup_seconds']==20
    assert costs['preparation_total_seconds']==pytest.approx(20+costs['preparation_seconds'])


def test_worker_initial_warmup_builds_in_memory_without_writing_shards(tmp_path,monkeypatch):
    import torch
    from gnss_field.generative import preparation as p
    from gnss_field.generative import campaign
    events=[]
    warm=hour('2016-01-02')
    class Data:
        def _other_view(self,purpose):
            events.append(('view',purpose)); return self
        def _build_compact(self,hours):
            events.append(('warm',hours.tolist()))
            return {'background':np.zeros((1,1,4,4))}
        def prepare(self,*args,**kwargs): pytest.fail('warmup must not write cache shards')
    def build(config,**kwargs):
        assert kwargs['allow_background_inference'] is True
        events.append(('build',None));return Data()
    monkeypatch.setattr(campaign,'build_data',build)
    monkeypatch.setattr(p.signal,'signal',lambda *args:None)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(p,'_WORKER_VIEWS',None)
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda value:None)
    for name in ('benchmark','deterministic','allow_tf32'):
        monkeypatch.setattr(torch.backends.cudnn,name,getattr(torch.backends.cudnn,name))
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    config={'runtime':{'device':'cuda:0','preparation_warmup_hour':warm,'preparation_has_development':True}}
    p._initialize_worker(config)
    assert events==[('build',None),('view','development'),('warm',[warm])]
    assert not list(tmp_path.iterdir())
    assert p._WORKER_STARTUP['finished']>=p._WORKER_STARTUP['started']
    assert p._WORKER_STARTUP_SECONDS==pytest.approx(p._WORKER_STARTUP['finished']-p._WORKER_STARTUP['started'])

def test_worker_initialization_applies_formal_torch_numerical_contract_before_build(monkeypatch):
    import os
    import torch
    from gnss_field.generative import preparation as p
    from gnss_field.generative import campaign
    called=[]
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda value:called.append(value))
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',True)
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',False)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',True)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',True)
    monkeypatch.setattr(p.signal,'signal',lambda *args:None)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    class Data:
        def _build_compact(self,hours): return {}
    def build(config,**kwargs):
        assert called==[True]
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        return Data()
    monkeypatch.setattr(campaign,'build_data',build)
    p._initialize_worker({'runtime':{'preparation_warmup_hour':hour('2016-01-02')}})
