"""Training-only diagnostic contracts; no real-data or GPU jobs."""
import importlib
import numpy as np
import pandas as pd
import pytest
import torch


def module():
    spec = importlib.util.find_spec('gnss_field.generative.diagnostics')
    assert spec is not None, 'training-only diagnostics module is required'
    return importlib.import_module('gnss_field.generative.diagnostics')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_calendar_panel_is_fixed_before_labels_and_has_weather_separation():
    hours = np.arange(np.datetime64('2016-01-01', 'h').astype('i8'),
                      np.datetime64('2019-01-01', 'h').astype('i8'))
    panel = module().calendar_panels(hours)
    assert len(panel['fit_hours']) == 1728
    assert len(panel['validation_hours']) == 864
    assert min(abs(a-b) for a in panel['fit_hours'] for b in panel['validation_hours']) > 18
    reduced = module().calendar_panels(hours[hours != panel['fit_hours'][0]])
    assert len(reduced['fit_hours']) == 1727
    assert reduced['validation_hours'] == panel['validation_hours']


def test_connected_alias_and_dense_components_stay_whole_and_seeded():
    # The first three stations connect transitively through physical then dense IDs.
    meta = pd.DataFrame(dict(physical_group=['p','p','q','r','s','t','u','v','w'],
                             dense_group=['a','b','b','c','d','e','f','g','h']))
    folds = module().physical_group_folds(meta)
    assert folds[0] == folds[1] == folds[2]
    assert set(folds) == {0,1,2}
    assert np.array_equal(folds, module().physical_group_folds(meta))
    assert max(np.bincount(folds))-min(np.bincount(folds)) <= 2


class TinyData:
    train_hours = np.arange(np.datetime64('2016-01-02', 'h').astype('i8'),
                            np.datetime64('2016-01-03', 'h').astype('i8'))
    development_hours = []
    metadata = {'synthetic': True}
    station_metadata = pd.DataFrame(dict(physical_group=list('abcdef'), dense_group=list('abcdef')))

    def batch(self, hours, role='B', device='cpu'):
        assert role == 'B', 'No C/D diagnostic label access'
        b, n = len(hours), 6
        indices = torch.arange(n, device=device)[:,None].expand(-1,4)
        return dict(hour_ids=torch.as_tensor(hours, device=device),
                    background=torch.zeros(b,1,4,4,device=device),
                    query_indices=indices, query_weights=torch.full((n,4), .25, device=device),
                    target_geometry=torch.zeros(n,3,device=device),
                    target=torch.full((b,n),2.,device=device),
                    target_mask=torch.ones(b,n,dtype=torch.bool,device=device))


def test_fit_view_removes_held_targets_everywhere_and_keeps_role_B():
    d = module(); data = TinyData()
    view = d.BTrainingView(data, data.train_hours, np.array([0,0,1,1,2,2]), 1)
    batch = view.batch(data.train_hours[:1])
    assert not batch['target_mask'][0,2:4].any()
    assert torch.equal(batch['target'][0,2:4], torch.zeros(2))
    assert view.metadata['diagnostic_role'] == 'B_training_internal'
    with pytest.raises(PermissionError): view.batch(data.train_hours[:1],role='C')


class FunctionalVAE(torch.nn.Module):
    def __init__(self, collapsed=False):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0. if collapsed else 1.))
    def prior_parameters(self, encoded):
        return torch.zeros(len(encoded),2), torch.zeros(len(encoded),2)
    def posterior_parameters(self, encoded, residual, indices, weights, geometry, mask):
        # q=p exactly: this must not by itself fail a functional latent path.
        return self.prior_parameters(encoded)
    def decoder(self, encoded, z, background):
        return background[:,None]+self.weight*z[:,:,:1,None,None]


class FunctionalModel(torch.nn.Module):
    family='vae'; deterministic=False; latent_dim=2
    def __init__(self,collapsed=False):
        super().__init__(); self.vae=FunctionalVAE(collapsed)
    def encode(self,batch): return batch['hour_ids']


def test_prior_scoring_uses_only_prior_and_independent_streams():
    d=module(); data=TinyData(); model=FunctionalModel()
    def forbidden(*args): raise AssertionError('prior scoring cannot use posterior q')
    model.vae.posterior_parameters=forbidden
    scores=d.score_B_prior(model,data,data.train_hours[:2],np.array([1,1,0,0,0,0],bool),
                           sigma_eff_mm=3.18, device='cpu')
    assert scores['role']=='B_training_diagnostic'
    assert scores['probability_members']==16 and scores['point_members']==32
    assert scores['station_macro_crps_mm'] > 0
    assert scores['station_macro_nll'] > 0
    assert scores['prior_rmse_mm'] > 0


def test_latent_128_requires_prior_and_withheld_posterior_gain_in_every_fold():
    d=module()
    base=[dict(prior_rmse_mm=10.,posterior_withheld_rmse_mm=8.,station_macro_crps_mm=5.)]*3
    good=[dict(prior_rmse_mm=9.8,posterior_withheld_rmse_mm=7.8,station_macro_crps_mm=4.9)]*3
    assert d.choose_latent_dimension(base, good)['latent_dim']==128
    bad=[*good[:2],dict(prior_rmse_mm=10.1,posterior_withheld_rmse_mm=7.,station_macro_crps_mm=4.)]
    assert d.choose_latent_dimension(base, bad)['latent_dim']==64


def test_representation_gate_accepts_zero_KL_if_field_depends_on_latent_and_rejects_noise_only():
    d=module(); data=TinyData()
    config=dict(runtime=dict(device='cpu'),diagnostics=dict(oracle_steps=2))
    result=d.representation_metrics(FunctionalModel(),data,data.train_hours[:2],config)
    assert result['passed'] and result['active_kl_units']==0
    assert result['field_latent_variance_mm2'] > 0
    assert result['label_isolation'] is True
    assert result['label_perturbation_max_error_mm']==0
    assert result['background_rmse_mm']==2
    collapsed=d.representation_metrics(FunctionalModel(True),data,data.train_hours[:2],config)
    assert not collapsed['passed']
    assert 'latent_path_collapsed' in collapsed['reasons']


def test_representation_detects_inference_target_label_path():
    class LeakyVAE(FunctionalVAE):
        def decoder(self,encoded,z,background):
            return super().decoder(encoded,z,background)+encoded[:,None,None,None,None]
    class LeakyModel(FunctionalModel):
        def __init__(self):
            super().__init__();self.vae=LeakyVAE()
        def encode(self,batch):
            # An illegal hidden dependence on B changes the decoded field.
            return batch['target'].sum(1) if 'target' in batch else batch['hour_ids']*0.
    data=TinyData()
    result=module().representation_metrics(LeakyModel(),data,data.train_hours[:1],
                                          dict(diagnostics=dict(oracle_steps=1)))
    assert not result['label_isolation']
    assert 'inference_target_label_path' in result['reasons']


def test_pilots_cannot_start_without_explicit_backend_runtime(tmp_path):
    with pytest.raises(RuntimeError,match='backend'):
        module().run_representation_diagnostics({},TinyData(),tmp_path)
    assert not (tmp_path/'training_only_selection.lock.json').exists()


class NeuralTinyData(TinyData):
    metadata = {'synthetic':True,'grid_shape':[16,16]}

    def batch(self,hours,role='B',device='cpu'):
        out=super().batch(hours,role,device)
        b=len(hours); h=w=16
        out['background']=torch.zeros(b,1,h,w,device=device)
        out.update(grid_condition=torch.zeros(b,37,h,w,device=device),
                   station_tokens=torch.zeros(b,3,31,device=device),
                   station_mask=torch.ones(b,3,dtype=torch.bool,device=device),
                   grid_geometry=torch.zeros(h,w,3,device=device),
                   input_geometry=torch.zeros(3,3,device=device),
                   support=torch.ones(b,h,w,dtype=torch.bool,device=device))
        return out


def test_real_cpu_pilot_runs_five_full_epochs_and_publishes_original_outcome(tmp_path):
    d=module();data=NeuralTinyData()
    hour=int(data.train_hours[0]); validation=hour+13*24
    cfg=dict(scientific=dict(grid_shape=[16,16]),
             runtime=dict(device='cpu',diagnostic_pilot_budget_seconds=40))
    result=d._pilot(cfg,data,dict(fit_hours=[hour],validation_hours=[validation]),
                    np.array([0,0,1,1,2,2]),0,3.18,64,tmp_path)
    assert result['training_outcome']['epochs']==5
    assert result['training_outcome']['step']==5
    assert result['posterior_withheld_rmse_mm'] >= 0
    assert result['checkpoint_sha256']
    checkpoint=torch.load(result['checkpoint'],weights_only=False)
    assert checkpoint['config']['min_epochs']==5
    assert checkpoint['config']['kl_warmup_updates']==1
    assert checkpoint['best']=={}  # No accidental C-development selection.
    assert (tmp_path/'training_only_diagnostics/sigma_3.18_latent_64_fold_0/outcome.json').is_file()


def test_budget_exhaustion_never_publishes_pilot_success(tmp_path):
    d=module(); data=NeuralTinyData(); hour=int(data.train_hours[0])
    cfg=dict(scientific=dict(grid_shape=[16,16]),
             runtime=dict(device='cpu',diagnostic_pilot_budget_seconds=1e-12))
    with pytest.raises(TimeoutError):
        d._pilot(cfg,data,dict(fit_hours=[hour],validation_hours=[hour+13*24]),
                 np.array([0,0,1,1,2,2]),0,3.18,64,tmp_path)
    assert not list(tmp_path.rglob('outcome.json'))
    ledgers=list(tmp_path.rglob('pilot_budget_ledger.json'))
    assert len(ledgers)==1
    import json
    first=json.loads(ledgers[0].read_text())
    assert first['charged_seconds'] > 0
    with pytest.raises(TimeoutError):
        d._pilot(cfg,data,dict(fit_hours=[hour],validation_hours=[hour+13*24]),
                 np.array([0,0,1,1,2,2]),0,3.18,64,tmp_path)
    assert json.loads(ledgers[0].read_text())['charged_seconds'] >= first['charged_seconds']


def test_latent_decision_does_not_turn_nonfinite_outcome_into_success():
    d=module()
    valid=dict(prior_rmse_mm=10.,posterior_withheld_rmse_mm=8.,station_macro_crps_mm=5.)
    bad={**valid,'prior_rmse_mm':float('nan')}
    with pytest.raises(ValueError,match='finite'):
        d.choose_latent_dimension([valid]*3,[valid,valid,bad])


def test_all_pilots_share_one_cumulative_budget(tmp_path,monkeypatch):
    d=module();clock=[0.];calls=[]
    monkeypatch.setattr(d.time,'perf_counter',lambda:clock[0])
    def fake(config,*args):
        calls.append(config['runtime']['_diagnostic_remaining_seconds'])
        clock[0]+=6
        return {'status':'complete'}
    monkeypatch.setattr(d,'_pilot',fake)
    config={'runtime':{'diagnostics_budget_seconds':5}}
    d._globally_bounded_pilot(config,None,None,None,0,3.18,64,tmp_path)
    with pytest.raises(TimeoutError,match='global'):
        d._globally_bounded_pilot(config,None,None,None,1,3.18,64,tmp_path)
    assert calls==[5]


def test_pilot_training_pause_is_resumable_and_completed_pilot_is_reused(tmp_path,monkeypatch):
    from gnss_field.generative import training
    d=module();data=NeuralTinyData();hour=int(data.train_hours[0])
    cfg=dict(scientific=dict(grid_shape=[16,16]),runtime=dict(device='cpu',diagnostic_pilot_budget_seconds=40))
    panel=dict(fit_hours=[hour],validation_hours=[hour+13*24]);folds=np.array([0,0,1,1,2,2])
    original=training.run_stage
    monkeypatch.setattr(training,'run_stage',lambda *args,**kwargs:dict(status='paused',epochs=1,step=1))
    with pytest.raises(TimeoutError,match='paused'):
        d._pilot(cfg,data,panel,folds,0,3.18,64,tmp_path)
    monkeypatch.setattr(training,'run_stage',original)
    completed=d._pilot(cfg,data,panel,folds,0,3.18,64,tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Completed pilot may not train or score again')
    monkeypatch.setattr(d,'_execute_pilot',forbidden)
    assert d._pilot(cfg,data,panel,folds,0,3.18,64,tmp_path)==completed


def test_batched_B_prior_and_posterior_keep_scores_and_whole_held_groups():
    d=module()
    class Data(TinyData):
        def __init__(self): self.calls=[]
        def batch(self,hours,**kwargs):
            self.calls.append(list(hours))
            out=super().batch(hours,**kwargs)
            for i,h in enumerate(hours): out['target_mask'][i,int(h)%6]=False
            return out
    hours=TinyData.train_hours[:7]; held=np.array([1,1,0,0,0,0],bool)
    a=Data(); b=Data(); model=FunctionalModel()
    sa=d.score_B_prior(model,a,hours,held,sigma_eff_mm=3.18)
    sb=d.score_B_prior(model,b,hours,held,sigma_eff_mm=3.18,
                       evaluation_hours_batch=4,evaluation_chunk=8)
    assert [len(x) for x in b.calls]==[4,3]
    for key in sa:
        if isinstance(sa[key],(float,int)): assert sb[key]==pytest.approx(sa[key],abs=.001)
        else: assert sb[key]==sa[key]
    folds=np.array([0,0,1,1,2,2]); a=Data(); b=Data()
    pa=d._posterior_withheld_rmse(model,a,hours,folds,0,'cpu',None)
    pb=d._posterior_withheld_rmse(model,b,hours,folds,0,'cpu',None,
                                evaluation_hours_batch=4,evaluation_chunk=8)
    assert [len(x) for x in b.calls]==[4,3]
    assert pb==pytest.approx(pa,abs=.001)
