from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from gnss_field.contracts import ScienceConfig, RuntimeConfig
from gnss_field.training.pipeline import TrainingPipeline
from gnss_field.training.engine import TrainEngine, StageSpec
from gnss_field.artifacts import write_json, read_json, sha256


def pipeline(tmp_path):
    return TrainingPipeline(replace(ScienceConfig(), bootstrap_replicates=100), RuntimeConfig(device='cpu'), tmp_path)


class Captured(Exception):
    pass


@pytest.mark.parametrize('arm,effective', [('basis',16), ('m2_f',8), ('m2_j',8)])
def test_production_stage_batch(tmp_path, monkeypatch, arm, effective):
    p = pipeline(tmp_path)
    p._manifest = SimpleNamespace(select=lambda *a: np.arange(4))
    source = tmp_path/'mean.pt'; source.write_bytes(b'mean')
    monkeypatch.setattr(p, '_mean_path', lambda: source)
    def engine(model, path, hours, stage, budget, horizon, batch, *args, **kw):
        assert batch == effective
        raise Captured
    monkeypatch.setattr(p, '_engine', engine)
    with pytest.raises(Captured):
        if arm == 'basis':
            p._train_basis(tmp_path, None, 10)
        else:
            p._pixel_run(arm, tmp_path, {'pretrain_lr':1e-4, 'refinement_lr':1e-4})


@pytest.mark.parametrize('arm,expected', [('det',[0,1,2,3]),('gaussian',[3]),('edm',[3]),('gan',[3])])
def test_production_refinement_qualified_hours(tmp_path, monkeypatch, arm, expected):
    p = pipeline(tmp_path)
    p._manifest = SimpleNamespace(select=lambda years, minimum=1: np.flatnonzero(np.arange(1,5)>=minimum))
    monkeypatch.setattr(p, '_foundation_dependencies', lambda: {})
    monkeypatch.setattr(p, '_shared_encoder', lambda m: 'encoder')
    monkeypatch.setattr(p, '_decoder_bundle', lambda: {'hours':np.arange(4)})
    source = tmp_path/'pretrain.pt'; source.write_bytes(b'weights')
    def engine(model, path, hours, stage, budget, horizon, batch, *args, **kw):
        assert batch == 32
        if stage.endswith('_refine'):
            assert hours.tolist() == expected
            raise Captured
        return SimpleNamespace(ema=model)
    monkeypatch.setattr(p, '_engine', engine)
    monkeypatch.setattr(p, '_run', lambda *a, **kw: source)
    with pytest.raises(Captured):
        p._functional_run(arm,tmp_path,{'pretrain_lr':1e-4,'refinement_lr':1e-4})


def test_probability_fixed_station_denominator(tmp_path):
    p = pipeline(tmp_path)
    a = np.array([[1.,np.nan],[np.nan,10.]])
    b = np.where(np.isfinite(a),0.,np.nan)
    result = p._cell_bootstrap(np.array([0,168]),a,b)
    assert result['ci95'] == [5.5,5.5]
    assert result['rejected_draws'] > 0
    assert result['active_column_indices'] == [0,1]


def test_probability_misaligned_validity_rejected(tmp_path):
    p = pipeline(tmp_path)
    with pytest.raises(ValueError,match='validity'):
        p._cell_bootstrap(np.array([0,168]),np.array([[1.],[np.nan]]),np.zeros((2,1)))


@pytest.mark.parametrize('crossed', [False, True])
def test_probability_rejection_bounded(tmp_path,monkeypatch,crossed):
    from gnss_field.evaluation import bootstrap
    p = pipeline(tmp_path)
    a = np.array([[np.nan,np.nan],[1.,10.]])
    monkeypatch.setattr(bootstrap, '_draw_blocks', lambda *args: np.array([0,0]))
    with pytest.raises(bootstrap.BootstrapExhaustedError) as exc:
        p._cell_bootstrap(np.array([0,168]), a, a, max_attempts=100, crossed=crossed)
    assert exc.value.rejected_draws == 100 and exc.value.accepted_replicates == 0


def test_equal_undercoverage_not_confirmed_by_formal_pipeline(tmp_path,monkeypatch):
    p = pipeline(tmp_path)
    cells = {k:np.full((4,2),.5) for k in ('80','90','95')}
    cells.update({k:np.ones((4,2)) for k in ('es','vs','crps')})
    monkeypatch.setattr(p,'_probability_cells',lambda s,a: {k:v-(.1 if a['name']=='joint' and k in ('es','vs','crps') else 0) for k,v in cells.items()})
    records={name:{'name':name,'hours':np.array([0,168,336,504])} for name in ('joint','gaussian')}
    results={'joint':{'metrics':{'primary':{'intervals':{k:{'coverage':.5} for k in ('80','90','95')}}}}}
    formal=p._formal_evaluation(None,records,results)
    assert not formal['probability']['gate']['calibrated_joint_pass']
    assert formal['probability']['gate']['formal_coverage_status'] != 'confirmed'
    assert formal['probability']['coverage_minus_nominal']['80']['difference'] == pytest.approx(-.3)
    assert formal['block_days'] == 7
    assert formal['sensitivity']['14']['block_days'] == 14
    assert formal['sensitivity']['14']['probability']['coverage_minus_nominal']['80']['active_column_indices'] == [0,1]


@pytest.mark.parametrize('stage', ['m2_f_observations','m2_j_observations'])
@pytest.mark.parametrize('improving_product', [None,'point','marginal'])
def test_m2_stopping_tracks_both_products(tmp_path,stage,improving_product):
    cfg=replace(ScienceConfig(),startup_evaluation=1,evaluate_every=1,minimum_refinement_steps=5,patience=6,log_every=100)
    model=torch.nn.Linear(1,1)
    e=TrainEngine(model,tmp_path,[0],StageSpec(stage,12,12,1,1,42,{}),cfg)
    def evaluate(m,step):
        return {k:{'value':1.-(.1 if k==improving_product and step>=7 else 0.)} for k in ('point','marginal')}
    result=e.run(lambda h,m,s:{'loss':m(torch.ones(1,1)).square().mean()},evaluator=evaluate)
    assert result['step'] == (8 if improving_product is None else 12)


def test_m2_marginal_lock_and_retrospective(tmp_path,monkeypatch):
    from gnss_field.training import pipeline as module
    p=pipeline(tmp_path)
    def record(name,point,crps):
        path=tmp_path/(name+'.pt');path.write_bytes(name.encode())
        return p._record('m2_f',path,tmp_path/name,{'values':{'point':{'value':point},'marginal':{'value':crps}},'metrics':{}},1000)
    rows=[record('point_winner',1.,9.),record('crps_winner',9.,1.)]
    bstar=dict(rows[0],model='det')
    write_json(p.root/'direction_gate.json',dict(families={'det':{'fine':{'fine_pass':True},'probability':{},'products':{'point':{'selected':bstar},'joint':{'selected':None}}}},bstar=bstar,gaussian=None))
    write_json(p.root/'stable2019.json',{'records':rows})
    write_json(p.root/'stability.json',dict(stable=True,members=64))
    (p.root/'condition').mkdir();(p.root/'condition'/'frozen.pt').write_bytes(b'encoder')
    for seed in (17,101):write_json(p.directory/f'seed_{seed}'/'replication.json',{})
    for variant in ('pretrain_only','no_geometry_bias','current_only','rank64','rank256','joint_encoder'):
        write_json(p.root/'ablations'/variant/'results.json',{})
    monkeypatch.setattr(p,'_advanced',lambda:{})
    monkeypatch.setattr(p,'_foundation_dependencies',lambda:{})
    monkeypatch.setattr(p,'_store',lambda *a:SimpleNamespace(manifest={'fingerprint':'d'*64}))
    lock=read_json(p.product_lock()['lock'])
    selected=lock['controls']['m2_f_marginal']
    assert selected['checkpoint_sha256']==sha256(tmp_path/'crps_winner.pt')
    assert set(lock['products']) == {'point'}
    monkeypatch.setattr(p,'_validated_lock',lambda:lock)
    monkeypatch.setattr(module,'PackedStore',lambda *a,**k:object())
    monkeypatch.setattr(p,'_hours',lambda *a:np.array([438288]))
    monkeypatch.setattr(p,'_mean_cache',lambda:SimpleNamespace(build=lambda *a,**k:None))
    monkeypatch.setattr(p,'_mean_model',lambda:None)
    monkeypatch.setattr(p,'_builder',lambda *a:None)
    seen=[]
    monkeypatch.setattr(p,'_control_predictor',lambda entry,store:seen.append(entry['checkpoint']) or object())
    monkeypatch.setattr(module,'evaluate_predictor',lambda *a,**k:{'metrics':{'primary':{'station_macro_crps_mm':2.}}})
    monkeypatch.setattr(module,'_arrays',lambda *a:{})
    monkeypatch.setattr(p,'_evaluate_auxiliary',lambda *a:{})
    monkeypatch.setattr(p,'_evaluate',lambda *a,**k:{'metrics':{}})
    monkeypatch.setattr(p,'_fine_bootstrap',lambda *a,**k:{'comparisons':{'a':{'replicate_differences_mm':[1.]}},'gate':{'fine_pass':False}})
    result=read_json(p.evaluate2020()['summary'])
    assert str(tmp_path/'crps_winner.pt') in seen
    assert result['results']['m2_f_marginal']['metrics']['primary']['station_macro_crps_mm']==2.
    assert 'joint' not in result['formal']['selected_product_claims']

def test_probability_nonfinite_forecast_not_missing(tmp_path):
    p=pipeline(tmp_path)
    arrays=dict(truth=np.ones((1,1)),samples=np.full((1,4,1),np.nan),population_F=np.ones((1,1),bool),normal_support_S0=np.ones((1,1),bool))
    with pytest.raises(ValueError,match='finite'):
        p._probability_cells(None,arrays)


def test_gate_rejects_equally_undercovered_controls():
    from gnss_field.evaluation import probability_gate
    gate=probability_gate(es_difference=-1.,vs_difference=-1.,crps_difference=-1.,
        intervals={k:{'coverage':.5} for k in ('80','90','95')},
        coverage_difference_ci={k:[0.,0.] for k in ('80','90','95')},formal=True)
    assert not gate['calibrated_joint_pass']
    assert gate['formal_coverage_status'] != 'confirmed'


def test_pipeline_enforces_registered_numerics(tmp_path):
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.use_deterministic_algorithms(False)
    p=TrainingPipeline(ScienceConfig(),RuntimeConfig(device='cpu',cpus=2),tmp_path,seed=17)
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads()==2
    assert torch.initial_seed()==17


def test_executing_package_identity_portable_and_sensitive(tmp_path):
    from gnss_field.artifacts import package_digest
    for folder in ('source/gnss_field','installed/gnss_field'):
        root=tmp_path/folder;root.mkdir(parents=True)
        (root/'__init__.py').write_text('value=1')
    a,b=tmp_path/'source/gnss_field',tmp_path/'installed/gnss_field'
    assert package_digest(a)==package_digest(b)
    (b/'__pycache__').mkdir();(b/'__pycache__/x.pyc').write_bytes(b'ignored')
    assert package_digest(a)==package_digest(b)
    (b/'__init__.py').write_text('value=2')
    assert package_digest(a)!=package_digest(b)
    with pytest.raises(ValueError,match='package'):
        package_digest(tmp_path/'missing')


def test_verified_release_rejects_stale_executing_package(tmp_path,monkeypatch):
    import shutil
    from gnss_field import runtime
    from gnss_field import data
    actual=Path(runtime.__file__).parent
    planned=tmp_path/'src/gnss_field'
    shutil.copytree(actual,planned,ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(data,'PackedStore',lambda *a:SimpleNamespace(manifest={'registered':True,'fingerprint':'d'*64},verify_all=lambda:None))
    runtime.verified_release_hashes(ScienceConfig(),RuntimeConfig(),tmp_path)
    (planned/'__init__.py').write_text('changed source')
    with pytest.raises(ValueError,match='executing package'):
        runtime.verified_release_hashes(ScienceConfig(),RuntimeConfig(),tmp_path)

def test_formal_summary_keeps_seven_and_fourteen_days(tmp_path,monkeypatch):
    p=pipeline(tmp_path)
    days=[]
    def fine(*args,block_days=None):
        days.append(block_days)
        return {'comparisons':{'point':{'replicate_differences_mm':[-1.]*100}},'gate':{'fine_pass':True},'block_days':block_days}
    monkeypatch.setattr(p,'_fine_bootstrap',fine)
    result=p._formal_evaluation(None,{'point':{},'bstar':{}},{})
    assert days == [7,14]
    assert result['sensitivity']['14']['selected_product_claims']==result['selected_product_claims']
    assert result['sensitivity']['14']['changed_conclusions']==[]

def test_engine_logs_actual_batch_telemetry_and_resumes_counts(tmp_path):
    import json
    science=replace(ScienceConfig(),log_every=1)
    def make():
        model=torch.nn.Linear(1,1)
        return TrainEngine(model,tmp_path,[0,1,2],StageSpec('edm_refine',2,2,5,2,42,{}),science,
            batch_telemetry=lambda batch:{'supervised_station_hours':int(np.array([2,3,4])[batch].sum()),
                'head_nfe_per_member':35,'head_network_invocations':105,'direct_observation_supervision':True})
    e=make()
    objective=lambda h,m,s:{'loss':m(torch.ones(len(h),1)).square().mean()}
    e.run(objective,stop_after=1)
    make().run(objective)
    rows=[json.loads(line) for line in (tmp_path/'training.jsonl').read_text().splitlines()]
    assert [r['effective_hours'] for r in rows]==[5,5]
    assert all(r['head_network_invocations']==105 for r in rows)
    assert rows[-1]['cumulative_supervised_station_hours']==sum(r['supervised_station_hours'] for r in rows)
    assert rows[-1]['training_gpu_card_hours']==0


@pytest.mark.parametrize('stage,expected', [('edm_refine',5),('m2_f_observations',5),('mean_all',5),('edm_pretrain',0)])
def test_pipeline_telemetry_counts_actual_qualified_targets(tmp_path,monkeypatch,stage,expected):
    p=pipeline(tmp_path)
    p._manifest=SimpleNamespace(contract='manifest',masks=lambda hours:np.array([[1,1,0],[1,1,1]],bool)[hours])
    monkeypatch.setattr(p,'_store',lambda:SimpleNamespace(manifest={'training_hash':'data'},grid_id='grid'))
    engine=p._engine(torch.nn.Linear(1,1),tmp_path/'engine',np.array([0,1]),stage,1,1,2,{})
    values=engine.batch_telemetry(np.array([0,1]))
    assert values['supervised_station_hours']==expected
    if stage=='edm_refine':
        assert values['head_nfe_per_member']==35
        assert values['head_network_invocations']==70

def test_mean_micro_sweep_includes_custom_partial_batches(tmp_path,monkeypatch):
    from gnss_field import runtime as module
    from gnss_field.training import fields
    from gnss_field import geometry
    model=torch.nn.Linear(1,1)
    batches=[]
    def context(hours,**kw):
        batches.append(len(hours))
        return torch.ones(len(hours),1)
    monkeypatch.setattr(fields,'predict_patches',lambda net,ctx,store,**kw:(net(ctx),None))
    monkeypatch.setattr(geometry,'surface_query',lambda field,*a:field[:,None,:])
    rows=module._profile_mean_updates(model,SimpleNamespace(build_context=context),[0,1],
        SimpleNamespace(stats={'surface_std':1.}),None,None,torch.ones(2,1),torch.ones(2,1,dtype=torch.bool),
        RuntimeConfig(device='cpu',micro_hours=3,patch_chunk=4))
    assert [r['micro_hours'] for r in rows]==[1,2,3]
    assert batches==[1]*8+[2]*4+[3,3,2]
    assert all(r['effective_hours']==8 for r in rows)
