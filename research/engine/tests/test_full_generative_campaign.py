import pytest


def test_tongji_cpu_allocation_counts_parent_and_two_threads_per_worker():
    from gnss_field.generative.campaign import allocation_cpus
    assert allocation_cpus('prepare',{'preparation_workers':3},{'max_cpus_per_gpu':7})==7
    assert allocation_cpus('control',{}, {'max_cpus_per_gpu':7})==2
    with pytest.raises(ValueError,match='CPU limit'):
        allocation_cpus('prepare',{'preparation_workers':4},{'max_cpus_per_gpu':7})


def test_frozen_background_code_identity_can_remain_anchored_only_for_equal_bytes(tmp_path):
    from gnss_field.generative.campaign import background_code_binding
    old=tmp_path/'v02'/'background.py';current=tmp_path/'v03'/'background.py'
    old.parent.mkdir();current.parent.mkdir();old.write_bytes(b'fixed adapter');current.write_bytes(b'fixed adapter')
    bound=background_code_binding({'background_identity_source':str(old)},current)
    assert list(bound)==[str(old)]
    current.write_bytes(b'changed adapter')
    with pytest.raises(ValueError,match='source bytes'):
        background_code_binding({'background_identity_source':str(old)},current)


def test_actual_release_configuration_passes_cli_contract(tmp_path,monkeypatch,capsys):
    from pathlib import Path
    from gnss_field.artifacts import read_json,write_json
    from gnss_field.generative.campaign import load_config,main
    project=Path(__file__).resolve().parents[1]
    path=project/'results/20261122_full_generative_design_v01/config.server.json'
    config=load_config(path)
    # Only the machine output path changes; scientific/training/eval stay exact.
    config['runtime']['run_root']=str(tmp_path/'campaign')
    output=tmp_path/'config.json';write_json(output,config)
    assert main(['--config',str(output),'plan','--qualified-hours','26262'])==0
    assert '98490' in capsys.readouterr().out


def test_operator_pause_prevents_automatic_next_gpu_submission(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from gnss_field.generative import campaign as module
    monkeypatch.setattr(module,'submit_lock',lambda root:nullcontext())
    campaign=module.Campaign(dict(scientific={},data={},runtime={'run_root':str(tmp_path)}))
    (tmp_path/'pause.request').write_text('operator pause')
    def prohibited(*args,**kwargs):raise AssertionError('Paused campaign cannot submit')
    assert campaign.submit(tmp_path/'config.json',run=prohibited)['status']=='paused_by_operator'


def test_full_epochs_cover_every_qualified_hour_without_old_caps():
    from gnss_field.generative.campaign import stage_work
    work = stage_work(26262)
    assert work['vae_representation']['updates'] == 3283*30
    assert work['gan_pretrain']['updates'] == 206*20
    assert work['diffusion_pretrain']['updates'] == 206*20
    assert work['complete_vae']['updates'] == 3283*3


def test_budget_lock_counts_prior_allocations_and_rejects_unmeasured_work():
    from gnss_field.generative.campaign import estimate_budget
    with pytest.raises(ValueError, match='Missing benchmark'):
        estimate_budget(26262, {}, cap_card_hours=24)


def test_representation_gate_checks_stochastic_function_not_kl_alone():
    from gnss_field.generative.campaign import representation_gate
    values = dict(finite=True, query_error_mm=0.0001, label_isolation=True,
                  latent_field_rms_mm=1., latent_station_rms_mm=.5,
                  prior_rmse_mm=8., background_rmse_mm=8.1, kl_per_hour=0.)
    assert representation_gate(values)['passed']
    values['latent_field_rms_mm'] = 0.
    assert not representation_gate(values)['passed']


def test_stage_order_keeps_prior_and_complete_model_experiments_distinct():
    from gnss_field.generative.campaign import stage_catalogue
    stages = stage_catalogue()
    names = [row['id'] for row in stages]
    assert names.index('posterior_cache') < names.index('gan_pretrain')
    assert names.index('diffusion_pretrain') < names.index('shared_diffusion')
    assert names.index('shared_vae') < names.index('complete_vae')
    assert stages[names.index('gan_pretrain')]['requires_representation_gate']


def measured_benchmark():
    return dict(stage_seconds_per_update={
        'control': .01, 'vae_representation': .01, 'gan_pretrain': .01,
        'diffusion_pretrain': .01, 'shared_vae': .01, 'shared_gan': .01,
        'shared_diffusion': .01, 'complete_vae': .01, 'complete_gan': .01,
        'complete_diffusion': .01}, posterior_cache_seconds_per_batch=.01,
        evaluation_seconds_per_call={'vae': .1, 'gan': .1, 'diffusion': .1},
        checkpoint_seconds_per_write=.01, preparation_seconds=3., diagnostics_seconds=2.,
        final_evaluation_seconds=5., benchmark_elapsed_seconds=4.,
        measured=True, device='A800', full_working_set=True,priced_latent_dim=128)


def test_budget_uses_safety_margin_and_elapsed_accounting():
    from gnss_field.generative.campaign import estimate_budget
    result = estimate_budget(8, measured_benchmark(), cap_card_hours=24, accounted_seconds=10)
    assert result['total_seconds'] > 10 + 1.5 * (3+2+5)
    assert result['accounted_seconds'] == 10
    assert result['stage_caps_seconds']['vae_representation'] > 30*.01
    with pytest.raises(ValueError, match='cap'):
        estimate_budget(26262, measured_benchmark(), cap_card_hours=.001)
    bad = measured_benchmark(); bad['stage_seconds_per_update']['control'] = float('nan')
    with pytest.raises(ValueError):
        estimate_budget(8, bad)


def test_dependency_uses_actual_selected_checkpoint_and_revalidates_hash(tmp_path):
    from gnss_field.generative.campaign import resolve_dependencies
    from gnss_field.artifacts import sha256
    path = tmp_path/'candidate_point_step000000021_ema.pt'; path.write_bytes(b'checkpoint')
    result = {'status': 'complete', 'best': {'point': {'checkpoint': str(path), 'sha256': sha256(path)}}}
    deps = resolve_dependencies('vae_representation', {'control': result})
    assert deps == {'representation': str(path)}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        resolve_dependencies('vae_representation', {'control': result})


def test_unmeasured_or_failed_diagnostics_stop_training(tmp_path):
    from gnss_field.generative.campaign import Campaign
    config = dict(scientific={}, data={}, runtime={'run_root': str(tmp_path)})
    campaign = Campaign(config)
    with pytest.raises(ValueError, match='budget'):
        campaign.require_training_gates('control')


def test_config_rejects_scientific_runtime_mixing(tmp_path):
    from gnss_field.generative.campaign import load_config
    import json
    path = tmp_path/'bad.json'
    path.write_text(json.dumps({'scientific': {'device': 'cuda'}, 'runtime': {}, 'data': {}}))
    with pytest.raises(ValueError, match='runtime'):
        load_config(path)


def test_verified_accounting_blocks_active_job_reuse():
    from gnss_field.generative.campaign import verified_job_accounting
    import subprocess
    def active(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, '123|RUNNING\n', '')
    with pytest.raises(ValueError, match='active'):
        verified_job_accounting(['123'], run=active)
    def done(args, **kwargs):
        assert kwargs['check'] is True
        output = '' if args[0] == 'squeue' else '123|COMPLETED|00:00:00|31|1|gpu:a800:1|0:0|\n'
        return subprocess.CompletedProcess(args, 0, output, '')
    report = verified_job_accounting(['123'], run=done)
    assert report['card_seconds'] == 31


def test_retry_only_controlled_pause_or_node_failure_with_budget():
    from gnss_field.generative.campaign import retry_allowed
    assert retry_allowed('paused', 'COMPLETED', attempts=1, remaining_seconds=100)
    assert retry_allowed('running', 'NODE_FAIL', attempts=2, remaining_seconds=100)
    assert not retry_allowed('error', 'FAILED', attempts=1, remaining_seconds=100)
    assert not retry_allowed('paused', 'COMPLETED', attempts=3, remaining_seconds=100)
    assert not retry_allowed('paused', 'COMPLETED', attempts=1, remaining_seconds=0)


def test_plan_cli_emits_complete_work_without_cuda_or_submission(tmp_path, capsys):
    from gnss_field.generative.campaign import main
    import json
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(scientific={},data={},runtime={'run_root':str(tmp_path/'run')})))
    assert main(['--config',str(path),'plan','--qualified-hours','26262']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['work']['control']['hours_exposed'] == 262620
    assert report['work']['posterior_cache']['updates'] == 3283


def test_complete_candidate_dependencies_never_use_mutable_last_checkpoint(tmp_path):
    from gnss_field.generative.campaign import resolve_dependencies
    from gnss_field.artifacts import sha256
    results = {}
    for stage in ('vae_representation','shared_diffusion'):
        path = tmp_path/(stage+'.pt');path.write_bytes(stage.encode())
        results[stage] = {'status':'complete','best':{'joint':{'checkpoint':str(path),'sha256':sha256(path)}}}
    dependencies = resolve_dependencies('complete_diffusion',results)
    assert dependencies['prior'].endswith('shared_diffusion.pt')
    assert 'posterior_cache' not in dependencies


def test_scientific_lock_includes_evaluation_and_diagnostic_choices(tmp_path):
    from gnss_field.generative.campaign import Campaign
    config = dict(scientific={},data={},runtime={'run_root':str(tmp_path)},evaluation={'point_members':64})
    Campaign(config)
    config['evaluation']['point_members'] = 32
    with pytest.raises(ValueError,match='Immutable'):
        Campaign(config)


def test_submit_preserves_gpu_script_and_just_in_time_cpu_dispatcher(tmp_path,monkeypatch):
    from gnss_field.generative import campaign as module
    from gnss_field.artifacts import write_json
    from contextlib import nullcontext
    import subprocess
    config = dict(scientific={},data={},runtime={'run_root':str(tmp_path),'project_root':str(tmp_path),'chunk_seconds':600})
    campaign = module.Campaign(config)
    write_json(tmp_path/'audit_B.json', {'qualified_hours':8})
    benchmark = measured_benchmark(); benchmark['stage_seconds_per_update']['control'] = 60
    campaign.lock_budget(benchmark)
    write_json(tmp_path/'diagnostics.json', {'passed':True})
    monkeypatch.setattr(module,'submit_lock',lambda root:nullcontext())
    commands=[]
    def scheduler(args,**kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,'123;cluster\n' if len(commands)==1 else '124\n','')
    submitted = campaign.submit(tmp_path/'config.json',run=scheduler)
    assert submitted['stage'] == 'control'
    assert '--gres=gpu:a800:1' in commands[0]
    assert commands[0][-1].startswith('exec ')
    assert 'afterany:123' in commands[1]
    assert not any('gres' in item for item in commands[1])
    assert (tmp_path/'intents'/f"{submitted['token']}.json").is_file()


def test_diagnostic_variances_normalize_without_changing_zero_kl_acceptance():
    from gnss_field.generative.campaign import representation_gate
    values = dict(finite=True,grid_query_max_error_mm=.0001,label_isolation=True,
                  field_latent_variance_mm2=1.,station_latent_variance_mm2=.25,
                  prior_rmse_mm=8.,background_rmse_mm=8.1,kl_per_dim=[0.]*64)
    assert representation_gate(values)['passed']


def test_preflight_jobs_enter_same_durable_accounting_ledger(tmp_path):
    from gnss_field.generative.campaign import Campaign
    config = dict(scientific={},data={},runtime={'run_root':str(tmp_path),
                  'preflight_jobs':[{'job_id':'123','stage':'benchmark'}]})
    campaign=Campaign(config)
    assert campaign.manifest['jobs'][0]['job_id']=='123'
    campaign=Campaign(config)
    assert len(campaign.manifest['jobs'])==1


def test_final_candidate_union_keeps_three_per_product_and_deduplicates(tmp_path):
    from gnss_field.generative.campaign import final_candidates
    from gnss_field.artifacts import sha256
    def selected(name):
        path=tmp_path/name;path.write_bytes(name.encode())
        return {'checkpoint':str(path),'sha256':sha256(path),'value':1.}
    a,b,c=selected('a.pt'),selected('b.pt'),selected('c.pt')
    results={'shared_gan':{'status':'complete','candidates':{'point':[a,b],'joint':[b,c]},'best':{'point':a,'joint':b}}}
    rows=final_candidates(results,stages=['shared_gan'])
    assert [r['checkpoint'] for r in rows['shared_gan']] == [a['checkpoint'],b['checkpoint'],c['checkpoint']]


def test_real_representation_diagnostics_feed_campaign_gate():
    from test_full_generative_diagnostics import TinyData, FunctionalModel
    from gnss_field.generative.diagnostics import representation_metrics
    from gnss_field.generative.campaign import representation_gate
    data=TinyData()
    config={'runtime':{'device':'cpu'},'diagnostics':{'oracle_steps':2}}
    metrics=representation_metrics(FunctionalModel(),data,data.train_hours[:2],config)
    assert representation_gate(metrics)['passed']
    collapsed=representation_metrics(FunctionalModel(True),data,data.train_hours[:2],config)
    assert not representation_gate(collapsed)['passed']


def test_vae_warmup_is_ten_percent_of_full_work_not_repeated_for_complete():
    from gnss_field.generative.campaign import stage_work
    work=stage_work(26262)
    assert work['vae_representation']['kl_warmup_updates']==9849
    assert work['complete_vae']['kl_warmup_updates']==0


def test_cache_preparation_includes_empty_C_calendar_hours(tmp_path,monkeypatch,capsys):
    from gnss_field.generative import campaign as module
    import json
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path/'run')})
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    class Data:
        def _other_view(self,purpose): return self
        def audit(self,role): return {'hour_ids':[1,2,3],'eligible_hours':[1,3]}
        def prepare(self,hours=None,role=None):
            return {'prepared_hours':len(hours) if hours is not None else 2}
    monkeypatch.setattr(module,'build_data',lambda *a,**k:Data())
    module.main(['--config',str(path),'prepare'])
    result=json.loads(capsys.readouterr().out)
    assert result['C']['prepared_hours']==3


def test_formal_development_hours_use_original_calendar_and_reject_other_population():
    from gnss_field.generative.campaign import registered_development_hours
    import numpy as np
    hours=registered_development_hours({'scientific':{},'training':{}})
    assert len(hours)==576
    assert hours[0]==int(np.datetime64('2019-01-11T00','h').astype('i8'))
    assert hours[-1]==int(np.datetime64('2019-12-21T23','h').astype('i8'))
    with pytest.raises(ValueError,match='development'):
        registered_development_hours({'scientific':{},'training':{'development_hours':hours[:-1]}})


def test_selected_latent_cannot_exceed_measured_pricing_dimension(tmp_path):
    from gnss_field.generative.campaign import Campaign
    from gnss_field.artifacts import write_json
    config=dict(scientific={'latent_dim':64},data={},runtime={'run_root':str(tmp_path)})
    campaign=Campaign(config)
    write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['priced_latent_dim']=64
    campaign.lock_budget(benchmark)
    write_json(tmp_path/'diagnostics.json',{'passed':True})
    write_json(tmp_path/'training_only_selection.lock.json',{'latent_dim':128,'sigma_eff_mm':6.36})
    with pytest.raises(ValueError,match='latent'):
        campaign.require_training_gates('control')


def test_final_evaluation_uses_one_job_deadline_instead_of_old_checkpoint_budget(monkeypatch):
    from gnss_field.generative.campaign import evaluation_candidate_config
    import time
    monkeypatch.setattr(time,'monotonic',lambda:100.)
    state={'config':{'budget_seconds':7000.}}
    first=evaluation_candidate_config(state,{},'cpu',[1],deadline=700.)
    assert first.budget_seconds==600.
    monkeypatch.setattr(time,'monotonic',lambda:200.)
    second=evaluation_candidate_config(state,{},'cpu',[1],deadline=700.)
    assert second.budget_seconds==500.
    monkeypatch.setattr(time,'monotonic',lambda:701.)
    from gnss_field.generative.evaluation import EvaluationPaused
    with pytest.raises(EvaluationPaused):
        evaluation_candidate_config(state,{},'cpu',[1],deadline=700.)


def test_final_evaluation_submission_passes_current_chunk_budget(tmp_path,monkeypatch):
    from gnss_field.generative import campaign as module
    from gnss_field.artifacts import write_json
    from contextlib import nullcontext
    import subprocess
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'chunk_seconds':600})
    campaign=module.Campaign(config)
    write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['final_evaluation_seconds']=600.
    campaign.lock_budget(benchmark)
    write_json(tmp_path/'diagnostics.json',{'passed':True})
    write_json(tmp_path/'representation_gate.json',{'passed':True})
    monkeypatch.setattr(module,'submit_lock',lambda root:nullcontext())
    monkeypatch.setattr(campaign,'_next_stage',lambda:'final_evaluation')
    commands=[]
    def scheduler(args,**kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,'123\n' if len(commands)==1 else '124\n','')
    campaign.submit(tmp_path/'config.json',run=scheduler)
    assert 'evaluate --chunk-seconds 540' in commands[0][-1]


def test_zero_benchmark_elapsed_is_not_a_positive_measured_cost():
    from gnss_field.generative.campaign import estimate_budget
    benchmark=measured_benchmark();benchmark['benchmark_elapsed_seconds']=0
    with pytest.raises(ValueError): estimate_budget(8,benchmark)


def test_diagnostic_preflight_rejects_cost_before_any_pilot(tmp_path):
    from gnss_field.generative.campaign import Campaign, diagnostic_preflight_config
    from gnss_field.artifacts import write_json
    campaign=Campaign(dict(scientific={},data={},runtime={'run_root':str(tmp_path)}))
    write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['diagnostics_seconds']=2500
    write_json(tmp_path/'benchmark.json',benchmark)
    with pytest.raises(ValueError,match='Preflight'):
        diagnostic_preflight_config(campaign)


def test_diagnostic_preflight_uses_original_global_clock_and_retains_remaining_budget(tmp_path,monkeypatch):
    from gnss_field.generative.campaign import Campaign, diagnostic_preflight_config
    from gnss_field.artifacts import write_json
    import time
    campaign=Campaign(dict(scientific={},data={},runtime={'run_root':str(tmp_path)}))
    write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    write_json(tmp_path/'benchmark.json',measured_benchmark())
    write_json(tmp_path/'preflight_clock.lock.json',{'started_unix':100.,'cap_seconds':3600})
    monkeypatch.setattr(time,'time',lambda:1000.)
    config=diagnostic_preflight_config(campaign)
    assert config['runtime']['diagnostics_deadline_unix']<=3700
    assert config['runtime']['diagnostics_remaining_seconds']<=2700
    monkeypatch.setattr(time,'time',lambda:3701.)
    with pytest.raises(ValueError,match='Preflight'):
        diagnostic_preflight_config(campaign)


def test_final_reuse_rejects_changed_C_labels_even_when_population_is_same(tmp_path):
    from gnss_field.generative.campaign import Campaign, verify_development_release
    from gnss_field.artifacts import write_json
    campaign=Campaign(dict(scientific={},data={},runtime={'run_root':str(tmp_path)}))
    write_json(tmp_path/'audit_C.json',{'labels_identity':'release-old','hour_ids':[1,2]})
    class Data:
        def audit(self,role): return {'labels_identity':'release-new','hour_ids':[1,2]}
    with pytest.raises(ValueError,match='C.*release'):
        verify_development_release(campaign,Data())


def test_staged_preparation_prefix_precedes_every_model_and_aux_gates_only_budget(tmp_path):
    from gnss_field.generative.campaign import Campaign
    from gnss_field.artifacts import write_json
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'staged_preparation':True})
    campaign=Campaign(config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    campaign.lock_budget(measured_benchmark())
    assert campaign._next_stage()=='prepare'
    assert campaign.require_training_gates('prepare')['cap_seconds']==86400
    campaign.manifest['stages']['prepare']={'status':'complete'}
    assert campaign._next_stage()=='diagnostics'
    assert campaign.require_training_gates('diagnostics')['cap_seconds']==86400
    campaign.manifest['stages']['diagnostics']={'status':'complete'}
    assert campaign._next_stage()=='control'
    with pytest.raises((FileNotFoundError,ValueError)):
        campaign.require_training_gates('control')


def test_full_cache_and_scientific_pilots_may_exceed_one_hour_inside_total24():
    from gnss_field.generative.campaign import estimate_budget
    benchmark=measured_benchmark();benchmark.update(preparation_seconds=4000.,diagnostics_seconds=8000.)
    plan=estimate_budget(8,benchmark,accounted_seconds=356.)
    assert plan['stage_caps_seconds']['prepare']==6000
    assert plan['stage_caps_seconds']['diagnostics']==12000
    assert plan['prior_seconds']==360
    benchmark['benchmark_elapsed_seconds']=3601
    with pytest.raises(ValueError,match='Preflight'):
        estimate_budget(8,benchmark)


def test_startup_gpu_compile_is_priced_once_per_stage_not_once_per_update():
    from gnss_field.generative.campaign import estimate_budget
    benchmark=measured_benchmark();before=estimate_budget(26262,benchmark)
    benchmark['stage_startup_seconds']={'control':100.}
    benchmark['posterior_cache_startup_seconds']=50.
    after=estimate_budget(26262,benchmark)
    assert after['stage_caps_seconds']['control']-before['stage_caps_seconds']['control']==pytest.approx(150.)
    assert after['stage_caps_seconds']['posterior_cache']-before['stage_caps_seconds']['posterior_cache']==pytest.approx(75.)


def test_short_prepare_allocation_progresses_without_vae_dependencies(tmp_path,monkeypatch):
    from gnss_field.generative import campaign as module
    from gnss_field.artifacts import write_json
    from contextlib import nullcontext
    import subprocess
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'staged_preparation':True,'chunk_seconds':600})
    campaign=module.Campaign(config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['preparation_seconds']=800.
    campaign.lock_budget(benchmark)
    monkeypatch.setattr(module,'submit_lock',lambda root:nullcontext())
    commands=[]
    def scheduler(args,**kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,'123\n' if len(commands)==1 else '124\n','')
    result=campaign.submit(tmp_path/'config.json',run=scheduler)
    assert result['stage']=='prepare'
    assert '--cpus-per-task=7' in commands[0]
    assert '--signal=B:USR1@600' in commands[0]
    assert 'prepare --chunk-seconds 600' in commands[0][-1]
    assert '--time' in commands[0] and commands[0][commands[0].index('--time')+1]=='20'



def test_short_model_allocation_keeps_work_time_after_scaled_grace(tmp_path,monkeypatch):
    from gnss_field.generative import campaign as module
    from gnss_field.artifacts import write_json
    from contextlib import nullcontext
    import subprocess
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'chunk_seconds':600})
    campaign=module.Campaign(config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['stage_seconds_per_update']['control']=4.
    campaign.lock_budget(benchmark);write_json(tmp_path/'diagnostics.json',{'passed':True})
    monkeypatch.setattr(module,'submit_lock',lambda root:nullcontext())
    commands=[]
    def scheduler(args,**kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args,0,'123\n' if len(commands)==1 else '124\n','')
    campaign.submit(tmp_path/'config.json',run=scheduler)
    assert '--signal=B:USR1@20' in commands[0]
    assert 'train --stage control --chunk-seconds 40' in commands[0][-1]
    assert commands[0][commands[0].index('--time')+1]=='1'


def test_staged_preflight_locks_budget_without_preparing_or_running_pilots(tmp_path,monkeypatch):
    from gnss_field.artifacts import write_json,read_json
    import importlib.util
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/'tools/run_full_generative_preflight.py'
    spec=importlib.util.spec_from_file_location('full_preflight_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'staged_preparation':True,
        'preflight_jobs':[{'job_id':'3697508','stage':'benchmark','elapsed_seconds':356.}]})
    path=tmp_path/'config.json';write_json(path,config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    monkeypatch.setenv('SLURM_JOB_ID','456')
    monkeypatch.setattr(module,'build_data',lambda *a,**k:object())
    import gnss_field.generative.benchmark as benchmark_module
    monkeypatch.setattr(benchmark_module,'run_benchmark',lambda *a,**k:measured_benchmark())
    module.main(['--config',str(path),'preflight'])
    assert read_json(tmp_path/'preflight_status.json')['state']=='COMPLETE'
    assert read_json(tmp_path/'budget.lock.json')['prior_seconds']==360
    assert not (tmp_path/'preparation.json').exists()
    assert not (tmp_path/'diagnostics.json').exists()


def test_auxiliary_prepare_pauses_without_destroying_partial_artifacts(tmp_path,monkeypatch):
    from gnss_field.generative.campaign import Campaign
    from gnss_field.artifacts import write_json,read_json
    import gnss_field.generative.preparation as preparation
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'staged_preparation':True})
    campaign=Campaign(config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8,'eligible_hours':[1,2]})
    write_json(tmp_path/'audit_C.json',{'hour_ids':[3,4,5]})
    campaign.lock_budget(measured_benchmark())
    partial=tmp_path/'atomic_month.json';partial.write_text('completed month')
    def pause(*args,**kwargs): raise TimeoutError('controlled deadline')
    monkeypatch.setattr(preparation,'prepare_parallel',pause)
    result=campaign.run_auxiliary('prepare',chunk_seconds=1)
    assert result['status']=='paused'
    assert read_json(tmp_path/'campaign.json')['stages']['prepare']['status']=='paused'
    assert partial.read_text()=='completed month'


def test_diagnostics_stage_receives_cumulative_cap_and_new_allocation_deadline(tmp_path,monkeypatch):
    from gnss_field.generative import campaign as module
    from gnss_field.artifacts import write_json
    import gnss_field.generative.diagnostics as diagnostics
    import time
    config=dict(scientific={},data={},runtime={'run_root':str(tmp_path),'staged_preparation':True,
        'evaluation_hours_batch':8,'evaluation_chunk':16})
    campaign=module.Campaign(config);write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['diagnostics_seconds']=8000
    campaign.lock_budget(benchmark)
    monkeypatch.setattr(module,'build_data',lambda *a,**k:object())
    monkeypatch.setattr(time,'time',lambda:1000.)
    def completed(config,data,root):
        return {'status':'complete','runtime_contract':config['runtime']}
    monkeypatch.setattr(diagnostics,'run_representation_diagnostics',completed)
    result=campaign.run_auxiliary('diagnostics',chunk_seconds=300)
    runtime=result['runtime_contract']
    assert runtime['diagnostics_budget_seconds']==12000
    assert 1299<runtime['diagnostics_deadline_unix']<=1300
    assert 299<runtime['diagnostics_remaining_seconds']<=300
    assert runtime['evaluation_hours_batch']==8 and runtime['evaluation_chunk']==16


def test_parallel_preparation_startup_is_debited_once_before_safety_margin():
    from gnss_field.generative.campaign import estimate_budget
    benchmark=measured_benchmark();before=estimate_budget(8,benchmark)
    benchmark.update(preparation_startup_seconds=100.,preparation_total_seconds=103.)
    after=estimate_budget(8,benchmark)
    assert after['stage_caps_seconds']['prepare']-before['stage_caps_seconds']['prepare']==pytest.approx(150.)


def test_subset_locked_final_products_use_actual_precommitted_checkpoint_shas(tmp_path):
    from gnss_field.generative.campaign import final_candidates
    from gnss_field.artifacts import sha256
    def selection(name):
        path=tmp_path/name;path.write_bytes(name.encode())
        return {'checkpoint':str(path),'sha256':sha256(path),'value':1.}
    results={}
    for stage in ('control','shared_vae','shared_gan','shared_diffusion','complete_vae','complete_gan','complete_diffusion'):
        results[stage]={'status':'complete','best':{'point':selection(stage+'-point.pt'),'joint':selection(stage+'-joint.pt')}}
    rows=final_candidates(results,protocol='subset_locked_products')
    assert set(rows)==set(results)
    assert sum(len(row['frozen_products']) for values in rows.values() for row in values)==10
    assert rows['control'][0]['frozen_products']==['point']
    assert rows['shared_gan'][0]['frozen_products']==['joint']
    assert rows['shared_gan'][0]['sha256']==results['shared_gan']['best']['joint']['sha256']


def test_full_year_crossing_scores_cannot_reselect_subset_locked_products():
    from gnss_field.generative.campaign import final_product_reports
    rows=[dict(checkpoint='point-A',sha256='a',registered_selection={'frozen_products':['point'],'value':3.},
        metrics_K128={'station_macro_rmse_mm':100.,'group_ES_mm':2.},
        metrics_K64={'station_macro_rmse_mm':101.,'group_ES_mm':1.}),
        dict(checkpoint='joint-B',sha256='b',registered_selection={'frozen_products':['joint'],'value':4.},
        metrics_K128={'station_macro_rmse_mm':1.,'group_ES_mm':100.},
        metrics_K64={'station_macro_rmse_mm':2.,'group_ES_mm':101.})]
    products=final_product_reports(rows,protocol='subset_locked_products')
    assert products['point']['checkpoint']=='point-A'
    assert products['joint']['checkpoint']=='joint-B'
    assert products['point']['value_K128']==100.
    assert 'stable_K64_K128' not in products['point']
    assert products['point']['selection_basis']=='fixed_C2019_576_subset'


def test_final_protocol_policy_is_not_forwarded_to_training_dataclass(monkeypatch):
    from gnss_field.generative.campaign import evaluation_candidate_config
    import time
    monkeypatch.setattr(time,'monotonic',lambda:1.)
    cfg=evaluation_candidate_config({'config':{}},{'final_selection_protocol':'subset_locked_products'},'cpu',[1],deadline=20.)
    assert cfg.eval_members==128


def test_budget_priced_for_ten_products_cannot_run_old_annual_reranking_policy(tmp_path):
    from gnss_field.generative.campaign import Campaign
    from gnss_field.artifacts import write_json
    campaign=Campaign(dict(scientific={},data={},runtime={'run_root':str(tmp_path)}))
    write_json(tmp_path/'audit_B.json',{'qualified_hours':8})
    benchmark=measured_benchmark();benchmark['max_final_candidates']=10
    campaign.lock_budget(benchmark);write_json(tmp_path/'diagnostics.json',{'passed':True})
    with pytest.raises(ValueError,match='final.*candidate'):
        campaign.require_training_gates('control')
