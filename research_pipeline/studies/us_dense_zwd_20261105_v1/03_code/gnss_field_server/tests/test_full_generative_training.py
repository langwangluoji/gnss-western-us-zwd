"""Small CPU tests for the full observation-supervised stage lifecycle."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from gnss_field.generative.training import (
    FullTrainingConfig, run_stage, energy_score, edm_dsm_loss,
    load_stage_model, make_model_and_objective,
)


class TinyData:
    train_hours = np.arange(6, dtype=np.int64)
    development_hours = np.array([100], dtype=np.int64)
    metadata = {'synthetic': True}

    def condition_batch(self, hour_ids, device='cpu'):
        ids = torch.as_tensor(hour_ids, device=device)
        b, h, w, n = len(ids), 32, 32, 4
        grid = torch.zeros(b, 37, h, w, device=device)
        grid[:, 0] = ids[:, None, None] / 10
        xy = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'), -1).float()*25000
        return dict(hour_ids=ids, grid_condition=grid,
                    station_tokens=torch.zeros(b, 3, 31, device=device),
                    station_mask=torch.ones(b, 3, dtype=torch.bool, device=device),
                    grid_geometry=torch.cat([xy, torch.zeros(h,w,1,device=device)],-1),
                    input_geometry=torch.zeros(3,3,device=device),
                    background=torch.full((b,1,h,w), 100.,device=device),
                    query_indices=torch.tensor([[33,34,65,66],[65,66,97,98],[97,98,129,130],[129,130,161,162]],device=device),
                    query_weights=torch.full((n,4),.25,device=device),
                    target_geometry=torch.zeros(n,3,device=device),
                    support=torch.ones(b,h,w,dtype=torch.bool,device=device))

    def batch(self,hour_ids,role='B',device='cpu'):
        assert role == 'B'
        batch = self.condition_batch(hour_ids,device)
        batch['target'] = 105 + batch['hour_ids'][:,None].float().expand(-1,4)/5
        batch['target_mask'] = torch.ones(len(hour_ids),4,dtype=torch.bool,device=device)
        return batch


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(tmp_path,stage='vae_representation',family='vae',**kw):
    return FullTrainingConfig(stage=stage,model_family=family,output_dir=str(tmp_path),
                              epochs=2,min_epochs=1,batch_size=2,latent_dim=8,
                              grid_shape=(32,32),device='cpu',warmup_updates=1,sampling_steps=3,
                              evaluate_every_epochs=1,evaluation_callback=lambda *args: {'point':1.,'joint':2.},**kw)


def checkpoint(result):
    return torch.load(result['checkpoint'],map_location='cpu',weights_only=False)


def test_energy_score_uses_iid_u_statistic_and_valid_count():
    # Two valid coordinates: distance=(sqrt(2)*10)/sqrt(2)/10=1.
    samples=torch.tensor([[[0.,0.,float('nan')],[10.,10.,float('nan')]]],requires_grad=True)
    target=torch.tensor([[0.,0.,float('nan')]])
    mask=torch.tensor([[True,True,False]])
    assert energy_score(samples,target,mask).item() == pytest.approx(.0)
    energy_score(samples,target,mask).backward()
    assert torch.isfinite(samples.grad).all()


def test_edm_dsm_target_includes_actual_posterior_draw():
    # D(x)=0, target=2, sigma=1 -> weight2 * error4 =8.
    class Zero(torch.nn.Module):
        def denoise(self,noisy,sigma,condition): return noisy*0
    loss=edm_dsm_loss(Zero(),torch.full((2,3),2.),torch.zeros(2,192),
                       sigma=torch.ones(2),noise=torch.zeros(2,3))
    assert loss.item() == pytest.approx(8.)


def test_full_epochs_and_safe_resume_are_bitwise_equivalent(tmp_path):
    data=TinyData()
    cfg=config(tmp_path/'continuous')
    full=run_stage(cfg,data,evaluate=lambda *args: {'point':1.,'joint':2.})
    stopped=run_stage(replace(cfg,output_dir=str(tmp_path/'resume'),max_updates=2),data,
                      evaluate=lambda *args: {'point':1.,'joint':2.})
    assert stopped['status']=='paused'
    resumed=run_stage(replace(cfg,output_dir=str(tmp_path/'resume')),data,
                      evaluate=lambda *args: {'point':1.,'joint':2.})
    a,b=checkpoint(full),checkpoint(resumed)
    assert full['step']==resumed['step']==6
    assert resumed['epochs']==2
    for key,value in a['raw'].items(): assert torch.equal(value,b['raw'][key]),key
    for key,value in a['ema'].items(): assert torch.equal(value,b['ema'][key]),key
    assert a['cursor']['position']==b['cursor']['position']
    assert a['history']==b['history']
    assert a['evaluation_history']==b['evaluation_history']
    assert {key:[(v['step'],v['weight_source'],v['value']) for v in values] for key,values in a['candidates'].items()}=={key:[(v['step'],v['weight_source'],v['value']) for v in values] for key,values in b['candidates'].items()}
    assert a['optimizers'] and a['rng'] and not a['pending_evaluation']
    initial,_=make_model_and_objective(cfg,data.batch([0,1]))
    assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in a['raw'].items())


def test_vae_synthetic_observation_fit_reduces_station_error(tmp_path):
    cfg=config(tmp_path)
    batch=TinyData().batch([0,1])
    torch.manual_seed(42)
    model,objective=make_model_and_objective(cfg,batch)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    errors=[]
    for _ in range(20):
        optimizer.zero_grad();values=objective(model,batch);values['loss'].backward();optimizer.step()
        # Subtract the fixed Gaussian entropy: this is observation squared error / (2*sigma^2).
        errors.append(values['nll'].item() - np.log(6.36) - .5*np.log(2*np.pi))
    assert np.mean(errors[-4:]) < errors[0]*.85


def test_posterior_cache_draws_bind_all_representation_hashes(tmp_path):
    data=TinyData()
    representation=run_stage(config(tmp_path/'vae'),data)
    dep={'representation':representation['checkpoint']}
    cached=run_stage(config(tmp_path/'cache','posterior_cache',dependency_paths=dep),data)
    payload=torch.load(cached['cache'],weights_only=False)
    assert payload['posterior_mu'].shape==(6,8)
    assert payload['posterior_logstd'].shape==(6,8)
    assert set(payload['representation_hashes'])=={'encoder','decoder','posterior'}
    # Total posterior variance includes conditional variance, not just means.
    assert torch.all(payload['standardization']['std'] > .1)
    gan=run_stage(config(tmp_path/'gan','gan_pretrain','gan',dependency_paths={**dep,'posterior_cache':cached['cache']}),data)
    assert gan['step']==6 and checkpoint(gan)['optimizer_steps']['critic']==18
    critic_group=checkpoint(gan)['optimizers']['critic']['param_groups'][0]
    assert critic_group['betas']==(0.,.9) and critic_group['weight_decay']==0.
    diffusion=run_stage(config(tmp_path/'diffusion','diffusion_pretrain','diffusion',dependency_paths={**dep,'posterior_cache':cached['cache']}),data)
    assert diffusion['step']==6 and np.isfinite(checkpoint(diffusion)['history'][-1]['loss'])
    damaged=torch.load(cached['cache'],weights_only=False)
    damaged['representation_hashes']['decoder']='changed'
    torch.save(damaged,tmp_path/'stale.pt')
    with pytest.raises(ValueError,match='representation'):
        run_stage(config(tmp_path/'bad','gan_pretrain','gan',dependency_paths={**dep,'posterior_cache':str(tmp_path/'stale.pt')}),data)


@pytest.mark.parametrize('family',['vae','gan','diffusion'])
def test_matched_es_and_complete_refinement_update_separate_products(tmp_path,family):
    data=TinyData()
    vae=run_stage(config(tmp_path/'vae'),data)
    dep={'representation':vae['checkpoint']}
    if family!='vae':
        cache=run_stage(config(tmp_path/'cache','posterior_cache',dependency_paths=dep),data)
        pre=run_stage(config(tmp_path/'pre',family+'_pretrain',family,dependency_paths={**dep,'posterior_cache':cache['cache']}),data)
        dep['prior']=pre['checkpoint']
    before,_=load_stage_model(vae['checkpoint'])
    shared=run_stage(config(tmp_path/'shared','prior_es',family,dependency_paths=dep),data)
    shared_model,_=load_stage_model(shared['checkpoint'])
    for k,v in before.vae.encoder.state_dict().items(): assert torch.equal(v,shared_model.vae.encoder.state_dict()[k])
    for k,v in before.vae.decoder.state_dict().items(): assert torch.equal(v,shared_model.vae.decoder.state_dict()[k])
    complete=run_stage(config(tmp_path/'complete','complete_model_es',family,dependency_paths={**dep,'prior':shared['checkpoint']}),data)
    model,_=load_stage_model(complete['checkpoint'])
    assert any(not torch.equal(v,model.vae.decoder.state_dict()[k]) for k,v in shared_model.vae.decoder.state_dict().items())
    assert checkpoint(complete)['latent_cache_active'] is False
    assert checkpoint(complete)['objective_name']==('station_elbo_plus_es' if family=='vae' else 'real_station_es')
    batch=data.condition_batch([0,1])
    fields=model.sample_fields(batch,2,42,steps=3)
    assert fields.shape==(2,2,1,32,32) and torch.isfinite(fields).all()


def test_pending_evaluation_recovery_does_not_skip_or_advance_training(tmp_path):
    data=TinyData()
    cfg=config(tmp_path)
    calls=0
    def fail(*args):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('evaluation interrupted')
        return {'point':1.,'joint':2.}
    with pytest.raises(RuntimeError,match='interrupted'):
        run_stage(cfg,data,evaluate=fail)
    state=torch.load(tmp_path/'last.pt',weights_only=False)
    assert state['pending_evaluation'] and state['step']==0
    assert state['best']=={}
    assert state['evaluation_history']==[] and state['candidates']=={'point':[],'joint':[]}
    result=run_stage(cfg,data,evaluate=lambda *args: {'point':1.,'joint':2.})
    assert result['status']=='complete' and result['step']==6
    assert checkpoint(result)['baseline']['point']['step']==0
    assert checkpoint(result)['best']['point']['step']==3


def test_deterministic_control_is_independent_of_member_seed(tmp_path):
    cfg=config(tmp_path,'deterministic_control')
    model,_=make_model_and_objective(cfg)
    batch=TinyData().condition_batch([0,1])
    a=model.sample_fields(batch,2,41)
    b=model.sample_fields(batch,2,42)
    assert torch.equal(a,b)
    assert torch.equal(a[:,0],a[:,1])

def test_development_compares_raw_and_ema_without_overwriting_optimizer_model(tmp_path):
    observations=[]
    def evaluate(model,*args):
        observations.append({k:v.clone() for k,v in model.state_dict().items()})
        return {'point':1.,'joint':2.}
    result=run_stage(config(tmp_path),TinyData(),evaluate=evaluate)
    state=checkpoint(result)
    assert len(observations)==6  # step0 + two epoch endpoints, each raw and EMA
    assert any(not torch.equal(v,observations[-1][k]) for k,v in observations[-2].items())
    for key,value in observations[-2].items(): assert torch.equal(value,state['raw'][key])
    for key,value in observations[-1].items(): assert torch.equal(value,state['ema'][key])
    assert state['best']['point']['weight_source'] in ('raw','ema')

@pytest.mark.parametrize('family',['gan','diffusion'])
def test_latent_pretraining_resume_preserves_critic_or_dsm_rng(tmp_path,family):
    data=TinyData()
    representation=run_stage(replace(config(tmp_path/'vae'),epochs=1),data)
    dep={'representation':representation['checkpoint']}
    cached=run_stage(config(tmp_path/'cache','posterior_cache',dependency_paths=dep),data)
    dep['posterior_cache']=cached['cache']
    cfg=config(tmp_path/'full',family+'_pretrain',family,dependency_paths=dep)
    full=run_stage(cfg,data)
    run_stage(replace(cfg,output_dir=str(tmp_path/'resumed'),max_updates=2),data)
    resumed=run_stage(replace(cfg,output_dir=str(tmp_path/'resumed')),data)
    a,b=checkpoint(full),checkpoint(resumed)
    for key,value in a['raw'].items(): assert torch.equal(value,b['raw'][key]),key
    assert a['history']==b['history']
    assert a['optimizer_steps']==b['optimizer_steps']
    for name,opt in a['optimizers'].items():
        for param,values in opt['state'].items():
            for key,value in values.items():
                if torch.is_tensor(value): assert torch.equal(value,b['optimizers'][name]['state'][param][key])

def test_checkpoint_reader_keeps_rng_on_cpu_and_optimizer_regularization_is_explicit(tmp_path,monkeypatch):
    import gnss_field.generative.training as training
    path=tmp_path/'state.pt'
    torch.save({'rng':{'torch':torch.get_rng_state()}},path)
    original_load=torch.load
    calls=[]
    def checked_load(*args,**kwargs):
        calls.append(kwargs.get('map_location'))
        return original_load(*args,**kwargs)
    monkeypatch.setattr(torch,'load',checked_load)
    loaded=training.read_training_checkpoint(path)
    assert calls==['cpu'] and loaded['rng']['torch'].device.type=='cpu'
    data=TinyData()
    vae=run_stage(replace(config(tmp_path/'vae'),epochs=1),data)
    opt=checkpoint(vae)['optimizers']['generator']['param_groups'][0]
    assert opt['weight_decay']==pytest.approx(1e-4)
    row=checkpoint(vae)['history'][-1]
    assert row['learning_rate']>0 and row['gradient_norm']>0
    assert checkpoint(vae)['telemetry']['model_seconds']>0

def test_resume_rejects_changed_data_contract_even_with_same_hour_ids(tmp_path):
    data=TinyData()
    cfg=config(tmp_path,max_updates=1)
    run_stage(cfg,data)
    data.metadata={**data.metadata,'changed_background_identity':'other'}
    with pytest.raises(ValueError,match='data'):
        run_stage(replace(cfg,max_updates=None),data)

def test_vae_kl_warmup_changes_loss_and_is_exact_across_resume(tmp_path):
    data=TinyData()
    cfg=config(tmp_path/'full',kl_warmup_updates=3)
    full=run_stage(cfg,data)
    run_stage(replace(cfg,output_dir=str(tmp_path/'resumed'),max_updates=2),data)
    resumed=run_stage(replace(cfg,output_dir=str(tmp_path/'resumed')),data)
    a,b=checkpoint(full),checkpoint(resumed)
    assert [row['beta'] for row in a['history']]==pytest.approx([0.,1/3,2/3,1.,1.,1.])
    assert a['history'][0]['kl_sum']>0
    for row in a['history']:
        assert row['loss']==pytest.approx(row['nll']+row['beta']*row['kl'])
        assert row['negative_elbo']==pytest.approx(row['nll']+row['kl'])
        assert row['kl_sum']==pytest.approx(row['kl']*4)
    assert a['history']==b['history']
    for key,value in a['raw'].items(): assert torch.equal(value,b['raw'][key])
    assert a['science_signature']==b['science_signature']
    # The complete VAE starts with the full ELBO even if a warmup is configured.
    complete_cfg=config(tmp_path/'complete','complete_model_es',kl_warmup_updates=100,
                        dependency_paths={'representation':full['checkpoint']})
    model,objective=make_model_and_objective(complete_cfg)
    values=objective(model,data.batch([0,1]),step=0)
    assert values['beta'].item()==1.

def test_resume_and_posterior_cache_bind_b_labels_with_unchanged_conditions(tmp_path):
    data=TinyData()
    cfg=config(tmp_path/'vae',max_updates=1)
    run_stage(cfg,data)
    original=data.batch
    def changed(*args,**kwargs):
        batch=original(*args,**kwargs)
        batch['target']=batch['target']+1000
        return batch
    data.batch=changed
    with pytest.raises(ValueError,match='data'):
        run_stage(replace(cfg,max_updates=None),data)


def test_candidate_write_failure_preserves_pending_evaluation_and_recovers(tmp_path,monkeypatch):
    import gnss_field.generative.training as training
    original=training.atomic_torch
    failed=False
    def interrupted(path,state):
        nonlocal failed
        if not failed and Path(path).name.startswith(('candidate_','baseline_')):
            failed=True
            raise OSError('candidate publication interrupted')
        return original(path,state)
    monkeypatch.setattr(training,'atomic_torch',interrupted)
    cfg=config(tmp_path,max_updates=0)
    with pytest.raises(OSError,match='publication'):
        run_stage(cfg,TinyData())
    state=torch.load(tmp_path/'last.pt',weights_only=False)
    assert state['pending_evaluation'] and state['best']=={}
    monkeypatch.setattr(training,'atomic_torch',original)
    result=run_stage(cfg,TinyData())
    for selection in result['best'].values():
        assert Path(selection['checkpoint']).is_file() and len(selection['sha256'])==64
    assert not checkpoint(result)['pending_evaluation']
    selected=Path(result['best']['point']['checkpoint'])
    with selected.open('ab') as stream: stream.write(b'tampered')
    with pytest.raises(ValueError,match='candidate'):
        run_stage(cfg,TinyData())

def test_synthetic_b_label_identity_is_independent_of_io_batching():
    from gnss_field.generative.training import _training_label_identity
    data=TinyData()
    assert _training_label_identity(data,data.train_hours,1)==_training_label_identity(data,data.train_hours,3)

def test_complete_evaluation_curves_and_formal_top_three_are_preserved(tmp_path):
    # Step0 beats every trained model: formal candidates must still be selected.
    pairs=[(-100.,-100.),(-99.,-99.),(6.,1.),(5.,2.),(4.,3.),(3.,4.),(2.,5.),(1.,6.),(8.,.5),(7.,.6)]
    calls=0
    def evaluate(*args):
        nonlocal calls
        point,joint=pairs[calls];calls+=1
        return {'point':point,'joint':joint,'coverage':{'90':.8},
                'station_errors':np.array([1.,2.]),'vs':torch.tensor(3.)}
    cfg=replace(config(tmp_path),epochs=4)
    result=run_stage(cfg,TinyData(),evaluate=evaluate)
    state=checkpoint(result)
    assert len(result['evaluation_history'])==10
    assert result['evaluation_history']==state['evaluation_history']
    assert result['history']==state['history']
    assert result['evaluation_history'][0]['metrics']['coverage']=={'90':.8}
    assert result['evaluation_history'][0]['metrics']['station_errors']==[1.,2.]
    assert result['evaluation_history'][0]['metrics']['vs']==3.
    assert {row['weight_source'] for row in result['evaluation_history']}=={'raw','ema'}
    assert result['baseline']['point']['step']==0
    point=result['candidates']['point'];joint=result['candidates']['joint']
    assert [item['value'] for item in point]==[1.,2.,3.]
    assert [item['value'] for item in joint]==[.5,.6,1.]
    assert all(item['step']>0 and item['purpose']=='formal' for item in point+joint)
    assert len({item['checkpoint'] for item in point+joint})==6
    assert result['best']['point']==point[0] and result['best']['joint']==joint[0]
    assert all(Path(item['checkpoint']).is_file() and len(item['sha256'])==64 for item in point+joint)
    # Integrity covers secondary finalists, rather than just the two best files.
    with Path(point[2]['checkpoint']).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='candidate'):
        run_stage(cfg,TinyData(),evaluate=evaluate)


def test_startup_and_preselection_evaluations_never_occupy_formal_slots(tmp_path):
    cfg=config(tmp_path,evaluation_check_updates=(2,),formal_selection_start_step=4)
    result=run_stage(cfg,TinyData())
    assert [(row['step'],row['purpose']) for row in result['evaluation_history']]==[
        (0,'baseline'),(0,'baseline'),(2,'startup'),(2,'startup'),
        (3,'preselection'),(3,'preselection'),(6,'formal'),(6,'formal')]
    assert all(item['step']==6 for items in result['candidates'].values() for item in items)
    assert result['best']['point']['step']==6
    assert len({item['checkpoint'] for items in result['candidates'].values() for item in items})==2

def test_formal_candidate_failure_rolls_back_curve_and_rankings_before_resume(tmp_path,monkeypatch):
    import gnss_field.generative.training as training
    data=TinyData()
    cfg=config(tmp_path/'interrupted')
    original=training.atomic_torch
    failed=False
    def interrupted(path,state):
        nonlocal failed
        name=Path(path).name
        if not failed and name.startswith('candidate_') and name.endswith('_ema.pt'):
            failed=True
            raise OSError('second finalist interrupted')
        return original(path,state)
    monkeypatch.setattr(training,'atomic_torch',interrupted)
    with pytest.raises(OSError,match='finalist'):
        run_stage(cfg,data)
    state=torch.load(Path(cfg.output_dir)/'last.pt',weights_only=False)
    assert state['pending_evaluation'] and state['step']==3
    assert len(state['evaluation_history'])==2
    assert state['candidates']=={'point':[],'joint':[]}
    assert state['best']==state['baseline']
    monkeypatch.setattr(training,'atomic_torch',original)
    resumed=run_stage(cfg,data)
    full=run_stage(replace(cfg,output_dir=str(tmp_path/'continuous')),data)
    a,b=checkpoint(full),checkpoint(resumed)
    assert a['evaluation_history']==b['evaluation_history'] and len(b['evaluation_history'])==6
    assert a['history']==b['history']
    for key,value in a['raw'].items(): assert torch.equal(value,b['raw'][key])
    for product in ('point','joint'):
        assert [(v['step'],v['weight_source'],v['value']) for v in a['candidates'][product]]==[
            (v['step'],v['weight_source'],v['value']) for v in b['candidates'][product]]


def test_startup_evaluation_at_epoch_end_is_diagnostic(tmp_path):
    result=run_stage(config(tmp_path,evaluation_check_updates=(3,)),TinyData())
    assert [(row['step'],row['purpose']) for row in result['evaluation_history']]==[
        (0,'baseline'),(0,'baseline'),(3,'startup'),(3,'startup'),(6,'formal'),(6,'formal')]
    assert all(item['step']==6 for group in result['candidates'].values() for item in group)

def test_evaluation_pause_returns_resumable_completed_updates_and_preserves_rng(tmp_path):
    from gnss_field.generative.evaluation import EvaluationPaused
    data=TinyData()
    cfg=config(tmp_path/'paused')
    calls=0
    captured={}
    def pausing(model,data,hours,config):
        nonlocal calls
        calls+=1
        assert callable(config._evaluation_should_stop)
        if calls==3:
            captured['raw']={key:value.clone() for key,value in model.state_dict().items()}
            captured['rng']=torch.get_rng_state().clone()
            torch.randn(5)  # An interrupted evaluator must not consume training RNG.
            raise EvaluationPaused('allocation ended during C development')
        return {'point':1.,'joint':2.}
    paused=run_stage(cfg,data,evaluate=pausing)
    assert paused['status']=='paused' and paused['step']==3
    state=checkpoint(paused)
    assert state['pending_evaluation'] and state['status']=='paused'
    assert len(state['evaluation_history'])==2 and state['candidates']=={'point':[],'joint':[]}
    assert torch.equal(state['rng']['torch'],captured['rng'])
    for key,value in captured['raw'].items(): assert torch.equal(value,state['raw'][key])
    assert not hasattr(cfg,'_evaluation_should_stop')
    resumed=run_stage(cfg,data)
    full=run_stage(replace(cfg,output_dir=str(tmp_path/'continuous')),data)
    a,b=checkpoint(full),checkpoint(resumed)
    assert a['history']==b['history'] and a['evaluation_history']==b['evaluation_history']
    for key,value in a['raw'].items(): assert torch.equal(value,b['raw'][key])


@pytest.mark.parametrize('trigger',['budget','signal'])
def test_evaluation_stop_predicate_observes_budget_or_signal_and_is_temporary(tmp_path,trigger):
    import signal
    from gnss_field.generative.evaluation import EvaluationPaused
    cfg=config(tmp_path,budget_seconds=0. if trigger=='budget' else None)
    def evaluator(model,data,hours,config):
        if trigger=='signal':
            assert not config._evaluation_should_stop()
            signal.raise_signal(signal.SIGTERM)
        assert config._evaluation_should_stop()
        raise EvaluationPaused('stop requested')
    result=run_stage(cfg,TinyData(),evaluate=evaluator)
    assert result['status']=='paused' and result['step']==0
    assert checkpoint(result)['pending_evaluation']
    assert not hasattr(cfg,'_evaluation_should_stop')
