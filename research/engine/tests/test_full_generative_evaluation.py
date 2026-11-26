from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_development_panel_is_calendar_only_and_covers_month_hour_strata():
    from gnss_field.generative.evaluation import fixed_development_hours
    hours = np.arange(np.datetime64('2019-01-01T00', 'h').astype('i8'),
                      np.datetime64('2020-01-01T00', 'h').astype('i8'))
    panel = fixed_development_hours(hours)
    assert len(panel) == 576 and len(np.unique(panel)) == 576
    months = panel.astype('datetime64[h]').astype('datetime64[M]')
    assert all(np.sum(months == m) == 48 for m in np.unique(months))
    assert np.array_equal(panel, fixed_development_hours(hours[::-1]))


class ConstantModel:
    def eval(self): return self

    def sample_fields(self, batch, members, seed, steps=18):
        return batch['background'][:, None].expand(-1, members, -1, -1, -1).clone()


class ConstantData:
    purpose = 'development'
    metadata = {'synthetic': True}

    def batch(self, hours, role='C', device='cpu'):
        assert role == 'C'
        b = len(hours)
        return dict(hour_ids=torch.tensor(hours, device=device),
            background=torch.full((b, 1, 16, 16), 100., device=device),
            query_indices=torch.tensor([[17, 18, 33, 34]]*4, device=device),
            query_weights=torch.full((4, 4), .25, device=device),
            target_geometry=torch.zeros(4, 3, device=device),
            target=torch.full((b, 4), 100., device=device),
            target_mask=torch.ones(b, 4, dtype=torch.bool, device=device))


def config(**kwargs):
    values = dict(device='cpu', sigma_eff_mm=2., eval_members=8, point_members=8,
                  evaluation_seed=11, sampling_steps=18, evaluation_chunk=4,
                  evaluation_population_manifest=None)
    return SimpleNamespace(**{**values, **kwargs})


def test_point_mean_does_not_include_zero_mean_station_discrepancy():
    from gnss_field.generative.evaluation import evaluate_model
    result = evaluate_model(ConstantModel(), ConstantData(), [1, 2], config())
    assert result['point'] == 0.
    assert result['CRPS_mm'] > 0 and result['width95_mm'] > 0
    assert result['point_mean_mc_se_mm'] == 0
    assert result['probability_members_iid'] is True


def test_nonfinite_published_field_fails_even_if_query_would_be_finite():
    from gnss_field.generative.evaluation import evaluate_model
    class BadField(ConstantModel):
        def sample_fields(self, *args, **kwargs):
            fields = super().sample_fields(*args, **kwargs)
            fields[..., -1, -1] = float('nan')
            return fields
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate_model(BadField(), ConstantData(), [1], config())


def test_empty_fixed_score_hours_are_counted_without_model_dependent_filtering():
    from gnss_field.generative.evaluation import evaluate_model
    class MissingHour(ConstantData):
        def batch(self, hours, **kwargs):
            batch = super().batch(hours, **kwargs)
            if hours[0] == 2:
                batch['target_mask'].zero_()
            return batch
    result = evaluate_model(ConstantModel(), MissingHour(), [1, 2], config())
    assert result['hours_requested'] == 2 and result['hours_scored'] == 1
    assert result['empty_fixed_population_hours'] == [2]


def test_coarse_mean_requires_one_coarsening_per_hour(monkeypatch):
    import gnss_field.generative.evaluation as evaluation
    original = evaluation.coarsen_100km
    shapes = []
    def tracked(fields):
        shapes.append(tuple(fields.shape))
        return original(fields)
    monkeypatch.setattr(evaluation, 'coarsen_100km', tracked)
    result = evaluation.evaluate_model(ConstantModel(), ConstantData(), [1,2], config())
    assert result['coarse_station_macro_rmse_mm'] == 0
    assert len(shapes) == 2 and all(s[0] == 1 for s in shapes)


def test_averaging_then_coarsening_preserves_public_query():
    from gnss_field.geometry import coarsen_100km
    from gnss_field.generative.observations import query_surface
    torch.manual_seed(17)
    fields = torch.randn(1, 8, 1, 16, 16)*10+100
    batch = ConstantData().batch([1])
    a = coarsen_100km(fields.flatten(0,1)).reshape_as(fields)
    b = coarsen_100km(fields.mean(1))[:,None]
    arguments = (batch['query_indices'],batch['query_weights'],batch['target_geometry'][:,2])
    assert (query_surface(a,*arguments).mean(1)-query_surface(b,*arguments)[:,0]).abs().max() < .001


def test_saved_evaluation_blocks_resume_without_recomputing_completed_hours(tmp_path):
    from gnss_field.generative.evaluation import evaluate_model
    class Interrupted(ConstantData):
        def batch(self, hours, **kwargs):
            if hours[0] == 169: raise RuntimeError('simulated allocation interruption')
            return super().batch(hours,**kwargs)
    hours = np.arange(1,171)
    with pytest.raises(RuntimeError,match='simulated'):
        evaluate_model(ConstantModel(),Interrupted(),hours,config(),output=tmp_path)
    assert (tmp_path/'blocks/00000.json').exists()
    class Resumed(ConstantData):
        def batch(self, hours, **kwargs):
            assert hours[0] >= 169
            return super().batch(hours,**kwargs)
    result = evaluate_model(ConstantModel(),Resumed(),hours,config(),output=tmp_path)
    assert result['hours_requested'] == 170
    with np.load(tmp_path/'station_predictions.npz') as z:
        assert np.array_equal(z['hours'], hours)
    with pytest.raises(ValueError,match='Immutable'):
        evaluate_model(ConstantModel(),Resumed(),hours,config(sigma_eff_mm=3),output=tmp_path)


def test_pause_inside_first_week_commits_completed_prefix(tmp_path):
    from gnss_field.generative.evaluation import evaluate_model, EvaluationPaused
    class Counted(ConstantData):
        def __init__(self): self.calls=[]
        def batch(self,hours,**kwargs):
            self.calls.extend(hours)
            return super().batch(hours,**kwargs)
    data=Counted();cfg=config();cfg._evaluation_should_stop=lambda:len(data.calls)>=3
    with pytest.raises(EvaluationPaused):
        evaluate_model(ConstantModel(),data,np.arange(1,11),cfg,output=tmp_path)
    assert (tmp_path/'blocks/00000.json').exists()
    cfg._evaluation_should_stop=lambda:False
    result=evaluate_model(ConstantModel(),data,np.arange(1,11),cfg,output=tmp_path)
    assert result['hours_requested']==10
    assert data.calls==list(range(1,11))


def test_C_label_revision_invalidates_partial_evaluation(tmp_path):
    from gnss_field.generative.evaluation import evaluate_model, EvaluationPaused
    class Revision(ConstantData):
        revision='1'
        def _label_identity(self,role): return self.revision
    data=Revision();cfg=config();cfg._evaluation_should_stop=lambda:True
    with pytest.raises(EvaluationPaused):
        evaluate_model(ConstantModel(),data,[1,2],cfg,output=tmp_path)
    data.revision='2'
    with pytest.raises(ValueError,match='Immutable'):
        evaluate_model(ConstantModel(),data,[1,2],config(),output=tmp_path)


@pytest.mark.parametrize('hours_batch,members,point_members',[(4,8,8),(8,128,64),(16,128,64)])
def test_hour_batches_preserve_all_arrays_metrics_masks_and_keyed_streams(hours_batch,members,point_members):
    from gnss_field.generative.evaluation import _evaluate_hours, _score_arrays
    class Data(ConstantData):
        def __init__(self): self.calls=[]
        def batch(self, hours, **kwargs):
            self.calls.append(list(hours))
            batch=super().batch(hours, **kwargs)
            for i,hour in enumerate(hours):
                batch['target'][i] += float(hour)/10
                if hour%2: batch['target_mask'][i, int(hour)%4]=False
                if hour==4: batch['target_mask'][i].zero_()
            return batch
    class VAE:
        def prior_parameters(self, encoded):
            return encoded[:,None].float()/100, torch.zeros(len(encoded),1)
        def decoder(self, encoded, latent, background):
            return background[:,None]+latent[:,:,:1,None,None]
    class Model:
        vae=VAE(); deterministic=False; family='vae'; latent_dim=1
        def eval(self): return self
        def encode(self, batch): return batch['hour_ids']
    hours=np.array([7,2,4,1,9,3,8,6,5,10,11,12,13,14,15,16,17,18,19])
    single=Data(); batched=Data()
    cfg=config(eval_members=members,point_members=point_members)
    a=_evaluate_hours(Model(),single,hours,cfg,arrays_only=True)
    b=_evaluate_hours(Model(),batched,hours,config(eval_members=members,point_members=point_members,
                      evaluation_hours_batch=hours_batch,evaluation_chunk=32),arrays_only=True)
    assert [len(x) for x in batched.calls]==[len(hours[i:i+hours_batch]) for i in range(0,len(hours),hours_batch)]
    for key in a:
        np.testing.assert_allclose(a[key],b[key],rtol=0,atol=.001,err_msg=key)
    ma=_score_arrays(a,cfg); mb=_score_arrays(b,cfg)
    for key in ma:
        if isinstance(ma[key],(float,int)):
            assert mb[key]==pytest.approx(ma[key],abs=.001), key
        else: assert mb[key]==ma[key], key
    assert mb['empty_fixed_population_hours']==[4]


def test_batch_pause_commits_prefix_and_resumes_with_another_batch_size(tmp_path):
    from gnss_field.generative.evaluation import evaluate_model, EvaluationPaused
    class Data(ConstantData):
        def __init__(self): self.hours=[]
        def batch(self, hours, **kwargs):
            self.hours.extend(hours)
            return super().batch(hours,**kwargs)
    data=Data(); cfg=config(evaluation_hours_batch=4)
    cfg._evaluation_should_stop=lambda:len(data.hours)>=4
    with pytest.raises(EvaluationPaused):
        evaluate_model(ConstantModel(),data,np.arange(1,11),cfg,output=tmp_path)
    assert data.hours==list(range(1,5))
    evaluate_model(ConstantModel(),data,np.arange(1,11),config(evaluation_hours_batch=3),output=tmp_path)
    assert data.hours==list(range(1,11))


@pytest.mark.parametrize('hours_batch,requested_chunk', [(8,16),(8,32),(16,8),(16,16),(16,64)])
def test_decoder_memberchunk_bounds_hour_member_product(hours_batch,requested_chunk):
    from gnss_field.generative.evaluation import _latent_fields
    class VAE:
        def prior_parameters(self, encoded):
            return torch.zeros(len(encoded),1), torch.zeros(len(encoded),1)
        def decoder(self, encoded, latent, background):
            assert latent.shape[0]*latent.shape[1]<=256
            return background[:,None]+latent[:,:,:1,None,None]
    class Model:
        vae=VAE(); deterministic=False; family='vae'; latent_dim=1
        def encode(self,batch): return batch['hour_ids']
    batch=ConstantData().batch(np.arange(hours_batch))
    fields=_latent_fields(Model(),batch,64,7,2,requested_chunk)
    assert fields.shape[:2]==(hours_batch,64)


def test_member_noise_is_exact_by_hour_and_stream_across_batch_shapes(monkeypatch):
    import gnss_field.generative.evaluation as evaluation
    original=evaluation.keyed_noise; snapshots={}
    def track(hours,members,seed,shape,**kwargs):
        noise=original(hours,members,seed,shape,**kwargs)
        for hour,row in zip(hours,noise):
            key=(int(hour),members,seed,tuple(shape))
            if key in snapshots: assert torch.equal(snapshots[key],row)
            else: snapshots[key]=row.clone()
        return noise
    monkeypatch.setattr(evaluation,'keyed_noise',track)
    class VAE:
        def prior_parameters(self, encoded): return torch.zeros(len(encoded),1),torch.zeros(len(encoded),1)
        def decoder(self,encoded,z,background): return background[:,None]+z[:,:,:1,None,None]
    class Model:
        vae=VAE(); family='vae';deterministic=False;latent_dim=1
        def eval(self): return self
        def encode(self,batch): return batch['hour_ids']
    for b in (1,4):
        evaluation._evaluate_hours(Model(),ConstantData(),[1,2,3,4],config(evaluation_hours_batch=b),arrays_only=True)
    assert {key[2] for key in snapshots}=={11,2000011,1000011}


def test_batched_locked_population_preserves_masks_and_rejects_missing_labels(tmp_path):
    from gnss_field.artifacts import sha256,write_json
    from gnss_field.generative.evaluation import _evaluate_hours
    hours=np.arange(np.datetime64('2019-01-01','h').astype('i8'),np.datetime64('2019-01-01','h').astype('i8')+5)
    masks=np.ones((5,4),bool);masks[1]=False;masks[3,2]=False
    archive=tmp_path/'population.npz';np.savez(archive,hours=hours,mask=masks)
    manifest=tmp_path/'population.json'
    write_json(manifest,dict(role='C',year=2019,monthly={'2019-01':dict(file=archive.name,sha256=sha256(archive))}))
    cfg=config(evaluation_hours_batch=4,evaluation_population_manifest=str(manifest))
    arrays=_evaluate_hours(ConstantModel(),ConstantData(),hours,cfg,arrays_only=True)
    assert np.array_equal(arrays['population'],masks)
    class Missing(ConstantData):
        def batch(self,hours,**kwargs):
            batch=super().batch(hours,**kwargs);batch['target_mask'][:,0]=False;return batch
    with pytest.raises(ValueError,match='cannot reproduce'):
        _evaluate_hours(ConstantModel(),Missing(),hours,cfg,arrays_only=True)


@pytest.mark.parametrize('family', ['vae','gan','diffusion'])
def test_neural_hour_batch_preserves_station_arrays(family):
    from gnss_field.generative.evaluation import _evaluate_hours
    from gnss_field.generative.training import FullGenerativeModel, FullTrainingConfig
    old=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(19)
        class Data(ConstantData):
            def batch(self,hours,**kwargs):
                out=super().batch(hours,**kwargs); b=len(hours); h=w=16
                out.update(grid_condition=torch.zeros(b,37,h,w),station_tokens=torch.zeros(b,3,31),
                           station_mask=torch.ones(b,3,dtype=torch.bool),grid_geometry=torch.zeros(h,w,3),
                           input_geometry=torch.zeros(3,3),support=torch.ones(b,h,w,dtype=torch.bool))
                return out
        model=FullGenerativeModel(FullTrainingConfig(stage='prior_es',model_family=family,grid_shape=(16,16)))
        a=_evaluate_hours(model,Data(),[1,2,3,4],config(sampling_steps=18),arrays_only=True)
        b=_evaluate_hours(model,Data(),[1,2,3,4],config(sampling_steps=18,evaluation_hours_batch=4),arrays_only=True)
        for key in a: np.testing.assert_allclose(a[key],b[key],atol=.001,rtol=0,err_msg=key)
    finally: torch.set_num_threads(old)
