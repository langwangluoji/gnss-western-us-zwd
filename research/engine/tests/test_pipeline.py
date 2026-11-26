from dataclasses import replace

import numpy as np
import pytest
import torch

from gnss_field.contracts import ScienceConfig, RuntimeConfig
from gnss_field.training.engine import TrainEngine, StageSpec


def test_auxiliary_optimizer_resume_and_effective_batch_hook(tmp_path):
    science = replace(ScienceConfig(), startup_evaluation=100, log_every=1)
    spec = StageSpec('gan_pretrain', 2, 2, 4, 1, 42, {})
    model = torch.nn.ModuleDict({'head': torch.nn.Linear(1, 1), 'critic': torch.nn.Linear(1, 1)})
    opt = torch.optim.AdamW(model['head'].parameters(), lr=1e-4)
    aux = torch.optim.Adam(model['critic'].parameters(), lr=1e-4, betas=(0., .9))
    engine = TrainEngine(model, tmp_path, np.arange(8), spec, science,
                         optimizers=[opt], auxiliary_optimizers=[aux])
    calls = []

    def before(hours, current, step):
        calls.append((len(hours), step))
        for _ in range(5):
            aux.zero_grad(set_to_none=True)
            current['critic'](torch.ones(1, 1)).square().mean().backward()
            aux.step()

    def loss(hours, current, step):
        return {'loss': current['head'](torch.ones(len(hours), 1)).square().mean()}

    engine.run(loss, before_update=before, stop_after=1)
    saved = torch.load(tmp_path/'last.pt', weights_only=False)
    assert calls == [(4, 1)]
    assert saved['auxiliary_optimizers'][0]['state'][0]['step'] == 5
    resumed = TrainEngine(model, tmp_path, np.arange(8), spec, science,
                          optimizers=[opt], auxiliary_optimizers=[aux])
    resumed.run(loss, before_update=before)
    assert calls == [(4, 1), (4, 2)]
    assert aux.state_dict()['state'][0]['step'] == 10


def test_stage_pause_blocks_hook(tmp_path):
    spec = StageSpec('gan_pretrain', 1, 1, 4, 1, 42, {})
    engine = TrainEngine(torch.nn.Linear(1, 1), tmp_path, [1, 2], spec, ScienceConfig())
    (tmp_path/'PAUSE').touch()
    engine.run(lambda *_: pytest.fail('updated during pause'),
               before_update=lambda *_: pytest.fail('critic updated during pause'))
    assert engine.step == 0


def test_pipeline_aliases_and_2020_gate_before_data_open(tmp_path, monkeypatch):
    from gnss_field.training.pipeline import TrainingPipeline
    expected = {'mean_base', 'mean_all', 'mean_cache', 'basis_cv', 'representation_gate',
                'basis_final', 'condition', 'search', 'promote', 'full2019_rerank', 'stability',
                'direction_gate', 'ablation', 'replicate', 'product_lock', 'evaluate2020', 'export'}
    assert set(TrainingPipeline.ALIASES) == expected
    pipeline = TrainingPipeline(ScienceConfig(), RuntimeConfig(device='cpu'), tmp_path)
    with pytest.raises((FileNotFoundError, PermissionError), match='lock'):
        pipeline.run_task('evaluate2020')
    with pytest.raises(ValueError):
        pipeline.run_task('unregistered')


def test_observation_edm_pipeline_35_calls_and_frozen_foundations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gnss_field.artifacts import model_digest
    from gnss_field.contracts import ContextBatch
    from gnss_field.objectives import CoefficientTransform
    from gnss_field.training.pipeline import TrainingPipeline
    from gnss_field.training.system import FrozenDecoder
    science = replace(ScienceConfig(), rank=2)
    pipeline = TrainingPipeline(science, RuntimeConfig(device='cpu'), tmp_path)
    context = ContextBatch(hours=torch.tensor([420000]), stations=torch.randn(1, 4, 17),
            padding=torch.zeros(1, 4, dtype=torch.bool), coarse=torch.randn(1, 4, 15, 6, 6),
            maps=torch.ones(1, 6, 24, 24), baseline=torch.ones(1, 1, 24, 24)*100,
            mean=torch.ones(1, 1, 24, 24)*100)
    store = SimpleNamespace(stats={'z0_std': 30., 'surface_std': 25.})
    phi = torch.randn(576, 2)
    transform = CoefficientTransform(torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool))
    pipeline._decoder = FrozenDecoder(phi, (24, 24), transform, 30.)
    monkeypatch.setattr(pipeline, '_store', lambda *a: store)
    monkeypatch.setattr(pipeline, '_context', lambda h: context)
    monkeypatch.setattr(pipeline, '_labels', lambda h: (torch.ones(1, 4)*105,
        torch.ones(1, 4, dtype=torch.bool), np.array([[8., 8.], [9., 8.], [8., 9.], [9., 9.]]), np.zeros(4)))
    model = pipeline._model('edm')
    encoder_before, phi_before = model_digest(model.encoder), phi.clone()
    count = []
    original = model.head.denoise
    def denoise(*args):
        count.append(1)
        return original(*args)
    monkeypatch.setattr(model.head, 'denoise', denoise)
    loss = pipeline._observation_loss([420000], model, 1)
    loss.backward()
    assert len(count) == 35
    assert model.head.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model_digest(model.encoder) == encoder_before
    assert torch.equal(phi, phi_before)


def test_promotion_rejects_observation_weights_and_changed_contract(tmp_path):
    from gnss_field.training.pipeline import TrainingPipeline
    pipeline = TrainingPipeline(ScienceConfig(), RuntimeConfig(device='cpu'), tmp_path)
    science = replace(ScienceConfig(), log_every=1)
    model = torch.nn.Linear(1, 1)
    source_spec = StageSpec('edm_refine', 1, 2, 1, 1, 42, {})
    source = TrainEngine(model, tmp_path/'source', [1, 2], source_spec, science)
    source.run(lambda *a: {'loss': model(torch.ones(1, 1)).square().mean()})
    target = TrainEngine(torch.nn.Linear(1, 1), tmp_path/'target', [1, 2],
                         replace(source_spec, stage='edm_pretrain', budget=2), science)
    with pytest.raises(ValueError, match='contract differs'):
        pipeline._fork(target, tmp_path/'source'/'last.pt')


def test_verified_promotion_preserves_optimizer_rng_cursor(tmp_path):
    from gnss_field.training.engine import capture_rng
    from gnss_field.training.pipeline import TrainingPipeline
    pipeline = TrainingPipeline(ScienceConfig(), RuntimeConfig(device='cpu'), tmp_path)
    science = replace(ScienceConfig(), log_every=1)
    source_spec = StageSpec('edm_pretrain', 1, 2, 1, 1, 42, {'data': 'same'})
    model = torch.nn.Linear(1, 1)
    source = TrainEngine(model, tmp_path/'source', [1, 2], source_spec, science)
    source.run(lambda *a: {'loss': model(torch.ones(1, 1)).square().mean()})
    source_path = tmp_path/'source'/'last.pt'
    saved = torch.load(source_path, weights_only=False)
    target = TrainEngine(torch.nn.Linear(1, 1), tmp_path/'target', [1, 2],
                         replace(source_spec, budget=2), science)
    pipeline._fork(target, source_path)
    assert target.step == 1
    assert target.cursor.position == saved['cursor']['position']
    assert torch.equal(capture_rng()['torch'], saved['rng']['torch'])
    assert target.optimizers[0].state_dict()['state'][0]['step'] == 1
    assert target.phase == 'RUNNING'


def test_mean_pipeline_runs_actual_published_field_update(legacy_fixture, tmp_path, monkeypatch):
    from gnss_field.artifacts import model_digest
    from gnss_field.data import pack_legacy
    from gnss_field.training.pipeline import TrainingPipeline
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    science = replace(ScienceConfig(), budgets={**ScienceConfig().budgets, 'mean': 1}, log_every=1)
    pipeline = TrainingPipeline(science, RuntimeConfig(device='cpu', data_root=str(tmp_path/'packed')),
                                tmp_path/'campaign')
    original = model_digest(pipeline._model('mean_base'))
    monkeypatch.setattr(pipeline, '_screen', lambda *args: lambda *a: {'point': {'value': 1.}})
    monkeypatch.setattr(pipeline, '_evaluate', lambda *a, **k: {'values': {'point': {'value': 1.}}})
    outputs = pipeline.run_task('mean_base')
    assert outputs['checkpoint'].is_file() and outputs['frozen'].is_file()
    trained = pipeline._mean_model()
    assert model_digest(trained) != original
    state = torch.load(outputs['checkpoint'], weights_only=False)
    assert state['spec']['effective_hours'] == 8
    assert state['spec']['dependencies']['years'] == (2016, 2017)


def test_basis_teacher_fit_excludes_held_current_B(legacy_fixture, tmp_path, monkeypatch):
    from gnss_field.data import pack_legacy
    from gnss_field.models.basis import SpatialBasis
    from gnss_field.training.pipeline import TrainingPipeline
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    pipeline = TrainingPipeline(replace(ScienceConfig(), rank=2),
                RuntimeConfig(device='cpu', data_root=str(tmp_path/'packed')), tmp_path/'campaign')
    hours = pipeline.manifest.select([2016])[:1]
    context = pipeline._builder().build_context(hours)
    context.mean = context.baseline
    monkeypatch.setattr(pipeline, '_context', lambda _: context)
    model = SpatialBasis(2)
    phi = pipeline._basis_inputs(model)
    fit = np.array([True, True, False, False])
    coefficients, loss = pipeline._basis_fit(hours, phi, fit)
    loss.backward()
    assert torch.isfinite(coefficients).all()
    assert model.network[-1].weight.grad.abs().sum() > 0
    original_labels = pipeline._labels
    def perturbed(selected, station_mask=None):
        truth, valid, xy, height = original_labels(selected, station_mask)
        truth[:, ~torch.as_tensor(fit)] += 10000
        return truth, valid, xy, height
    monkeypatch.setattr(pipeline, '_labels', perturbed)
    changed, changed_loss = pipeline._basis_fit(hours, phi.detach(), fit)
    torch.testing.assert_close(changed, coefficients)
    torch.testing.assert_close(changed_loss, loss)


def test_campaign_pause_is_observed_inside_nested_stage(tmp_path):
    campaign_pause = tmp_path/'PAUSE'
    directory = tmp_path/'seed_42'/'search'/'gan'/'trial_0000'/'pretrain'
    engine = TrainEngine(torch.nn.Linear(1, 1), directory, [1],
                StageSpec('gan_pretrain', 1, 1, 1, 1, 42, {}), ScienceConfig(), pause_paths=[campaign_pause])
    campaign_pause.touch()
    result = engine.run(lambda *a: pytest.fail('campaign pause ignored'))
    assert result['status'] == 'PAUSED' and result['step'] == 0


@pytest.mark.parametrize('arm', ['det', 'gaussian', 'edm', 'gan'])
def test_functional_stage_dispatch_uses_distinct_saved_pretrain_and_observations(tmp_path, monkeypatch, arm):
    from types import SimpleNamespace
    from gnss_field.artifacts import model_digest
    from gnss_field.contracts import ContextBatch
    from gnss_field.objectives import CoefficientTransform
    from gnss_field.training.pipeline import TrainingPipeline
    from gnss_field.training.system import FrozenDecoder
    torch.set_num_threads(2)
    budgets = {**ScienceConfig().budgets, f'{arm}_pretrain': 1, f'{arm}_refine': 1}
    pipeline = TrainingPipeline(replace(ScienceConfig(), rank=2, budgets=budgets, log_every=1),
                 RuntimeConfig(device='cpu', micro_hours=32), tmp_path)
    store = SimpleNamespace(stats={'z0_std': 30., 'surface_std': 25.}, grid_id='grid',
                             manifest={'training_hash': 'training'})
    pipeline._manifest = SimpleNamespace(contract='manifest', select=lambda *a: np.arange(420768, 420776),
        masks=lambda hours: np.ones((len(hours), 4), dtype=bool))
    decoder = FrozenDecoder(torch.randn(576, 2), (24, 24),
                    CoefficientTransform(torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool)), 30.)
    pipeline._decoder = decoder
    monkeypatch.setattr(pipeline, '_store', lambda *a: store)
    monkeypatch.setattr(pipeline, '_decoder_bundle', lambda: {'hours': np.arange(420768, 420776)})
    monkeypatch.setattr(pipeline, '_foundation_dependencies', lambda: {'mean': 'fixed', 'basis': 'fixed'})
    monkeypatch.setattr(pipeline, '_shared_encoder', lambda m: model_digest(m.encoder))
    def context(hours):
        n = len(hours)
        return ContextBatch(hours=torch.as_tensor(hours), stations=torch.ones(n, 4, 17),
            padding=torch.zeros(n, 4, dtype=torch.bool), coarse=torch.zeros(n, 4, 15, 6, 6),
            maps=torch.ones(n, 6, 24, 24), baseline=torch.ones(n, 1, 24, 24)*100,
            mean=torch.ones(n, 1, 24, 24)*100)
    monkeypatch.setattr(pipeline, '_context', context)
    monkeypatch.setattr(pipeline, '_coefficients', lambda hours: torch.ones(len(hours), 2)*.2)
    monkeypatch.setattr(pipeline, '_labels', lambda hours: (torch.ones(len(hours), 4)*105,
        torch.ones(len(hours), 4, dtype=torch.bool), np.array([[8., 8.], [9., 8.], [8., 9.], [9., 9.]]), np.zeros(4)))
    def screen(*args):
        values = {'point': {'value': 1.}}
        if arm != 'det':
            values['joint'] = {'value': 1.}
        return lambda *_: values
    monkeypatch.setattr(pipeline, '_screen', screen)
    result = pipeline._functional_run(arm, tmp_path/arm, {'pretrain_lr': 1e-4, 'refinement_lr': 1e-4})
    pretrain = torch.load(result['pretrain_checkpoint'], weights_only=False)
    observation = torch.load(tmp_path/arm/'refine'/'last.pt', weights_only=False)
    assert pretrain['spec']['stage'] == f'{arm}_pretrain'
    assert observation['spec']['stage'] == f'{arm}_refine'
    assert pretrain['products'] == {}
    assert result['point']['checkpoint'] != result['pretrain_checkpoint']
    assert observation['spec']['dependencies']['pretraining']
    if arm == 'gan':
        assert pretrain['auxiliary_optimizers'][0]['state'][0]['step'] == 5
        assert observation['auxiliary_optimizers'] == []
