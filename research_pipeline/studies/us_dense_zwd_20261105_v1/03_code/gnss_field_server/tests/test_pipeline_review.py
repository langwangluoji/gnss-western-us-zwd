from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from gnss_field.artifacts import read_json, write_json
from gnss_field.contracts import RuntimeConfig, ScienceConfig
from gnss_field.training.engine import StageSpec, TrainEngine, atomic_torch
from gnss_field.training.pipeline import TrainingPipeline


def test_resumed_allocation_advances_after_prior_wall_exhaustion(tmp_path, monkeypatch):
    from gnss_field.training import engine as module
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    science = replace(ScienceConfig(), log_every=1)
    spec = StageSpec('det_pretrain', 2, 2, 1, 1, 42, {})
    calls = []
    def run():
        model = torch.nn.Linear(1, 1)
        engine = TrainEngine(model, tmp_path, [1, 2], spec, science, max_seconds=10)
        def objective(*args):
            calls.append(1)
            clock[0] += 11
            return {'loss': model(torch.ones(1, 1)).square().mean()}
        return engine.run(objective)
    first = run()
    assert first['status'] == 'BUDGET_EXHAUSTED' and first['step'] == 1
    second = run()
    assert second['status'] == 'COMPLETE' and second['step'] == 2
    assert second['elapsed_seconds'] == 22
    assert len(calls) == 2


def test_promoted_allocation_uses_new_wall_clock(tmp_path, monkeypatch):
    from gnss_field.training import engine as module
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    science = replace(ScienceConfig(), log_every=1)
    spec = StageSpec('edm_pretrain', 1, 2, 1, 1, 42, {})
    source_model = torch.nn.Linear(1, 1)
    source = TrainEngine(source_model, tmp_path/'source', [1, 2], spec, science, max_seconds=10)
    def objective(*args):
        clock[0] += 100
        return {'loss': source_model(torch.ones(1, 1)).square().mean()}
    source.run(objective)
    target = TrainEngine(torch.nn.Linear(1, 1), tmp_path/'target', [1, 2],
                         replace(spec, budget=2), science, max_seconds=10)
    pipeline = TrainingPipeline(science, RuntimeConfig(device='cpu'), tmp_path/'campaign')
    pipeline._fork(target, tmp_path/'source'/'last.pt')
    result = target.run(lambda *a: {'loss': target.model(torch.ones(1, 1)).square().mean()})
    assert result['status'] == 'COMPLETE' and result['step'] == 2
    assert result['elapsed_seconds'] == 100


@pytest.mark.parametrize('arm,role', [('mean_base', 'C'), ('det', 'D')])
def test_pipeline_evaluation_preserves_current_only_even_without_h_cache(legacy_fixture, tmp_path, monkeypatch, arm, role):
    from gnss_field.data import pack_legacy, PackedStore
    from gnss_field.objectives import CoefficientTransform
    from gnss_field.training.system import FrozenDecoder
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    store = PackedStore(tmp_path/'packed', purpose='audit')
    science = replace(ScienceConfig(), rank=2, use_history=False)
    pipeline = TrainingPipeline(science, RuntimeConfig(device='cpu', data_root=str(tmp_path/'packed')),
                                tmp_path/'campaign')
    hours = np.asarray(store.hours[20:22])
    monkeypatch.setattr(pipeline, '_hours', lambda *a: hours)
    model = pipeline._model(arm)
    checkpoint = tmp_path/f'{arm}.pt'
    atomic_torch(checkpoint, {'ema': model.state_dict()})
    seen = []
    if arm == 'mean_base':
        from gnss_field.training import prediction
        original = prediction.predict_patches
        def patches(net, context, *args, **kwargs):
            seen.append(int(torch.count_nonzero(context.coarse[:, 1:, :6])))
            return original(net, context, *args, **kwargs)
        monkeypatch.setattr(prediction, 'predict_patches', patches)
    else:
        monkeypatch.setattr(pipeline, '_model', lambda *a, **k: model)
        model.encoder.register_forward_pre_hook(lambda module, args: seen.append(
            int(torch.count_nonzero(args[0].coarse[:, 1:, :6]))))
        cache = SimpleNamespace(scenario_hash=pipeline.scenario.identity,
                                get=lambda h, **k: torch.ones(len(h), 1, 24, 24)*100)
        monkeypatch.setattr(pipeline, '_mean_cache', lambda: cache)
        decoder = FrozenDecoder(torch.randn(576, 2), (24, 24),
            CoefficientTransform(torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool)), 30.)
        monkeypatch.setattr(pipeline, '_get_decoder', lambda: decoder)
        monkeypatch.setattr(pipeline, '_condition_provider', lambda m: None)
    result = pipeline._evaluate(arm, checkpoint, tmp_path/'evaluation', full=True, store=store, role=role)
    assert seen == [0, 0]
    assert result['context_policy']['use_history'] is False


def test_evaluation_contract_binds_history_policy(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore
    from gnss_field.training.prediction import evaluate_predictor
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    store = PackedStore(tmp_path/'packed', purpose='audit')
    def predictor(hours, context, ids, members):
        return {'mean': np.zeros((len(hours), len(ids))), 'coarsened': np.zeros((len(hours), len(ids)))}
    evaluate_predictor(store, store.hours[20:21], predictor, tmp_path/'evaluation', use_history=False)
    with pytest.raises(ValueError, match='contract differs'):
        evaluate_predictor(store, store.hours[20:21], predictor, tmp_path/'evaluation', use_history=True)


def test_current_only_mean_screen_uses_masked_contexts(legacy_fixture, tmp_path, monkeypatch):
    from gnss_field.data import pack_legacy, PackedStore
    from gnss_field.training import pipeline as module, prediction
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    store = PackedStore(tmp_path/'packed', purpose='audit')
    pipeline = TrainingPipeline(replace(ScienceConfig(), use_history=False),
                 RuntimeConfig(device='cpu'), tmp_path/'campaign')
    monkeypatch.setattr(pipeline, '_store', lambda *a: store)
    monkeypatch.setattr(pipeline, '_hours', lambda *a: store.hours[20:22])
    monkeypatch.setattr(module, 'stratified_hours', lambda hours, year: hours)
    original, seen = prediction.predict_patches, []
    def patches(model, context, *args, **kwargs):
        seen.append(int(torch.count_nonzero(context.coarse[:, 1:, :6])))
        return original(model, context, *args, **kwargs)
    monkeypatch.setattr(prediction, 'predict_patches', patches)
    directory = tmp_path/'screen'
    pipeline._screen('mean_base', directory)(pipeline._model('mean_base'), 201)
    assert seen == [0, 0]
    assert read_json(directory/'eval_000201'/'metrics.json')['context_policy']['use_history'] is False


def _child_case(tmp_path, monkeypatch, seed, reverse=False):
    parent = TrainingPipeline(ScienceConfig(), RuntimeConfig(device='cpu'), tmp_path/'campaign')
    child = TrainingPipeline(ScienceConfig(), parent.runtime, parent.directory, seed)
    write_json(parent.root/'stability.json', {'members': 64})
    (child.root/'condition').mkdir(exist_ok=True)
    (child.root/'condition'/'frozen.pt').write_bytes(b'encoder')
    monkeypatch.setattr(child, '_foundation_dependencies', lambda: {'mean': 'm', 'basis': 'b'})
    configurations = {
        'edm': [dict(key='point_config', params={'pretrain_lr': 1e-4, 'refinement_lr': 1e-4}, products=['point']),
                dict(key='joint_config', params={'pretrain_lr': 2e-4, 'refinement_lr': 1e-5}, products=['joint'])],
        'gan': [dict(key='gan_config', params={'pretrain_lr': 1e-4, 'refinement_lr': 1e-4}, products=['point', 'joint'])]}
    monkeypatch.setattr(parent, '_locked_configurations', lambda arm: configurations[arm])
    trained = []
    for arm, configs in configurations.items():
        for config in configs:
            directory = child.root/arm/config['key']
            checkpoint = directory/'refine'/'candidate_001000.pt'
            atomic_torch(checkpoint, {'saved': True})
            atomic_torch(directory/'refine'/'last.pt', {'products': {'point': [dict(checkpoint=str(checkpoint), step=1000)],
                                                                       'joint': [dict(checkpoint=str(checkpoint), step=1000)]}})
            trained.append(dict(arm=arm, directory=str(directory), config=config))
    def evaluate(arm, checkpoint, output, **kwargs):
        # The opposite configuration/family is deliberately better for a product.
        point = .01 if 'joint_config' in str(checkpoint) else (10. if arm == 'edm' else 1.)
        joint = .01 if 'point_config' in str(checkpoint) else (1. if arm == 'edm' else 10.)
        if reverse:
            point, joint = 20-point, 20-joint
        return dict(values={'point': {'value': point}, 'joint': {'value': joint}}, metrics={})
    monkeypatch.setattr(child, '_evaluate', evaluate)
    return parent, child, trained


def test_auxiliary_selection_keeps_every_family_and_parent_product_configuration(tmp_path, monkeypatch):
    for seed, reverse in ((17, False), (101, True)):
        parent, child, trained = _child_case(tmp_path, monkeypatch, seed, reverse)
        summary = parent._select_child_products(child, trained)
        selected = summary['selected_products']
        assert set(selected) == {'edm', 'gan'}
        assert selected['edm']['point']['configuration_key'] == 'point_config'
        assert selected['edm']['joint']['configuration_key'] == 'joint_config'
        assert selected['gan']['point']['model'] == selected['gan']['joint']['model'] == 'gan'


def test_auxiliary_selection_rejects_changed_parent_hyperconfiguration(tmp_path, monkeypatch):
    parent, child, trained = _child_case(tmp_path, monkeypatch, 17)
    trained = copy.deepcopy(trained)
    trained[0]['config']['params']['pretrain_lr'] = 7e-4
    with pytest.raises(ValueError, match='parent locked hyperconfiguration'):
        parent._select_child_products(child, trained)


def test_pretrain_ablation_keeps_fixed_parent_family_and_configuration(tmp_path, monkeypatch):
    parent, child, trained = _child_case(tmp_path, monkeypatch, 17)
    (parent.root/'condition').mkdir(exist_ok=True)
    (parent.root/'condition'/'frozen.pt').write_bytes(b'encoder')
    configurations = {}
    for item in trained:
        config, arm = copy.deepcopy(item['config']), item['arm']
        source = parent.root/'parent_pretrain'/arm/config['key']/'last.pt'
        atomic_torch(source, {'spec': {'stage': f'{arm}_pretrain'}, 'step': 1000})
        config['pretrain_checkpoint'] = str(source)
        configurations.setdefault(arm, []).append(config)
    monkeypatch.setattr(parent, '_advanced', lambda: {'edm': {}, 'gan': {}})
    monkeypatch.setattr(parent, '_locked_configurations', lambda arm: configurations[arm])
    monkeypatch.setattr(parent, '_foundation_dependencies', lambda: {'mean': 'm', 'basis': 'b'})
    monkeypatch.setattr(parent, '_evaluate', lambda *a, **k: {'values': {'point': {'value': 1.}, 'joint': {'value': 1.}}, 'metrics': {}})
    result = parent.ablation('pretrain_only')
    selected = read_json(result['results'])['selected_products']
    assert set(selected) == {'edm', 'gan'}
    assert selected['edm']['point']['configuration_key'] == 'point_config'
    assert selected['edm']['joint']['configuration_key'] == 'joint_config'
    for arm, products in selected.items():
        for product, row in products.items():
            config = next(c for c in configurations[arm] if product in c['products'])
            assert row['checkpoint'] == str(Path(config['pretrain_checkpoint']).resolve())


def test_auxiliary_lock_checks_family_coverage_and_parent_configuration(tmp_path, monkeypatch):
    from gnss_field.artifacts import sha256
    parent, child, trained = _child_case(tmp_path, monkeypatch, 17)
    summary = parent._select_child_products(child, trained)
    configurations = {}
    for entry in trained:
        arm, config = entry['arm'], copy.deepcopy(entry['config'])
        checkpoint = Path(entry['directory'])/'refine'/'candidate_001000.pt'
        config.update(pretrain_checkpoint=str(checkpoint), pretrain_checkpoint_sha256=sha256(checkpoint),
                      selected_checkpoints={p: {'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint)} for p in config['products']})
        configurations.setdefault(arm, []).append(config)
    lock = dict(family_configurations=configurations, replicates={'17': summary}, ablations={})
    monkeypatch.setattr(TrainingPipeline, '_foundation_dependencies', lambda *a: {'mean': 'm', 'basis': 'b'})
    parent._validate_auxiliary(lock)
    missing = copy.deepcopy(lock)
    del missing['replicates']['17']['selected_products']['gan']
    with pytest.raises(ValueError, match='family coverage differs'):
        parent._validate_auxiliary(missing)
    changed = copy.deepcopy(lock)
    changed['replicates']['17']['selected_products']['edm']['point']['configuration_key'] = 'joint_config'
    with pytest.raises(ValueError, match='parent configuration differs'):
        parent._validate_auxiliary(changed)


def test_auxiliary_retrospective_dispatch_keeps_every_family(tmp_path, monkeypatch):
    from gnss_field.training import pipeline as module
    parent, child, trained = _child_case(tmp_path, monkeypatch, 17)
    selected = {arm: {product: dict(model=arm, checkpoint=f'{arm}_{product}.pt', configuration_key='locked')
                     for product in ('point', 'joint')} for arm in ('edm', 'gan')}
    auxiliary = dict(science=vars(child.science), directory=str(child.directory), seed=17,
                     selected_products=selected)
    lock = {'replicates': {'17': auxiliary}, 'ablations': {'current_only': auxiliary}, 'stability': {'members': 64}}
    calls = []
    monkeypatch.setattr(TrainingPipeline, '_mean_cache', lambda self: SimpleNamespace(build=lambda *a, **k: None))
    monkeypatch.setattr(TrainingPipeline, '_mean_model', lambda *a: None)
    monkeypatch.setattr(TrainingPipeline, '_builder', lambda *a: None)
    monkeypatch.setattr(TrainingPipeline, '_hours', lambda *a: np.array([438288]))
    def evaluate(self, arm, checkpoint, output, **kwargs):
        calls.append((arm, checkpoint, str(output)))
        return {'metrics': {}}
    monkeypatch.setattr(TrainingPipeline, '_evaluate', evaluate)
    monkeypatch.setattr(TrainingPipeline, '_formal_evaluation', lambda *a: {})
    monkeypatch.setattr(module, '_arrays', lambda *a: {})
    result = parent._evaluate_auxiliary(lock, object(), tmp_path/'evaluation', {}, {})
    assert len(calls) == 8
    assert set(result['replicates_17']['families']) == {'edm', 'gan'}
    assert set(result['ablations_current_only']['families']) == {'edm', 'gan'}


def test_runtime_hot_cache_root_is_used_and_history_specific(legacy_fixture, tmp_path, monkeypatch):
    from gnss_field.data import pack_legacy
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    hot = tmp_path/'explicit_hot_cache'
    checkpoint = tmp_path/'mean.pt'
    checkpoint.write_bytes(b'mean')
    roots = []
    for use_history in (True, False):
        pipeline = TrainingPipeline(replace(ScienceConfig(), use_history=use_history),
             RuntimeConfig(device='cpu', data_root=str(tmp_path/'packed'), cache_root=str(hot)),
             tmp_path/f'campaign_{use_history}')
        monkeypatch.setattr(pipeline, '_mean_path', lambda *a: checkpoint)
        monkeypatch.setattr(pipeline, '_foundation_dependencies', lambda: {'mean': 'm', 'basis': 'b'})
        roots.append(pipeline._mean_cache().root)
        assert roots[-1].is_relative_to(hot)
        assert pipeline._condition_cache_root(pipeline._model('det')).is_relative_to(hot)
    assert roots[0] != roots[1]


def test_pipeline_stability_supplies_matching_noise_provenance(tmp_path, monkeypatch):
    from gnss_field.training import pipeline as module
    pipeline = TrainingPipeline(ScienceConfig(), RuntimeConfig(device='cpu'), tmp_path)
    checkpoint = tmp_path/'candidate.pt'
    checkpoint.write_bytes(b'checkpoint')
    original = np.ones((2, 128, 4), dtype='f4')*1000
    updated = original.copy()
    updated[:, :64] += np.float32(.0002)
    record = pipeline._record('edm', checkpoint, tmp_path/'K64',
                   {'values': {'point': {'value': 1.}, 'joint': {'value': 1.}}, 'metrics': {}}, 1000)
    write_json(pipeline.root/'full2019.json', {'records': [record]})
    monkeypatch.setattr(pipeline, '_direction_decisions', lambda *a: {'families': {}})
    monkeypatch.setattr(pipeline, '_evaluate', lambda *a, **k: {'values': {'point': {'value': 1.001}, 'joint': {'value': 1.}}, 'metrics': {}})
    monkeypatch.setattr(pipeline, '_store', lambda *a: SimpleNamespace(
        grid_id='grid', stats={'input_std': [1.]}, manifest={'files': {'A/x': {'sha256': 'a'}}, 'fingerprint': 'packed'}))
    monkeypatch.setattr(pipeline, '_foundation_dependencies', lambda: {'mean': 'm', 'basis': 'b'})
    monkeypatch.setattr(module, '_arrays', lambda path: {'samples': original[:, :64] if Path(path).name == 'K64' else updated,
                            'hours': np.array([429552, 429553]), 'station_ids': np.arange(4)})
    pipeline.stability()
    result = read_json(pipeline.root/'stability.json')
    assert result['stable'] and result['members'] == 64
    check = result['checks'][0]
    assert check['member_ids_match'] and check['noise_provenance_matches']
    assert check['prefix_tolerance_mm'] == .001 and check['tolerance_mm'] == .005
