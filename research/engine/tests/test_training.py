import copy
import torch
import numpy as np


def test_actual_edm_optimizer_and_cursor_resume_is_exact(tmp_path):
    from gnss_field.models.diffusion import EDMHead
    from gnss_field.objectives import edm_loss
    from gnss_field.training.engine import TrainEngine, StageSpec
    from gnss_field.contracts import ScienceConfig
    torch.set_num_threads(2)
    cond = torch.arange(24*128).reshape(24, 128).float()/5000
    truth = torch.arange(24*4).reshape(24, 4).float()/100
    cfg = ScienceConfig(evaluate_every=50, startup_evaluation=20, log_every=100)
    def run(where, stop=None):
        torch.manual_seed(42)
        model = EDMHead(4, width=16)
        spec = StageSpec('edm_pretrain', 200, 200, 4, 2, 17, {'dataset': 'fixed_synthetic_for_resume'})
        def objective(hours, net, step):
            idx = torch.as_tensor(hours)
            return {'loss': edm_loss(net, truth[idx], cond[idx]), 'labels': len(hours)*4}
        engine = TrainEngine(model, where, np.arange(24), spec, cfg)
        result = engine.run(objective, stop_after=stop)
        return engine, result
    continuous, _ = run(tmp_path/'continuous')
    _, partial = run(tmp_path/'split', 100)
    assert partial['status'] == 'PAUSED'
    resumed, result = run(tmp_path/'split')
    assert result['step'] == 200
    for name, value in continuous.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    assert np.array_equal(continuous.cursor.order, resumed.cursor.order)
    assert continuous.cursor.position == resumed.cursor.position


def test_failed_evaluation_is_replayed_without_repeating_update(tmp_path):
    import pytest
    from gnss_field.models.heads import DeterministicHead
    from gnss_field.training.engine import TrainEngine, StageSpec
    from gnss_field.contracts import ScienceConfig
    cfg = ScienceConfig(evaluate_every=2, startup_evaluation=1, log_every=1)
    def create():
        torch.manual_seed(3)
        return TrainEngine(DeterministicHead(4, width=16), tmp_path, np.arange(4),
                           StageSpec('det_refine', 4, 4, 2, 1, 5, {'test': 'pending'}), cfg)
    def loss(hours, model, step):
        return {'loss': model(torch.ones(len(hours), 128)).square().mean()}
    def bad(model, step):
        raise RuntimeError('evaluation interrupted')
    with pytest.raises(RuntimeError):
        create().run(loss, evaluator=bad)
    saved = torch.load(tmp_path/'last.pt', weights_only=False)
    assert saved['step'] == 1 and saved['pending_evaluation']
    seen = []
    def evaluate(model, step):
        seen.append(step)
        return {'point': {'value': 1./step}}
    e = create()
    result = e.run(loss, evaluator=evaluate)
    assert seen[0] == 1 and result['step'] == 4
    assert result['products']['point'][0]['step'] == 4
def test_pending_evaluation_waits_when_remaining_budget_is_zero(tmp_path):
    from gnss_field.training.engine import TrainEngine, StageSpec
    from gnss_field.contracts import ScienceConfig
    engine = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8),
        StageSpec('budget_refine', 4, 4, 2, 2, 42, {}), ScienceConfig(), max_seconds=0)
    engine.pending_evaluation = True
    def forbidden(*args):
        raise AssertionError('No update or pending evaluation may start at zero budget')
    result = engine.run(forbidden, evaluator=forbidden)
    assert result['status'] == 'BUDGET_EXHAUSTED'
    assert engine.pending_evaluation and engine.step == 0

def test_candidate_commit_interruption_replays_without_duplicate_selection(tmp_path, monkeypatch):
    import pytest
    import gnss_field.training.engine as module
    from gnss_field.contracts import ScienceConfig
    from gnss_field.training.observable import _same
    cfg = ScienceConfig(evaluate_every=2, startup_evaluation=0, log_every=100, patience=99)
    spec = module.StageSpec('commit_refine', 4, 4, 2, 2, 42, {})
    def create(path):
        torch.manual_seed(7)
        return module.TrainEngine(torch.nn.Linear(1, 1), path, np.arange(8), spec, cfg)
    def objective(hours, model, step):
        return {'loss': model(torch.randn(len(hours), 1)).square().mean()}
    def evaluate(model, step):
        return {'joint': {'value': 1./step}}
    continuous = create(tmp_path/'continuous')
    continuous.run(objective, evaluator=evaluate)
    original = module.atomic_torch
    interrupted = [False]
    def fail_after_candidate(path, value):
        original(path, value)
        if path.name == 'candidate_000002.pt' and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError('after candidate commit before last commit')
    monkeypatch.setattr(module, 'atomic_torch', fail_after_candidate)
    with pytest.raises(RuntimeError):
        create(tmp_path/'split').run(objective, evaluator=evaluate)
    resumed = create(tmp_path/'split')
    resumed.run(objective, evaluator=evaluate)
    for key in ('model', 'ema', 'optimizers', 'rng', 'cursor', 'step', 'stagnant'):
        _same(continuous.state()[key], resumed.state()[key])
    assert [(r['step'], r['value']) for r in resumed.products['joint']] == [(4, .25), (2, .5)]


def test_early_stopped_COMPLETE_is_terminal_on_restart(tmp_path):
    from gnss_field.training.engine import TrainEngine, StageSpec
    from gnss_field.contracts import ScienceConfig
    cfg = ScienceConfig(evaluate_every=1, startup_evaluation=0, minimum_refinement_steps=1, patience=1)
    spec = StageSpec('terminal_refine', 8, 8, 2, 2, 42, {})
    engine = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8), spec, cfg)
    engine.run(lambda h,m,s: {'loss': m(torch.ones(len(h),1)).square().mean()},
               evaluator=lambda m,s: {'joint': {'value': 1.}})
    assert engine.phase == 'COMPLETE' and engine.step == 2
    restored = TrainEngine(torch.nn.Linear(1, 1), tmp_path, np.arange(8), spec, cfg)
    def forbidden(*args):
        raise AssertionError('A completed stage must not update or evaluate again')
    restored.run(forbidden, evaluator=forbidden)
    assert restored.step == 2 and restored.phase == 'COMPLETE'

