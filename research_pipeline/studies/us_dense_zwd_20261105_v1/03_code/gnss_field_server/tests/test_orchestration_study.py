from pathlib import Path
import math

import optuna
import pytest

from gnss_field.artifacts import read_json


def run_study(*args, **kwargs):
    from gnss_field.orchestration import run_study as run
    return run(*args, **kwargs)


def result(number, directory, *, point=None, joint=None, marginal=False):
    product = 'marginal' if marginal else 'joint'
    return {'point': {'value': float(number + 1 if point is None else point),
                      'checkpoint': str(directory / 'point.pt')},
            product: {'value': float(10 - number if joint is None else joint),
                      'checkpoint': str(directory / f'{product}.pt')},
            'pretrain_checkpoint': str(directory / 'pretrain.pt'),
            'contract': str(directory / 'contract.json')}


def test_completed_trials_are_not_repeated_and_dual_selection_keeps_checkpoint_association(tmp_path):
    calls = []

    def objective(params, directory, number):
        calls.append((params, directory, number))
        return result(number, directory)

    first = run_study(tmp_path, 'edm', objective, trials=4)
    second = run_study(tmp_path, 'edm', objective, trials=4)
    assert [row[2] for row in calls] == [0, 1, 2, 3]
    assert calls[0][0] == {'pretrain_lr': 1e-4, 'refinement_lr': 1e-4}
    for params, directory, number in calls:
        assert 3e-5 <= params['pretrain_lr'] <= 3e-4
        assert 1e-5 <= params['refinement_lr'] <= 1e-4
        assert directory == tmp_path / 'trials' / f'trial_{number:04d}'
    assert first == second
    assert second['selected']['point']['trial_number'] == 0
    assert second['selected']['point']['checkpoint'].endswith('trial_0000/point.pt')
    assert second['selected']['joint']['trial_number'] == 3
    assert second['selected']['joint']['checkpoint'].endswith('trial_0003/joint.pt')
    assert len(second['promotions']) == 2
    assert all(p['pretrain_checkpoint'].endswith('pretrain.pt') for p in second['promotions'])
    assert (tmp_path / 'journal.log').is_file()


def test_keyboard_interrupt_leaves_running_and_resume_reuses_number_params_and_path(tmp_path):
    calls = []

    def interrupted(params, directory, number):
        calls.append((dict(params), directory, number))
        (directory / 'checkpoint.pt').write_bytes(b'checkpoint')
        raise KeyboardInterrupt('planned pause')

    with pytest.raises(KeyboardInterrupt):
        run_study(tmp_path, 'det', interrupted, trials=2)
    assert read_json(tmp_path / 'status.json')['state'] == 'PAUSED'

    def resumed(params, directory, number):
        calls.append((dict(params), directory, number))
        return result(number, directory)

    finished = run_study(tmp_path, 'det', resumed, trials=2)
    assert calls[0] == calls[1]
    assert [row[2] for row in calls] == [0, 0, 1]
    assert finished['completed'] == 2
    assert set(finished['selected']) == {'point'}


def test_failed_trials_count_against_budget_and_three_identical_errors_stop(tmp_path):
    calls = []

    def broken(params, directory, number):
        calls.append(number)
        raise ValueError('same recoverable model failure')

    report = run_study(tmp_path, 'gan', broken, trials=8)
    assert calls == [0, 1, 2]
    assert report['failed'] == 3
    assert report['state'] == 'STOPPED_REPEATED_ERROR'
    again = run_study(tmp_path, 'gan', broken, trials=8)
    assert calls == [0, 1, 2]
    assert again == report


def test_nonfinite_scores_fail_trials_and_never_become_bad_scores(tmp_path):
    def objective(params, directory, number):
        return result(number, directory, point=math.nan if number == 0 else 1.)

    report = run_study(tmp_path, 'gaussian', objective, trials=2)
    assert report['failed'] == 1
    assert report['completed'] == 1
    assert report['selected']['point']['trial_number'] == 1
    assert report['trials'][0]['state'] == 'FAIL'


def test_multiobjective_never_reports_or_prunes_and_marginal_product_is_selected(tmp_path, monkeypatch):
    def unsupported(*args, **kwargs):
        pytest.fail('multiobjective report or pruning was called')

    monkeypatch.setattr(optuna.Trial, 'report', unsupported)
    monkeypatch.setattr(optuna.Trial, 'should_prune', unsupported)
    report = run_study(tmp_path, 'm2_f',
                       lambda params, path, number: result(number, path, marginal=True), trials=3)
    assert set(report['selected']) == {'point', 'marginal'}
    assert report['selected']['marginal']['checkpoint'].endswith('marginal.pt')


def test_production_reference_validation_fails_missing_saved_pretraining(tmp_path):
    report = run_study(tmp_path, 'edm', lambda p, d, n: result(n, d),
                       trials=1, validate_references=True)
    assert report['failed'] == 1
    assert report['promotions'] == []
    assert 'pretrain' in report['trials'][0]['error']


def test_existing_trial_artifact_cannot_be_overwritten_on_resume(tmp_path):
    def objective(params, directory, number):
        return result(number, directory)

    run_study(tmp_path, 'gaussian', objective, trials=1)
    path = tmp_path / 'trials' / 'trial_0000' / 'result.json'
    content = path.read_bytes()
    run_study(tmp_path, 'gaussian', objective, trials=1)
    assert path.read_bytes() == content
    with pytest.raises(ValueError, match='Immutable'):
        run_study(tmp_path, 'det', objective, trials=1)


def test_production_checkpoint_validation_accepts_registered_stage_and_hashes(tmp_path):
    import torch
    from gnss_field.artifacts import write_json

    def objective(params, directory, number):
        pretrain = {'signature': 'abc', 'spec': {'stage': 'edm_pretrain'}, 'phase': 'COMPLETE',
                    'model': {'weight': torch.tensor(1.)}}
        refined = dict(pretrain, signature='def', spec={'stage': 'edm_refine'})
        torch.save(pretrain, directory / 'pretrain.pt')
        torch.save(refined, directory / 'point.pt')
        torch.save(refined, directory / 'joint.pt')
        write_json(directory / 'contract.json', {'signature': 'abc', 'spec': {'stage': 'edm_pretrain'}})
        return result(number, directory)

    report = run_study(tmp_path, 'edm', objective, trials=1, validate_references=True)
    assert report['completed'] == 1
    assert report['failed'] == 0
    saved = read_json(tmp_path / 'trials' / 'trial_0000' / 'result.json')
    assert len(saved['reference_hashes']['pretrain_checkpoint']['sha256']) == 64


def test_production_checkpoint_validation_rejects_refinement_as_promotion_source(tmp_path):
    import torch
    from gnss_field.artifacts import write_json

    def objective(params, directory, number):
        checkpoint = {'signature': 'abc', 'spec': {'stage': 'edm_refine'}, 'phase': 'pretrain',
                      'model': {'weight': torch.tensor(1.)}}
        for name in ('pretrain', 'point', 'joint'):
            torch.save(checkpoint, directory / f'{name}.pt')
        write_json(directory / 'contract.json', {'signature': 'abc', 'spec': {'stage': 'edm_refine'}})
        return result(number, directory)

    report = run_study(tmp_path, 'edm', objective, trials=1, validate_references=True)
    assert report['failed'] == 1
    assert report['promotions'] == []


def test_completed_production_references_are_rechecked_before_promotion(tmp_path):
    import torch
    from gnss_field.artifacts import write_json

    def objective(params, directory, number):
        checkpoint = {'signature': 'abc', 'spec': {'stage': 'edm_pretrain'}, 'model': {}}
        for name in ('pretrain', 'point', 'joint'):
            torch.save(checkpoint, directory / f'{name}.pt')
        write_json(directory / 'contract.json', {'signature': 'abc', 'spec': {'stage': 'edm_pretrain'}})
        return result(number, directory)

    run_study(tmp_path, 'edm', objective, trials=1, validate_references=True)
    (tmp_path / 'trials' / 'trial_0000' / 'pretrain.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        run_study(tmp_path, 'edm', objective, trials=1, validate_references=True)


def test_interrupt_after_sampler_commit_can_resume_same_running_trial(tmp_path, monkeypatch):
    import gnss_field.orchestration.study as study_module
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    original = study_module.atomic_bytes
    interrupted = False

    def commit_then_interrupt(path, data):
        nonlocal interrupted
        original(path, data)
        if Path(path).name.startswith('sampler') and not interrupted:
            interrupted = True
            raise KeyboardInterrupt('interrupt immediately after durable sampler commit')

    monkeypatch.setattr(study_module, 'atomic_bytes', commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_study(tmp_path, 'edm', lambda p, d, n: result(n, d), trials=2)
    storage = JournalStorage(JournalFileBackend(str(tmp_path / 'journal.log')))
    saved = optuna.load_study(study_name='edm', storage=storage).get_trials()[0]
    assert saved.state.name == 'RUNNING'
    calls = []

    def resumed(params, directory, number):
        calls.append((dict(params), directory, number))
        return result(number, directory)

    finished = run_study(tmp_path, 'edm', resumed, trials=2)
    assert finished['completed'] == 2
    assert [call[2] for call in calls] == [0, 1]
    assert calls[0][0] == saved.params
    assert calls[0][1] == tmp_path / 'trials' / 'trial_0000'
