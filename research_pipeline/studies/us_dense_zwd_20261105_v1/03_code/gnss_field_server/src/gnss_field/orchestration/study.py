"""Sequential, restartable Optuna studies with product-specific promotion references."""
from pathlib import Path
from typing import Callable
from collections.abc import Mapping
import math
import pickle
import hashlib

from ..artifacts import atomic_bytes, immutable_json, read_json, sha256, write_json
from .state import single_owner

PRODUCTS = {'det': ('point',), 'm2_f': ('point', 'marginal'),
            'gaussian': ('point', 'joint'), 'edm': ('point', 'joint'), 'gan': ('point', 'joint')}


def _validate_result(result, products, *, validate_references, reference_validator):
    if not isinstance(result, Mapping):
        raise ValueError('Objective must return a product mapping')
    normalized = {}
    references = []
    # Check the pretraining source first, since every promotion depends on it.
    for role in ('pretrain_checkpoint', 'contract'):
        value = result.get(role)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Missing {role} reference')
        normalized[role] = Path(value).as_posix()
        references.append((role, Path(value)))
    for product in products:
        item = result.get(product)
        if not isinstance(item, Mapping):
            raise ValueError(f'Missing objective product {product}')
        value = item.get('value')
        try:
            valid = not isinstance(value, bool) and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f'Nonfinite objective {product}')
        checkpoint = item.get('checkpoint')
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f'Missing {product} checkpoint reference')
        normalized[product] = {'value': float(value), 'checkpoint': Path(checkpoint).as_posix()}
        references.append((product, Path(checkpoint)))
    if validate_references:
        hashes = {}
        for role, path in references:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing saved {role} artifact: {path}')
            if reference_validator is not None:
                reference_validator(path, role)
            elif role == 'contract':
                contract = read_json(path)
                if (not isinstance(contract, dict) or not isinstance(contract.get('signature'), str) or
                        not isinstance(contract.get('spec'), dict)):
                    raise ValueError('Scientific contract must contain signature and spec metadata')
            else:
                import torch
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if (not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get('signature'), str) or
                        not isinstance(checkpoint.get('spec'), Mapping) or 'model' not in checkpoint):
                    raise ValueError(f'Invalid saved {role} checkpoint contents')
                if role == 'pretrain_checkpoint':
                    stage = checkpoint['spec'].get('stage', '')
                    if not isinstance(stage, str) or not stage.endswith('_pretrain'):
                        raise ValueError('Promotion source must be a saved pretraining checkpoint')
            hashes[role] = {'path': path.resolve().as_posix(), 'sha256': sha256(path)}
        normalized['reference_hashes'] = hashes
    return normalized


def _error_stop(trials) -> bool:
    terminal = [trial for trial in trials if trial.state.name in {'COMPLETE', 'FAIL', 'PRUNED'}]
    last = terminal[-3:]
    return (len(last) == 3 and all(trial.state.name == 'FAIL' for trial in last) and
            bool(last[0].user_attrs.get('error')) and
            len({trial.user_attrs.get('error') for trial in last}) == 1)


def _report(study, root, arm, products, budget):
    trials = study.get_trials(deepcopy=False)
    complete = [trial for trial in trials if trial.state.name == 'COMPLETE']
    for trial in complete:
        for reference in trial.user_attrs['result'].get('reference_hashes', {}).values():
            path = Path(reference['path'])
            if not path.is_file() or sha256(path) != reference['sha256']:
                raise ValueError('Completed study checkpoint or contract artifact hash differs')
    selected = {}
    for product in products:
        if complete:
            winner = min(complete, key=lambda trial: (trial.user_attrs['result'][product]['value'], trial.number))
            result = winner.user_attrs['result']
            selected[product] = {'trial_number': winner.number, **result[product],
                                 'params': dict(winner.params), 'pretrain_checkpoint': result['pretrain_checkpoint'],
                                 'contract': result['contract'],
                                 'trial_directory': (root / 'trials' / f'trial_{winner.number:04d}').as_posix()}
    promotions = []
    for number in sorted({entry['trial_number'] for entry in selected.values()}):
        entry = next(value for value in selected.values() if value['trial_number'] == number)
        promotions.append({'trial_number': number, 'params': entry['params'],
                           'pretrain_checkpoint': entry['pretrain_checkpoint'], 'contract': entry['contract'],
                           'trial_directory': entry['trial_directory'],
                           'products': [product for product, value in selected.items() if value['trial_number'] == number]})
    failed = sum(trial.state.name == 'FAIL' for trial in trials)
    running = sum(trial.state.name == 'RUNNING' for trial in trials)
    state = 'COMPLETE' if len(complete) + failed >= budget else 'INCOMPLETE'
    if _error_stop(trials):
        state = 'STOPPED_REPEATED_ERROR'
    report = {'arm': arm, 'state': state, 'completed': len(complete), 'failed': failed, 'running': running,
              'selected': selected, 'promotions': promotions,
              'trials': [{'number': trial.number, 'state': trial.state.name, 'params': dict(trial.params),
                          'error': trial.user_attrs.get('error')} for trial in trials]}
    write_json(root / 'report.json', report)
    write_json(root / 'status.json', {'state': state, 'completed': len(complete), 'failed': failed,
                                     'running': running})
    return report


def run_study(directory, arm: str, objective: Callable, *, trials=8, seed=20261117, startup=4,
              validate_references=False, reference_validator: Callable | None = None) -> dict:
    """Run or resume one JournalStorage owner and at most ``trials`` total attempts.

    ``objective(params, trial_directory, trial_number)`` returns point and joint
    (or marginal for m2_f) product dictionaries, each with finite value and saved
    checkpoint reference, plus pretrain_checkpoint and contract references.
    det has one point objective. Baseline trial zero uses both rates at 1e-4.
    KeyboardInterrupt leaves the same RUNNING trial ready to resume. Product
    promotions always carry the saved pretraining source. Production callers
    enable reference validation and may supply their checkpoint-content validator.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.trial import TrialState

    if arm not in PRODUCTS or not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError('Unknown arm or invalid trial budget')
    if not isinstance(startup, int) or startup < 1:
        raise ValueError('TPE startup must be positive')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    products = PRODUCTS[arm]
    with single_owner(root / '.owner.lock'):
        immutable_json(root / 'study.json', {'arm': arm, 'trials': trials, 'seed': seed, 'startup': startup,
                                            'products': products, 'validate_references': validate_references})
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup)
        sampler_path = root / 'sampler.state'
        if sampler_path.exists():
            digest, separator, payload = sampler_path.read_bytes().partition(b'\n')
            if not separator or hashlib.sha256(payload).hexdigest().encode('ascii') != digest:
                raise ValueError('Persisted TPE sampler hash differs')
            # This file is created only by the single study owner in its private artifact directory.
            sampler = pickle.loads(payload)
        elif (root / 'sampler.pkl').exists():
            # Existing valid studies migrate to the single atomic container on their next trial.
            legacy = root / 'sampler.pkl'
            sidecar = root / 'sampler.json'
            if not sidecar.exists() or sha256(legacy) != read_json(sidecar)['sha256']:
                raise ValueError('Persisted TPE sampler hash differs')
            sampler = pickle.loads(legacy.read_bytes())
        study = optuna.create_study(storage=JournalStorage(JournalFileBackend(str(root / 'journal.log'))),
                                    study_name=arm, directions=['minimize'] * len(products),
                                    sampler=sampler, load_if_exists=True)
        if not study.get_trials():
            study.enqueue_trial({'pretrain_lr': 1e-4, 'refinement_lr': 1e-4})
        while True:
            known = study.get_trials(deepcopy=False)
            if _error_stop(known):
                break
            running = [trial for trial in known if trial.state == TrialState.RUNNING]
            if len(running) > 1:
                raise ValueError('A sequential study has multiple RUNNING trials')
            if running:
                trial = optuna.Trial(study, running[0]._trial_id)
            elif sum(trial.state in {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED} for trial in known) >= trials:
                break
            else:
                trial = study.ask()
            params = {'pretrain_lr': trial.suggest_float('pretrain_lr', 3e-5, 3e-4, log=True),
                      'refinement_lr': trial.suggest_float('refinement_lr', 1e-5, 1e-4, log=True)}
            payload = pickle.dumps(study.sampler)
            digest = hashlib.sha256(payload).hexdigest().encode('ascii')
            atomic_bytes(sampler_path, digest + b'\n' + payload)
            trial_root = root / 'trials' / f'trial_{trial.number:04d}'
            trial_root.mkdir(parents=True, exist_ok=True)
            immutable_json(trial_root / 'params.json', params)
            result_path = trial_root / 'result.json'
            write_json(root / 'status.json', {'state': 'RUNNING', 'trial_number': trial.number})
            try:
                if trial.user_attrs.get('error'):
                    study.tell(trial, state=TrialState.FAIL)
                    continue
                if result_path.exists():
                    result = read_json(result_path)
                else:
                    raw = objective(dict(params), trial_root, trial.number)
                    result = _validate_result(raw, products, validate_references=validate_references,
                                              reference_validator=reference_validator)
                    immutable_json(result_path, result)
                trial.set_user_attr('result', result)
                study.tell(trial, values=[result[product]['value'] for product in products])
            except KeyboardInterrupt:
                write_json(root / 'status.json', {'state': 'PAUSED', 'trial_number': trial.number,
                                                 'trial_state': 'RUNNING'})
                raise
            except Exception as error:
                message = f'{type(error).__module__}.{type(error).__name__}: {error}'
                trial.set_user_attr('error', message)
                immutable_json(trial_root / 'error.json', {'error': message})
                study.tell(trial, state=TrialState.FAIL)
        return _report(study, root, arm, products, trials)
