"""Resumable fixed-budget training and sequential Optuna studies."""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import random
import time
import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class Cursor:
    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices):
            raise ValueError('Empty training split')
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.indices)
        self.position = 0

    def take(self, n):
        result = []
        while n:
            count = min(n, len(self.order) - self.position)
            result.append(self.order[self.position:self.position + count])
            self.position += count
            n -= count
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.position = 0
        return np.concatenate(result)

    def state_dict(self):
        return dict(order=self.order.copy(), position=self.position, rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        if not np.array_equal(np.sort(state['order']), np.sort(self.indices)):
            raise ValueError('Training indices changed on resume')
        self.order = state['order'].copy()
        self.position = state['position']
        self.rng.bit_generator.state = state['rng']


def train_updates(model, data, directory, config, seed, updates, eval_every=2000,
                  evaluator=None, trial=None, checkpoint_every=200, microbatch=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(config=config, seed=seed, data=fingerprint(data.manifest), arm=model.arm,
                    effective_batch=32, precision='float32')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    cursor = Cursor(data.train_indices, seed)
    step = 0
    best = float('inf')
    elapsed = 0.
    last = directory/'last.pt'
    if last.exists():
        state = load_checkpoint(last)
        if state['contract'] != contract:
            raise ValueError('Checkpoint contract mismatch')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        cursor.load_state_dict(state['cursor'])
        restore_rng(state['rng'])
        step, best, elapsed = state['step'], state['best'], state['seconds']
        microbatch = state['microbatch']
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()

    def save(path):
        atomic_checkpoint(path, dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), cursor=cursor.state_dict(), rng=rng_state(),
            step=step, best=best, microbatch=microbatch, contract=contract,
            seconds=elapsed + time.perf_counter()-started))

    while step < updates:
        indices = cursor.take(32)
        before = rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total = 0.
            try:
                for start in range(0, 32, microbatch):
                    batch = data.batch(indices[start:start+microbatch])
                    loss = model.loss(batch, p_mean=config.get('p_mean', -1.2), p_std=config.get('p_std', 1.2))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss * (microbatch / 32)).backward()
                    total += float(loss.detach()) * (microbatch / 32)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if microbatch <= 2:
                    raise
                microbatch //= 2
                restore_rng(before)
                torch.cuda.empty_cache()
                with (directory/'hardware.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(step=step, microbatch=microbatch, accumulation=32//microbatch))+'\n')
        # AdamW can mutate parameters before an allocation failure. Never retry
        # optimizer.step() as if a failed forward/backward were atomic.
        optimizer.step()
        step += 1
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1-config['ema'])
            for eb, b in zip(ema.buffers(), model.buffers()):
                eb.copy_(b)
        if step % 100 == 0 or step == updates:
            record = dict(step=step, loss=total, grad_norm=float(norm),
                          seconds=elapsed+time.perf_counter()-started, microbatch=microbatch)
            with (directory/'training.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(dict(directory=str(directory), **record)), flush=True)
        if evaluator is not None and (step % eval_every == 0 or step == updates):
            before_evaluation = rng_state()
            try:
                metrics = evaluator(ema)
            finally:
                restore_rng(before_evaluation)
            value = float(metrics['station_macro_rmse_mm'])
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite development objective')
            write_json(directory/f'development_{step}.json', dict(step=step, **metrics))
            if value < best:
                best = value
                save(directory/'best.pt')
                write_json(directory/'best.json', dict(step=step, value=best, checkpoint='best.pt'))
            save(last)
            if trial is not None:
                trial.report(value, step)
                if step >= 4000 and trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned(f'Development RMSE at step {step}: {value}')
        elif step % checkpoint_every == 0 or step == updates:
            save(last)
    return dict(step=step, best=best, checkpoint=str(directory/'best.pt'),
                seconds=elapsed+time.perf_counter()-started)


def study_remaining(study, budget):
    # WAITING is queued work, not an attempted trial. RUNNING is recovered below.
    import optuna
    attempted = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    return max(0, budget-attempted)


def search(data, root, arm, stage='stage1', first_config=None):
    import optuna
    from .model import ResidualModel
    from .evaluation import evaluate
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sampler_path = root/f'{arm}_sampler.pkl'
    sampler = (pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else
               optuna.samplers.TPESampler(seed=20261113, n_startup_trials=4))
    study = optuna.create_study(study_name=f'{stage}_{arm}',
        storage=f"sqlite:///{(root/'studies.sqlite3').resolve().as_posix()}", direction='minimize',
        load_if_exists=True, sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4000, interval_steps=2000))
    registration = dict(stage=stage, arm=arm, data=fingerprint(data.manifest), budget=12 if stage=='stage1' else 6,
                        steps=10000, train_seed=42, sample_seed=20261113)
    if study.user_attrs.get('registration', registration) != registration:
        raise ValueError('Study registration changed')
    study.set_user_attr('registration', registration)
    if not study.trials:
        baseline = first_config or dict(lr=1e-4, weight_decay=.01, ema=.999, p_mean=-1.2, p_std=1.2)
        study.enqueue_trial({k:v for k,v in baseline.items() if arm=='edm' or k not in ('p_mean','p_std')})

    def objective(trial):
        config = dict(lr=trial.suggest_float('lr', 3e-5, 3e-4, log=True),
                      weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
                      ema=trial.suggest_categorical('ema', [.99,.999,.9999]))
        if arm == 'edm':
            config.update(p_mean=trial.suggest_float('p_mean',-1.8,-.6),
                          p_std=trial.suggest_float('p_std',.8,1.5))
        directory = root/arm/f'trial_{trial.number:03d}'
        write_json(directory/'config.json', config)
        torch.manual_seed(42)
        model = ResidualModel(arm).to(data.device)
        try:
            result = train_updates(model,data,directory,config,42,10000,
                evaluator=lambda m:evaluate(m,data,data.screen_indices,members=8,seed=20261113), trial=trial)
            trial.set_user_attr('best_checkpoint', result['checkpoint'])
            return result['best']
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            import traceback
            (directory/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            trial.set_user_attr('failure_kind', type(exc).__name__+':'+str(exc)[:200])
            raise
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Resume the same running trial and its checkpoint; never enqueue a duplicate.
    pending = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.RUNNING]
    if len(pending)>1:
        raise RuntimeError('Multiple running trials violate sequential study contract')
    while pending or study_remaining(study, registration['budget']):
        trial = optuna.trial.Trial(study, pending.pop()._trial_id) if pending else study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            recent = study.get_trials()[-3:]
            if len(recent)==3 and all(t.state==optuna.trial.TrialState.FAIL for t in recent):
                kinds=[t.user_attrs.get('failure_kind') for t in recent]
                if len(set(kinds))==1:
                    write_json(root/'paused.json', dict(reason=kinds[0], arm=arm, trials=[t.number for t in recent]))
                    raise RuntimeError('Three consecutive identical trial failures; study paused')
        finally:
            temporary=sampler_path.with_suffix('.partial')
            temporary.write_bytes(pickle.dumps(study.sampler))
            temporary.replace(sampler_path)
            write_json(root/f'{arm}_trials.json',[dict(number=t.number,state=t.state.name,
                value=t.value,params=t.params,attrs=t.user_attrs) for t in study.trials])
    return study
