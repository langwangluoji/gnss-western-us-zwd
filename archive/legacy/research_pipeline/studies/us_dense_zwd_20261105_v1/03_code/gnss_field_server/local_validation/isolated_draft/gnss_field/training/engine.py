"""Stage-aware, exact-resume training with atomic dual-product checkpoints."""
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import math
import random
import signal
import time

import numpy as np
import torch

from gnss_field.artifacts import atomic_bytes, fingerprint, immutable_json, write_json


class PauseRequested(KeyboardInterrupt):
    """A planned pause preserves an Optuna RUNNING trial for recovery."""


@dataclass(frozen=True)
class StageSpec:
    stage: str
    budget: int
    schedule_horizon: int
    effective_hours: int
    micro_hours: int
    seed: int
    dependencies: dict

    def __post_init__(self):
        if not 0 < self.budget <= self.schedule_horizon or min(self.effective_hours, self.micro_hours) < 1:
            raise ValueError('Invalid stage budget/batch')


class HourCursor:
    def __init__(self, hours, seed):
        self.hours = np.asarray(hours, dtype='i8')
        if not len(self.hours):
            raise ValueError('No qualified training hours')
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.hours)
        self.position = 0

    def take(self, n):
        result = []
        while n:
            k = min(n, len(self.order)-self.position)
            result.append(self.order[self.position:self.position+k])
            self.position += k
            n -= k
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.hours)
                self.position = 0
        return np.concatenate(result)

    def state(self):
        return dict(order=self.order.copy(), position=self.position, rng=copy.deepcopy(self.rng.bit_generator.state))

    def restore(self, state):
        if not np.array_equal(np.sort(state['order']), np.sort(self.hours)):
            raise ValueError('Training hour manifest changed')
        self.order = state['order'].copy()
        self.position = state['position']
        self.rng.bit_generator.state = state['rng']


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch(path, value):
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


def dataloader_worker_init(_):
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)


class TrainEngine:
    def __init__(self, model, directory, hours, spec, science, *, optimizers=None, clip_groups=None,
                 adversarial=False, max_seconds=None, auxiliary_optimizers=None, pause_paths=()):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.spec, self.science = spec, science
        self.cursor = HourCursor(hours, spec.seed)
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.optimizers = optimizers or [torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=science.learning_rate,
            weight_decay=science.weight_decay)]
        self.clip_groups = clip_groups or [[p for p in model.parameters() if p.requires_grad]]
        self.auxiliary_optimizers = list(auxiliary_optimizers or [])
        self.pause_paths = tuple(Path(p) for p in pause_paths)
        self.max_seconds = max_seconds
        self.step = 0
        self.pending_evaluation = False
        self.products = {}
        self.stagnant = 0
        self.elapsed = 0.
        self.allocation_elapsed = 0.
        self.pause = False
        self.phase = 'RUNNING'
        self.signature = fingerprint(dict(spec=asdict(spec), science=asdict(science),
                                          hours=np.asarray(hours), model=type(model).__name__))
        immutable_json(self.directory/'contract.json', dict(signature=self.signature, spec=spec, science=science))
        random.seed(spec.seed)
        np.random.seed(spec.seed)
        torch.manual_seed(spec.seed)
        if (self.directory/'last.pt').exists():
            self._load(self.directory/'last.pt')

    def _load(self, path, *, promotion=False):
        s = torch.load(path, map_location='cpu', weights_only=False)
        if not promotion and s['signature'] != self.signature:
            raise ValueError('Checkpoint scientific/runtime/data contract mismatch')
        self.model.load_state_dict(s['model'])
        self.ema.load_state_dict(s['ema'])
        for opt, saved in zip(self.optimizers, s['optimizers'], strict=True):
            opt.load_state_dict(saved)
        for opt, saved in zip(self.auxiliary_optimizers, s.get('auxiliary_optimizers', []), strict=True):
            opt.load_state_dict(saved)
        self.cursor.restore(s['cursor'])
        restore_rng(s['rng'])
        self.step = s['step']
        self.pending_evaluation = s['pending_evaluation']
        self.products = s['products']
        self.stagnant = s['stagnant']
        self.elapsed = s['elapsed']
        self.phase = s['phase']

    def state(self):
        return dict(signature=self.signature, spec=asdict(self.spec), model=self.model.state_dict(),
                    ema=self.ema.state_dict(), optimizers=[o.state_dict() for o in self.optimizers],
                    auxiliary_optimizers=[o.state_dict() for o in self.auxiliary_optimizers],
                    rng=capture_rng(), cursor=self.cursor.state(), step=self.step,
                    pending_evaluation=self.pending_evaluation, products=self.products,
                    stagnant=self.stagnant, elapsed=self.elapsed, phase=self.phase,
                    allocation_elapsed=self.allocation_elapsed)

    def save(self):
        atomic_torch(self.directory/'last.pt', self.state())
        write_json(self.directory/'status.json', dict(status=self.phase, stage=self.spec.stage,
                   step=self.step, budget=self.spec.budget, elapsed_seconds=self.elapsed,
                   allocation_elapsed_seconds=self.allocation_elapsed,
                   pending_evaluation=self.pending_evaluation, updated=time.time(), products=self.products))

    def _signal(self, *_):
        self.pause = True

    def _evaluate(self, evaluator):
        state = capture_rng()
        try:
            with torch.no_grad():
                metrics = evaluator(self.ema, self.step)
        finally:
            restore_rng(state)
        startup = self.step == self.science.startup_evaluation
        improved = False
        if not startup:
            candidate = self.directory/f'candidate_{self.step:06d}.pt'
            for product, result in metrics.items():
                value = float(result['value'])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite development objective')
                rows = self.products.setdefault(product, [])
                improved |= not rows or value < rows[0]['value']
                rows.append({**result, 'value': value, 'step': self.step, 'checkpoint': str(candidate.resolve())})
                rows.sort(key=lambda r: (r['value'], r['step']))
                del rows[3:]
            self.stagnant = 0 if improved else self.stagnant+1
            atomic_torch(candidate, self.state())
        with (self.directory/'development.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(__import__('json').dumps(dict(step=self.step, startup=startup, metrics=metrics,
                         updated=time.time()), allow_nan=False)+'\n')
        self.pending_evaluation = False
        self.save()

    def run(self, objective, *, evaluator=None, stop_after=None, before_update=None):
        if self.phase == 'COMPLETE':
            return dict(status=self.phase, step=self.step, products=self.products)
        signals = {}
        for sig in [signal.SIGINT, signal.SIGTERM] + ([signal.SIGUSR1] if hasattr(signal, 'SIGUSR1') else []):
            try:
                signals[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal)
            except ValueError:
                pass
        start = time.monotonic()
        prior_elapsed = self.elapsed
        try:
            self.phase = 'RUNNING'
            if self.pending_evaluation and evaluator:
                self._evaluate(evaluator)
            while self.step < self.spec.budget:
                self.allocation_elapsed = time.monotonic()-start
                self.elapsed = prior_elapsed+self.allocation_elapsed
                if self.pause or any(p.exists() for p in (*self.pause_paths, self.directory/'PAUSE', self.directory.parent/'PAUSE')):
                    self.phase = 'PAUSED'
                    break
                if self.max_seconds is not None and self.allocation_elapsed >= self.max_seconds:
                    self.phase = 'BUDGET_EXHAUSTED'
                    break
                if stop_after is not None and self.step >= stop_after:
                    self.phase = 'PAUSED'
                    break
                self.model.train()
                step = self.step+1
                warmup = min(self.science.warmup, self.spec.schedule_horizon)
                fraction = max(0., (step-warmup)/max(1, self.spec.schedule_horizon-warmup))
                peak_lr = self.science.refinement_lr if 'refine' in self.spec.stage else self.science.learning_rate
                lr = peak_lr*step/max(1, warmup) if step <= warmup else (
                    self.science.minimum_lr+.5*(peak_lr-self.science.minimum_lr)*(1+math.cos(math.pi*fraction)))
                for opt in self.optimizers:
                    for group in opt.param_groups:
                        group['lr'] = lr*group.get('lr_scale', 1.)
                    opt.zero_grad(set_to_none=True)
                batch = self.cursor.take(self.spec.effective_hours)
                if before_update is not None:
                    before_update(batch, self.model, step)
                sums = {}
                for i in range(0, len(batch), self.spec.micro_hours):
                    micro = batch[i:i+self.spec.micro_hours]
                    losses = objective(micro, self.model, step)
                    loss = losses['loss']
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training objective')
                    (loss*len(micro)/len(batch)).backward()
                    for key, value in losses.items():
                        number = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                        sums[key] = sums.get(key, 0.)+number*len(micro)/len(batch)
                for group in self.clip_groups:
                    torch.nn.utils.clip_grad_norm_(group, self.science.gradient_clip, error_if_nonfinite=True)
                for opt in self.optimizers:
                    opt.step()
                with torch.no_grad():
                    for target, value in zip(self.ema.parameters(), self.model.parameters(), strict=True):
                        target.lerp_(value, 1-self.science.ema)
                    for target, value in zip(self.ema.buffers(), self.model.buffers(), strict=True):
                        target.copy_(value)
                self.step = step
                self.allocation_elapsed = time.monotonic()-start
                self.elapsed = prior_elapsed+self.allocation_elapsed
                if evaluator and (step == self.science.startup_evaluation or step % self.science.evaluate_every == 0 or step == self.spec.budget):
                    self.pending_evaluation = True
                    self.save()
                    self._evaluate(evaluator)
                if step % self.science.log_every == 0 or step == self.spec.budget:
                    row = dict(step=step, stage=self.spec.stage, lr=lr, **sums, effective_hours=len(batch),
                               elapsed_seconds=self.elapsed, updates_per_second=step/max(self.elapsed, 1e-9),
                               allocation_elapsed_seconds=time.monotonic()-start,
                               gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
                               updated=time.time())
                    with (self.directory/'training.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(__import__('json').dumps(row, allow_nan=False)+'\n')
                    print(__import__('json').dumps(row), flush=True)
                    self.save()
                if evaluator and 'refine' in self.spec.stage and self.step >= self.science.minimum_refinement_steps and self.stagnant >= self.science.patience:
                    break
            if self.phase == 'RUNNING':
                self.phase = 'COMPLETE'
            self.allocation_elapsed = time.monotonic()-start
            self.elapsed = prior_elapsed+self.allocation_elapsed
            self.save()
            return dict(status=self.phase, step=self.step, products=self.products, elapsed_seconds=self.elapsed)
        finally:
            for sig, handler in signals.items():
                signal.signal(sig, handler)
