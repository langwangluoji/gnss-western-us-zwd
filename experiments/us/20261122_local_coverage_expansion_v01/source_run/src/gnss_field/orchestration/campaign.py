"""Immutable, phase-separated scientific task descriptions."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import math
import re

from ..artifacts import fingerprint, immutable_json


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Unsafe identifier: {value!r}')
    return value


@dataclass(frozen=True)
class Task:
    task_id: str
    stage: str
    phase: int
    argv: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    run_id: str = ''
    card_hours: float = 0.
    max_attempts: int = 1
    gpu: bool = True
    scientific_gate: bool = False

    def __post_init__(self):
        safe_name(self.task_id)
        safe_name(self.stage)
        object.__setattr__(self, 'argv', tuple(self.argv))
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))
        if not self.argv or any(not isinstance(arg, str) or '\x00' in arg for arg in self.argv):
            raise ValueError('Task argv must contain literal string arguments')
        if self.phase < 0 or not isinstance(self.phase, int) or self.max_attempts < 1:
            raise ValueError('Invalid phase or attempt limit')
        if not math.isfinite(self.card_hours) or self.card_hours < 0:
            raise ValueError('Card-hour reservation must be finite and nonnegative')
        if not self.gpu and self.card_hours:
            raise ValueError('CPU tasks have zero GPU card hours')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    tasks: tuple[Task, ...]
    hashes: Mapping[str, str]

    def __post_init__(self):
        safe_name(self.campaign_id)
        object.__setattr__(self, 'tasks', tuple(
            task if task.run_id else replace(task, run_id=f'{self.campaign_id}/{task.task_id}')
            for task in self.tasks))
        object.__setattr__(self, 'hashes', MappingProxyType(dict(self.hashes)))
        if set(self.hashes) != {'plan', 'data', 'code', 'protocol'} or any(
                not re.fullmatch(r'[0-9a-f]{64}', value) for value in self.hashes.values()):
            raise ValueError('Campaign requires SHA256 plan/data/code/protocol hashes')
        by_id = {task.task_id: task for task in self.tasks}
        if not self.tasks or len(by_id) != len(self.tasks):
            raise ValueError('Campaign must contain unique task IDs')
        for phase in {task.phase for task in self.tasks}:
            if len({task.gpu for task in self.tasks if task.phase == phase}) != 1:
                raise ValueError('Each phase must use one homogeneous CPU/GPU resource class')
        for task in self.tasks:
            if len(set(task.dependencies)) != len(task.dependencies):
                raise ValueError('Duplicate dependency')
            for dep in task.dependencies:
                if dep not in by_id:
                    raise ValueError(f'Missing dependency {dep}')
                # Strict phase order also proves the graph has no cycle.
                if by_id[dep].phase >= task.phase:
                    raise ValueError('Dependencies must precede their phase (cycle or same-phase dependency)')

    def to_dict(self) -> dict:
        return {'campaign_id': self.campaign_id, 'tasks': [task.to_dict() for task in self.tasks],
                'hashes': dict(self.hashes)}

    @classmethod
    def from_dict(cls, value: dict) -> 'Campaign':
        return cls(value['campaign_id'], tuple(Task(**task) for task in value['tasks']), value['hashes'])


def build_default_campaign(campaign_id: str, hashes: Mapping[str, str], *,
                           search_arms=('det', 'm2_f', 'gaussian', 'edm', 'gan'),
                           card_hours: Mapping[str, float] | None = None) -> Campaign:
    """Build the registered DAG. Override conservative per-task GPU reservations after benchmarking."""
    costs = dict(card_hours or {})
    tasks = []
    previous = ()
    phase = 0

    def add_group(specifications):
        nonlocal previous, phase
        current = []
        for task_id, stage, args, gpu, gate in specifications:
            cost = costs.get(task_id, costs.get(stage, 4. if stage == 'benchmark' else 24.)) if gpu else 0.
            tasks.append(Task(task_id, stage, phase, (stage, *args), previous,
                              f'{campaign_id}/{task_id}', cost, max_attempts=1 if stage == 'benchmark' else 2,
                              gpu=gpu, scientific_gate=gate))
            current.append(task_id)
        previous = tuple(current)
        phase += 1

    for stage in ('pack', 'audit', 'doctor', 'benchmark', 'budget_gate'):
        add_group([(stage, stage, (), stage == 'benchmark', stage == 'budget_gate')])
    add_group([(stage, stage, (), True, False) for stage in ('mean_base', 'mean_all')])
    add_group([('mean_cache', 'mean_cache', (), True, False)])
    add_group([(f'basis_cv_{fold}', 'basis_cv', ('--fold', str(fold)), True, False) for fold in range(3)])
    for stage in ('representation_gate', 'basis_final', 'condition'):
        add_group([(stage, stage, (), stage != 'representation_gate', stage == 'representation_gate')])
    add_group([(f'search_{safe_name(arm)}', 'search', ('--arm', arm), True, False) for arm in search_arms])
    for stage in ('promote', 'full2019_rerank', 'stability', 'direction_gate'):
        add_group([(stage, stage, (), stage != 'direction_gate', stage == 'direction_gate')])
    add_group([(f'ablation_{variant}', 'ablation', ('--variant', variant), True, variant == 'rank256')
               for variant in ('pretrain_only', 'no_geometry_bias', 'current_only', 'rank64', 'rank256', 'joint_encoder')] +
              [(f'seed_{seed}', 'replicate', ('--seed', str(seed)), True, False) for seed in (17, 101)])
    for stage in ('product_lock', 'evaluate2020', 'report', 'export'):
        add_group([(stage, stage, (), stage == 'evaluate2020', stage == 'product_lock')])
    return Campaign(campaign_id, tuple(tasks), hashes)


def write_task_file(path: Path, campaign: Campaign, tasks: tuple[Task, ...]):
    immutable_json(path, {'campaign_id': campaign.campaign_id, 'hashes': dict(campaign.hashes),
                          'campaign_hash': fingerprint(campaign.to_dict()),
                          'tasks': [task.to_dict() for task in tasks]})
