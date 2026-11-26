"""Conservative GPU reservation limits. CPU pricing is deliberately unspecified."""
from collections import defaultdict
from collections.abc import Mapping
import math

from .campaign import Campaign


def _positive(value, label, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid {label}')
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'Invalid {label}')
    return float(value)


def validate_preflight(preflight: Mapping | None, campaign: Campaign, runtime, *, require_driver=True):
    if not preflight or preflight.get('verified') is not True:
        raise ValueError('A verified Tongji preflight is required')
    if preflight.get('site') != 'Tongji' or preflight.get('profile') != 'tongji_l40':
        raise ValueError('Only the verified Tongji L40 profile unlocks submission')
    checks = {'gpu': 'L40', 'partition': runtime.partition, 'gres': runtime.gres,
              'cpus': runtime.cpus, 'memory_gb': runtime.memory_gb,
              'account_ok': True, 'hashes': dict(campaign.hashes)}
    if require_driver:
        checks['driver_ok'] = True
    if any(preflight.get(key) != value for key, value in checks.items()):
        raise ValueError('Tongji preflight resource or scientific hashes differ')


def validate_budget(campaign: Campaign, budget_lock: Mapping | None) -> dict:
    if not budget_lock or budget_lock.get('locked') is not True:
        raise ValueError('Full submission requires a budget lock')
    if budget_lock.get('hashes') != dict(campaign.hashes):
        raise ValueError('Budget lock scientific hashes differ')
    total = _positive(budget_lock.get('total_card_hours'), 'total budget')
    stages = budget_lock.get('stage_card_hours', {})
    if not isinstance(stages, Mapping):
        raise ValueError('A per-stage budget mapping is required')
    planned = defaultdict(float)
    for task in campaign.tasks:
        if task.gpu and task.card_hours <= 0:
            raise ValueError('Each GPU task needs a positive worst-case reservation')
        planned[task.stage] += task.card_hours * task.max_attempts
    if sum(planned.values()) > total + 1e-9:
        raise ValueError('Declared worst-case attempts exceed total budget')
    for stage, needed in planned.items():
        bound = _positive(stages.get(stage, 0), f'{stage} stage budget', allow_zero=True)
        if needed > bound + 1e-9:
            raise ValueError(f'Declared attempts exceed {stage} stage budget')
    return {'total_card_hours': total, 'stage_card_hours': dict(stages),
            'planned_card_hours': sum(planned.values())}


def estimate_gpu_cost(card_hours: float, gpu='L40') -> float:
    """Public GPU-only estimate in yuan. Billing authority remains the HPC account."""
    rate = {'L40': 2., 'A800': 4.}.get(gpu)
    if rate is None:
        raise ValueError('GPU price estimate supports L40 and A800')
    return _positive(card_hours, 'card hours', allow_zero=True) * rate
