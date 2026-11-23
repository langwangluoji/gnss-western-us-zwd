"""Scientific and runtime contracts are separate and explicit."""
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ScienceConfig:
    rank: int = 128
    base_years: tuple[int, ...] = (2016, 2017)
    residual_years: tuple[int, ...] = (2018,)
    development_year: int = 2019
    retrospective_year: int = 2020
    frame_offsets: tuple[int, ...] = (0, 6, 12, 18)
    use_history: bool = True
    geometry_bias: bool = True
    condition_dim: int = 128
    basis_width: int = 256
    coefficient_penalty: float = .01
    curvature_weight: float = .001
    learning_rate: float = 1e-4
    refinement_lr: float = 1e-4
    weight_decay: float = 1e-4
    ema: float = .999
    gradient_clip: float = 1.
    warmup: int = 500
    minimum_lr: float = 1e-6
    sigma_data: float = .5
    p_mean: float = -1.2
    p_std: float = 1.2
    sampler_steps: int = 18
    rho: float = 7.
    sigma_min: float = .002
    sigma_max: float = 80.
    train_members: int = 4
    screen_members: int = 16
    evaluation_members: int = 64
    stability_members: int = 128
    stability_tolerance_mm: float = .005
    log_every: int = 100
    evaluate_every: int = 1000
    startup_evaluation: int = 200
    patience: int = 6
    minimum_refinement_steps: int = 5000
    study_trials: int = 8
    study_startup: int = 4
    study_seed: int = 20261117
    sampling_seed: int = 20261117
    bootstrap_replicates: int = 2000
    bootstrap_days: int = 7
    budgets: dict[str, int] = field(default_factory=lambda: {
        'mean': 30000, 'mean_all': 30000, 'basis_cv': 3000, 'basis': 10000,
        'condition_pretrain': 5000, 'condition_refine': 15000,
        'det_pretrain': 5000, 'det_refine': 15000,
        'm2_f': 20000, 'm2_j': 20000,
        'gaussian_pretrain': 10000, 'gaussian_refine': 10000,
        'edm_pretrain': 20000, 'edm_refine': 10000,
        'gan_pretrain': 20000, 'gan_refine': 10000})

    def __post_init__(self):
        for key in ('base_years', 'residual_years', 'frame_offsets'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if self.base_years != (2016, 2017) or self.residual_years != (2018,):
            raise ValueError('The registered temporal roles are fixed; use a new protocol for another split')
        if (self.development_year, self.retrospective_year) != (2019, 2020):
            raise ValueError('Development/retrospective years are fixed')
        if self.frame_offsets != (0, 6, 12, 18):
            raise ValueError('Current-only ablation masks history, preserving the four-frame schema')
        if not 1 <= self.rank <= self.basis_width or self.basis_width < 256:
            raise ValueError('Spatial rank must be supported by the final hidden width')
        if self.train_members < 2 or self.sampler_steps < 2:
            raise ValueError('ES requires >=2 members and Heun requires >=2 steps')
        if not 0 < self.ema < 1 or any(v <= 0 for v in self.budgets.values()):
            raise ValueError('Invalid training limits')


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: str = 'data/packed'
    run_root: str = 'runs'
    cache_root: str = 'cache'
    device: str = 'cuda'
    partition: str = 'L40'
    gres: str = 'gpu:l40:1'
    cpus: int = 7
    memory_gb: int = 64
    wall_hours: float = 24.
    max_parallel: int = 2
    patch_chunk: int = 8
    micro_hours: int = 1
    workers: int = 0
    precision: str = 'float32'

    def __post_init__(self):
        if min(self.cpus, self.memory_gb, self.max_parallel, self.patch_chunk, self.micro_hours) < 1:
            raise ValueError('Resource counts must be positive')
        if not 0 < self.wall_hours <= 168:
            raise ValueError('Tongji default job limit is seven days')
        if self.precision != 'float32':
            raise ValueError('The registered reference implementation uses FP32')


@dataclass
class ContextBatch:
    hours: torch.Tensor
    stations: torch.Tensor
    padding: torch.Tensor
    coarse: torch.Tensor
    maps: torch.Tensor
    baseline: torch.Tensor
    grid_id: str = ''
    scenario_hash: str = 'normal'
    mean: torch.Tensor | None = None
    coarse_z0: torch.Tensor | None = None

    def to(self, device: str | torch.device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                                for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})


@dataclass
class FieldProduct:
    mean: torch.Tensor
    support: torch.Tensor
    grid_id: str
    samples: torch.Tensor | None = None
    features: torch.Tensor | None = None
    marginal_std: torch.Tensor | None = None


@dataclass
class StationForecast:
    mean: torch.Tensor
    valid: torch.Tensor
    samples: torch.Tensor | None = None
    std: torch.Tensor | None = None


def load_config(path: str | Path) -> tuple[ScienceConfig, RuntimeConfig, dict[str, Any]]:
    import yaml
    with Path(path).open(encoding='utf-8') as handle:
        obj = yaml.safe_load(handle)
    unknown = set(obj) - {'science', 'runtime', 'campaign'}
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    return ScienceConfig(**obj.get('science', {})), RuntimeConfig(**obj.get('runtime', {})), obj.get('campaign', {})
