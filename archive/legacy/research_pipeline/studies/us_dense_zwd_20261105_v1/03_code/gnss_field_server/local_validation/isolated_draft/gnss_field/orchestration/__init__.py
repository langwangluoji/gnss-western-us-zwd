"""Public orchestration APIs.

Task / Campaign describe immutable argv-based phase DAGs. build_default_campaign
constructs the registered campaign and write_campaign writes local dry-run files.
Coordinator.submit_phase executes only after an explicit flag, Tongji preflight
and a hash-bound budget lock (or initial benchmark cap). Coordinator.resume,
pause and status preserve campaign-owned job and attempt records.

Workers call set_task_state and commit_task_result. A successful atomic receipt
contains task identity and output SHA256 digests. The coordinator combines these
receipts with Slurm completion before releasing scientific dependencies.

run_study(directory, arm, objective, *, trials=8, seed=20261117, startup=4,
          validate_references=False, reference_validator=None) resumes one
single-owner JournalStorage. The callback receives (params, Path, trial_number)
and returns point/joint/marginal checkpoint scores, a saved pretrain_checkpoint
and a scientific contract path. The report contains selected[product] and a
union of at most two promotions, each sourced from a pretraining checkpoint.
Production callers set validate_references=True.

Use Coordinator.from_directory(directory) for existing campaign manifests.
submit_phase and resume default to execute=False. Explicit execution accepts
preflight, budget_lock or benchmark_cap. status(query_slurm=False) reads local
state only. pause() targets campaign-owned jobs. The CPU afterany summary uses
python -m gnss_field.orchestration status --campaign-dir DIRECTORY.
"""
from .campaign import Campaign, Task, build_default_campaign
from .budget import estimate_gpu_cost, validate_budget, validate_preflight
from .scheduler import Coordinator, SlurmClient, render_script, write_campaign
from .state import commit_task_result, set_task_state, verified_receipt
from .study import run_study

__all__ = ['Campaign', 'Task', 'build_default_campaign', 'Coordinator', 'SlurmClient',
           'write_campaign', 'render_script', 'commit_task_result', 'set_task_state',
           'verified_receipt', 'validate_budget', 'validate_preflight', 'estimate_gpu_cost', 'run_study']
