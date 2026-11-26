# Task 5 implementation report

## Delivered scope

Implementation is confined to `src/gnss_field/orchestration/`, `slurm/`, `tests/test_orchestration_campaign.py`, `tests/test_orchestration_study.py` and this requested report. Shared contracts and artifact utilities are reused. The parent owns training, CLI and evaluation integration.

The package provides an immutable campaign DAG, local dry-run artifacts, explicit Slurm submission controls, conservative GPU reservations, per-task atomic results, campaign-owned pause/resume and sequential restartable Optuna studies. Local verification used synthetic objectives and scheduler adapters. Real jobs, SSH and remote installation were outside the executed work.

## Public APIs

`Task(task_id, stage, phase, argv, dependencies=(), run_id='', card_hours=0., max_attempts=1, gpu=True, scientific_gate=False)` is a frozen dataclass. Literal argv is serialized into task JSON. The constructor validates identifiers, arguments, phase and resource reservations.

`Campaign(campaign_id, tasks, hashes)` is frozen, copies hashes into a read-only mapping and fills missing run IDs with `campaign_id/task_id`. It rejects duplicate IDs, missing dependencies, duplicate dependency entries and any dependency in the same or a later phase. Strict dependency phase ordering proves the DAG is acyclic. `to_dict()` and `Campaign.from_dict()` support immutable JSON round trips.

`build_default_campaign(campaign_id, hashes, *, search_arms=('det', 'm2_f', 'gaussian', 'edm', 'gan'), card_hours=None)` creates 35 tasks in 21 phases. The registered order is pack, audit, doctor, benchmark, budget gate, two mean tasks, mean cache, three basis folds, representation gate, basis final, condition, five searches, promote, full-2019 rerank, stability, direction gate, six ablations plus seeds 17/101, product lock, 2020 evaluation, report and export. The basis folds share a phase. Each later group depends on the preceding group.

The six ablation argv values are `pretrain_only`, `no_geometry_bias`, `current_only`, `rank64`, `rank256` and `joint_encoder`. Rank-256 carries scientific-gate metadata for the worker's observability decision. Representation and direction gates also carry scientific-gate metadata. Gate outcomes are worker-owned scientific decisions.

`write_campaign(directory, campaign, runtime=None, *, environment='gnss-field', working_directory=None)` writes immutable `campaign.json`, `runtime.json`, `execution.json` and per-phase task files, generated shell scripts and an initial `manifest.json`. It returns the manifest and creates `logs/` before submission. It has no subprocess side effects and creates no verified preflight.

`Coordinator(directory, campaign, *, runtime=None, client=None, environment=None, working_directory=None)` owns the campaign manifest. Omitted execution settings reuse saved values and default to `gnss-field` and the initial working directory for a new campaign. `Coordinator.from_directory(directory, *, client=None)` reconstructs an existing coordinator through a read-only path.

- `submit_phase(phase, *, execute=False, preflight=None, budget_lock=None, benchmark_cap=None)` returns a dry-run description by default. Explicit execution returns `job_id`, `summary_job_id`, `phase` and task IDs.
- `resume(phase, **kwargs)` regenerates an array containing incomplete task members, retaining run IDs and consumed attempt reservations. Fully completed members retain their successful state.
- `pause(*, job_id=None)` pauses the campaign and cancels owned active jobs. A supplied ID must be present in the campaign manifest. Without an ID, all owned active IDs are selected. Existing completed member receipts are preserved.
- `status(*, query_slurm=True, write=False)` returns manifest fields plus per-task states. `query_slurm=False` provides a local-only status read.
- `commit_task_result(directory, task, outputs, *, gate='PASS')` hashes declared files and atomically commits an immutable successful receipt, followed by per-task state.
- `set_task_state(directory, task, state, *, detail='', attempt=None)` records per-task JSONL events and atomic status. Supported states are `RUNNING`, `PAUSED`, `FAILED`, `SUCCEEDED` and `STOPPED_NO_GAIN`.
- `verified_receipt(directory, task)` validates task/run identity, gate PASS and every recorded artifact SHA256.
- `estimate_gpu_cost(card_hours, gpu='L40')` gives the published GPU-only estimate. L40 uses 2 yuan per card hour and A800 uses 4.

`python -m gnss_field.orchestration status --campaign-dir DIRECTORY` is the CPU afterany summary entry point, independent of the parent's CLI parser.

## Slurm and budget contracts

The real adapter requires explicit execution, POSIX execution on the hostname recorded in preflight, a verified Tongji site/profile and matching account/resource/hash declarations. The static preflight schema is shown below.

```python
{
    'verified': True,
    'site': 'Tongji',
    'profile': 'tongji_l40',
    'hostname': '<actual submission host>',
    'gpu': 'L40',
    'partition': runtime.partition,
    'gres': runtime.gres,
    'cpus': runtime.cpus,
    'memory_gb': runtime.memory_gb,
    'account_ok': True,
    'hashes': {'plan': '<sha256>', 'data': '<sha256>',
               'code': '<sha256>', 'protocol': '<sha256>'},
    'driver_ok': True,
}
```

Bootstrap CPU preparation and the first benchmark can use an explicit `benchmark_cap` between zero and four card hours after static Tongji verification. The allocated benchmark checks CUDA and the driver. Formal budget-lock submission additionally requires `driver_ok=True` from the GPU benchmark. This permits the first tracked GPU job to perform the GPU-specific validation.

The budget-lock schema is shown below. All four hashes must match exactly.

```python
{
    'locked': True,
    'hashes': dict(campaign.hashes),
    'total_card_hours': 100.0,
    'stage_card_hours': {'benchmark': 4.0, 'mean_base': 12.0},
}
```

The lock must cover `sum(task.card_hours * task.max_attempts)` for the whole declared campaign and each stage. The illustrative schema omits additional stages for brevity. A real full-campaign lock supplies every positive-cost stage.

Each submission reserves worst-case card hours before contacting Slurm. Failed, timed-out, cancelled and paused attempts retain reservations. Formal default DAG tasks allow two attempts. The benchmark allows one. The conservative uncalibrated full default is 1,204 card hours, including all declared second attempts. This is a planning upper bound. Benchmark-derived overrides should be supplied through `card_hours` before the campaign and budget hashes are locked.

Generated GPU arrays request one L40, at most seven CPUs, 64 GB and a time limit bounded by 24 hours and the shortest task reservation in the array. All arrays use `%2`. A coordinator refuses another active or unknown campaign job before opening a new phase. Later submissions include `afterok` barriers. Every array receives a CPU-only `afterany` summary. Logs exist before sbatch. Scripts source the public Conda profile, activate the configured environment and execute `srun python -u -m gnss_field worker --task-file ... --index "${SLURM_ARRAY_TASK_ID}"`.

The signal directive is `--signal=USR1@180`, which targets the job steps containing the worker. The generated scripts have no automatic requeue loop. Resume is explicit and bounded by each task's attempts and the retained reservation ledger.

Manifest updates are atomic and protected by an OS-held owner lock. Locks release after normal exit and process crashes. Per-task files use separate owner locks, with one JSONL stream per task. Slurm submission intentions are persisted before sbatch and returned IDs are registered immediately afterward. An uncertain submission response retains its intention and reservation and blocks duplicate submission. That condition requires reconciliation against the real scheduler before the manifest can proceed.

Downstream scientific completion requires a matching successful receipt, unchanged output hashes and Slurm COMPLETED status for the submitted task. A receipt or a checkpoint file alone cannot release a dependency. `PAUSED`, `FAILED` and `STOPPED_NO_GAIN` block downstream execution.

## CPU compatibility correction and public sources

Read-only HTTP requests on 2026-11-17 returned the current Tongji guide pages. The job submission page explicitly permits at most seven CPUs per L40/A800 GPU and demonstrates rejection of eight CPUs for one GPU. The parent approved the compatibility correction to seven CPUs and updated the shared RuntimeConfig default. Orchestration rejects eight CPU cores for this one-GPU profile. [Tongji job submission](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/quickStart/jobSubmit)

The resource guide lists L40 and A800 partitions and typed GPU GRES syntax. [Tongji resource configuration](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/quickStart/resourceConfig)

The Conda guide supplies `/share/apps/miniconda3/etc/profile.d/conda.sh`. [Tongji Conda](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/software/conda)

The charging guide lists L40 PCIe 48 GB at 2 yuan per GPU card hour and A800 SXM 80 GB at 4. Estimates expose these GPU rates. Account billing determines CPU and other charges. [Tongji charges](http://nic.pages.tongji.edu.cn/hpc-doc/#/pages/platformDes/charge)

## Optuna behavior

`run_study(directory, arm, objective, *, trials=8, seed=20261117, startup=4, validate_references=False, reference_validator=None)` owns exactly the supplied directory. One arm belongs to each study directory. Each trial uses `trials/trial_0000`, `trial_0001` and subsequent stable directories.

The callback receives `(params, trial_directory: Path, trial_number: int)` and returns the registered product/checkpoint mapping. Det minimizes point RMSE. M2-F jointly minimizes point RMSE and marginal CRPS. Gaussian, EDM and GAN jointly minimize point RMSE and joint ES.

The default total is eight trials, including fixed baseline trial zero with both rates at `1e-4`. TPE uses startup four and seed 20261117. Search rates use log intervals of `3e-5..3e-4` for pretraining and `1e-5..1e-4` for refinement. JournalStorage has one sequential owner. Sampler state and its SHA256 checksum are persisted in one atomic `sampler.state` container, preserving its progression through restarts.

RUNNING trials resume before new trials, retaining trial number, parameters and directory. KeyboardInterrupt and the parent's KeyboardInterrupt-derived PauseRequested leave the trial RUNNING and the study PAUSED. Completed callback calls are retained through their immutable result artifact and Optuna journal metadata. Failed attempts count toward the total budget. Nonfinite objectives produce FAIL. Three consecutive matching recorded errors stop the study and remain stopped on restart. Multiobjective execution calls neither trial.report nor should_prune.

The return value contains `arm`, `state`, `completed`, `failed`, `running`, `trials`, `selected` and `promotions`. Each selected product retains its own score/checkpoint association, parameters, trial number, pretraining source, contract and directory. Promotions form the union of selected configurations, with at most two entries. Every promotion references its saved pretraining checkpoint.

Production callers enable `validate_references=True`. Default validation loads checkpoint mappings, checks signature/spec/model metadata, requires the promotion source's `spec.stage` to end in `_pretrain`, parses the scientific contract and records all reference SHA256 values. Separate pretraining and refinement signatures are accepted. Completed production references are rechecked before promotion. A caller can supply `reference_validator(path, role)` for a stricter phase-specific content contract. Synthetic tests can exercise selection with reference paths and inexpensive callback results.

## Test-first evidence and verification

New tests were run while their APIs were absent and failed. Subsequent regressions were reproduced before their changes, including completed array members being marked paused, pretraining source metadata being read from the wrong key, the benchmark time limit exceeding its reserved hours, stale owner locks after process exit, production checkpoint mutation and the seven-CPU correction.

The final project-wide command was executed with `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q`. It completed with **66 passed in 13.58 seconds**. This includes 28 orchestration tests across the two delivered test files. The suite covers dry-run side effects, DAG validation/barriers, ownership and duplicate submissions, missing budgets, local profile rejection, benchmark limits, stage/total/attempt bounds, failed/resumed accounting, immutable task payloads, hashes plus scheduler completion, stop/pause propagation, retained successful array members, unknown submission responses, crash-release locks, study restart, resumed RUNNING trials, failure budgets, three-error stopping, product/checkpoint association, production reference content/hash checks and unsupported pruning/report avoidance.

## Remaining integration and deployment validation

The parent maps Task argv aliases into the training/CLI implementation and connects actual scientific gates and objective callbacks. The standalone summary entry point is already callable. Live account permissions, current partition/resource acceptance, CUDA delivery, USR1 behavior on Tongji job steps, cluster filesystem lock behavior and real training throughput remain deployment checks. The public documentation and local tests establish the generated contract. They provide no server execution measurement.

## Round 1 review fixes

The four findings in `docs/task-05-review.md` were reproduced with six failing regression cases before implementation changes. The reproductions used temporary directories, synthetic studies and scheduler adapters.

The first fix enforces the four-card-hour initial benchmark ceiling independently of the budget authorization branch. The coordinator sums `card_hours * max_attempts` across all declared benchmark tasks before either a full-budget lock or a bootstrap cap can authorize execution. Tests reject a five-hour benchmark with a matching full lock and a three-hour benchmark declaring two attempts. Both cases leave job submissions and reserved card hours at zero.

The second fix persists the Conda environment and resolved working directory in immutable `execution.json`. Coordinator reload bypasses campaign generation and preserves saved settings. Resume uses the same settings when generating a new attempt script. Existing campaign scripts retain their bytes and modification times after reload and status. Campaigns created before `execution.json` recover the environment and directory from their saved generated scripts. Pure status reads use atomic snapshots and create no owner-lock or planned files. The regression deletes one planned script, blocks all scheduler artifact writers and verifies that a status read leaves the complete directory snapshot unchanged.

The third fix requires one CPU/GPU resource class per phase and repeats the check at the direct renderer boundary. Mixed phases are rejected before array allocation. Every GPU array member therefore participates in GPU card-hour reservation accounting.

The fourth fix replaces the two independently published sampler files with one atomic `sampler.state` containing the checksum and serialized sampler payload. Reads verify the payload before deserialization. Existing valid `sampler.pkl` and `sampler.json` pairs remain readable and migrate when a trial next commits sampler state. The fault-injection regression raises KeyboardInterrupt immediately after the durable sampler write. Restart then reuses the RUNNING trial's number, parameters and directory and completes both budgeted trials.

After these fixes, the covering command `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_orchestration_campaign.py tests/test_orchestration_study.py -q` completed with **34 passed in 6.53 seconds**. The project-wide command `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q` completed with **111 passed in 19.42 seconds**. Implementation changes remained inside orchestration, its tests and this report. Server execution remains a deployment validation task.
