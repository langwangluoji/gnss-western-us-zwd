# Task 5 orchestration review

Review performed on 2026-11-17 against `docs/task-05-orchestration.md` and `docs/task-05-report.md`, with seven CPUs accepted as the approved Tongji correction.

## Findings

1. **[P1] Enforce the initial benchmark ceiling on the budget-lock path.** `src/gnss_field/orchestration/scheduler.py` lines 189-200 applies the four-card-hour check exclusively when `budget_lock` is absent. A valid campaign containing a five-hour benchmark and a matching five-hour full lock passes authorization and submits. A temporary-directory reproduction with a mocked adapter registered the job and recorded `reserved_card_hours = 5.0`. Enforce the initial benchmark bound independently of the authorization branch, including declared benchmark attempts.

2. **[P1] Persist the execution environment and working directory across coordinator reloads.** `src/gnss_field/orchestration/scheduler.py` lines 152-155 reconstructs the coordinator with default environment and working directory. Its constructor immediately rewrites every script through lines 148-149 and 122-126. Consequently the normal summary/status entry point and a later resume discard a configured Conda environment and can change the import/data working directory. Reproduction created scripts using `custom-env` and a custom work directory, then called `Coordinator.from_directory`. The restored environment became `gnss-field`, the working directory became `None`, and the generated script lost the custom environment. Persist and reload both settings, and make status reconstruction preserve existing execution configuration.

3. **[P2] Reject mixed CPU/GPU members within one Slurm array phase.** `src/gnss_field/orchestration/scheduler.py` lines 93-96 adds one GPU to the entire array whenever any member has `gpu=True`, while lines 200-215 and 274-282 reserve the CPU member's zero card-hours. `Campaign` currently accepts this phase. A phase with one one-hour GPU task and one CPU task renders `--array=0-1%2`, `--time=01:00:00` and `--gres=gpu:l40:1`, allocating a GPU to both members while reserving one card-hour. Enforce homogeneous resource classes per phase or split resource classes into independently accounted arrays.

4. **[P2] Commit the sampler state and checksum as one recoverable generation.** `src/gnss_field/orchestration/study.py` lines 172-173 independently replaces `sampler.pkl` and then writes `sampler.json`. An interruption between these writes leaves a missing or stale sidecar. Lines 147-149 reject every subsequent restart before reopening the RUNNING trial. An injected `KeyboardInterrupt` at the sidecar write reproduced `Persisted TPE sampler hash differs` on the next call. Store state and checksum in one atomic container or publish a completed generation with an atomic pointer while retaining the prior valid generation.

## Verification

`D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q tests/test_orchestration_campaign.py tests/test_orchestration_study.py` completed with **28 passed in 6.14 seconds**.

All four reproductions used temporary directories, synthetic callbacks and mocked scheduler adapters. This review executed local Python and tests. Live Slurm submission, network calls and cluster signal delivery remain unverified. Source files were reviewed without implementation edits. This file is the documentation-only deliverable.
