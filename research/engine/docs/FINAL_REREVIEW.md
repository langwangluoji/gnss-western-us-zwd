# Final scoped rereview — 2026-11-17

**Disposition: F1–F7 resolved. No additional actionable finding in the reviewed fix scope.** The executing-package boundary and the additional numerical, telemetry and profiling changes are consistent with their stated contracts. The controller's final full-suite and isolated-wheel artifacts have also been inspected and passed. Tongji acceptance remains a separate deployment gate.

This rereview read `FINAL_FIX_REPORT.md`, checked the changed production paths and regression assertions, and independently ran:

```text
D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_final_review_fixes.py -q --tb=short
31 passed in 6.38s
```

## Original finding dispositions

| Finding | Disposition and evidence |
|---|---|
| F1 — effective batches | Resolved. Actual basis engine construction uses 16; both M2 arms use 8. Functional stages retain 32. The tests intercept production engine arguments for basis/M2 and both phases of functional dispatch. Runtime estimates retain the same declared stage values. |
| F2 — probability bootstrap population | Resolved. `_cell_bootstrap` requires aligned validity, fixes original eligible columns, and rejects a draw if any selected column occurrence is empty. Crossed group duplicates retain their weights. Bounded retry metadata and shared exhaustion exceptions are present. The original disjoint-availability regression now returns [5.5, 5.5] and records rejected draws. Both ordinary and crossed exhaustion tests pass. `_probability_cells` also rejects nonfinite forecasts on scored cells before constructing missing-value score arrays. |
| F3 — coverage reference | Resolved. Formal inference now constructs coverage-minus-nominal cells separately for 0.80/0.90/0.95. These CIs feed calibration; Gaussian differences remain separately identified. The gate additionally requires nominal engineering coverage checks. The equal-50%-coverage regression exercises the production formal wrapper and no longer confirms calibration. |
| F4 — M2 marginal product | Resolved. `product_lock` independently chooses the full-C2019 CRPS winner for each M2 arm, checks its checkpoint hash and persists it under the controls. Existing retrospective control iteration evaluates these entries on D2020 without reselection. The regression verifies a distinct CRPS winner while Bstar belongs to another model, and confirms that this does not create a joint claim. Locked marginal controls also pass through the existing control-checksum validation path. |
| F5 — M2 early stopping | Resolved. `StageSpec.observation_early_stopping` explicitly includes both M2 observation names and functional refinement. Pending evaluation is replayed before the stopping condition is checked, and the condition is also tested before another optimizer update. Both M2 names stop on stagnation; improvement in either point or marginal score resets patience. The controller's stated interpretation retains fixed-budget M0 training, documented in the code and fix report. |
| F6 — 14-day sensitivity | Resolved. `_formal_evaluation` calculates seven-day primary and fourteen-day sensitivity results from the same forecasts, reusing probability cells. Both paths include nominal coverage and fixed-population uncertainty calculations. Output identifies changed claims, gates and coverage confirmations. Tests cover both fine-wrapper dispatch and probability summary contents. |
| F7 — ES eligibility | Resolved. Functional dispatch selects minimum four qualified observations for Gaussian/EDM/GAN and minimum one for deterministic MSE. The production-boundary regression covers counts 1, 2, 3 and 4 for all four arms. This remains a boundary correction; the previously reported current-pack minimum of 82 was not reaudited here. |

## Additional integration boundaries

**Executing package versus planned release.** `package_digest` uses package-relative filenames and actual bytes, excludes bytecode caches, and rejects a missing package root. Pipeline dependency/lock identity now uses that digest rather than a location-derived release root. `verified_release_hashes` compares the executing package digest with the planned source package before returning the full release hashes. Inspection confirms both the worker and direct benchmark entry paths call this verification; the benchmark itself also verifies its supplied identity. The portability, content-change, missing-root and stale-package regressions pass. An installed-wheel test outside the checkout remains the appropriate final packaging verification, and is assigned to the controller.

**Numerical configuration.** Pipeline construction calls the shared configuration before constructing its models. It sets the requested CPU thread count and seed, disables matmul/cuDNN TF32, and enables deterministic algorithms. The regression deliberately starts from opposite settings and verifies the real construction path restores these values. This is local configuration evidence, not proof of Tongji driver behavior.

**Training telemetry.** Counts derive from the actual update batch and qualified observation masks, with the basis fitting-station mask applied. Estimated-coefficient pretraining records zero direct observation supervision. The cumulative station-hour count is included in checkpoints and restored; the split-run telemetry test passes. Head invocation counts account for microbatch count and EDM's 35 evaluations, and are explicitly scoped to the primary functional objective. Adversarial, critic, encoder and patch-network work is excluded by the label, so these fields are not presented as total network cost. GPU elapsed time is separately labeled as process time rather than Slurm billing.

**Configured microbatch profiling.** The mean sweep includes the configured value and weights a partial final chunk by its actual size. The production profiler regression verifies microbatch 3 consumes 3+3+2 hours and still represents one effective-eight-hour update. Existing 1/2 probes remain present.

The monitor/UI edits were a separate controller-owned scope. The controller reported 16 CLI tests and browser QA for the engineering banner, charts, checkpoints and completed-full-year table. This rereview does not claim to have repeated that browser inspection.

## Evidence and remaining acceptance

The fix report supplies 101 passing targeted tests and describes the earlier four fixture failures and their corrections. This reviewer independently executed the 31 new regression cases and inspected their implementation integration; the full suite was deliberately not duplicated. No production code, legacy file or data was edited, and no remote action or scientific training was performed during rereview.

Final controller evidence arrived during rereview and was inspected directly: `local_validation/final_tests.xml` records **193 tests, zero failures/errors/skips**, with suite time 31.493 s (controller terminal result: 31.54 s). `local_validation/wheel_validation.json` records an actual import from a temporary installed-package directory outside the source checkout, `isolated_install_verified=true`, resources present and CLI help passing. It binds release hash `a5e5a2b378475f10feb1334d0bd7bde6e8465e54ee1f6732147c337b6ecd15c3` and actual package digest `23be004d127f5a3e9724472c34a47444311ad9e37a14df51beff3ee41ec55abb`; `training_started=false`. These are inspected controller-run artifacts, not repeated executions by this reviewer.

Local fix and release verification is complete for this scope. Real Tongji L40 availability/driver behavior, Slurm step-signal delivery, measured throughput and accounting still require the declared bounded benchmark and budget gates. These pending deployment checks do not reopen the resolved local findings.
