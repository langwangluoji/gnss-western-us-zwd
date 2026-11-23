# Final review fix report — 2026-11-17

All seven findings in `FINAL_REVIEW.md` have implementation corrections and regression coverage. The final scoped verification passed **101 tests in 15.16 s**. This wave also implements the controller-requested source/wheel binding, production numerical configuration, training telemetry and configured microbatch profiling. No remote jobs, paid work, legacy changes or scientific training were performed.

## Corrected scientific contracts

| Finding | Implementation and evidence |
|---|---|
| F1 | Production `_train_basis` passes effective 16 hours; `_pixel_run` passes 8 for both M2 arms. Functional pretraining/refinement retains 32. Tests intercept the actual engine boundary for all stages, rather than checking benchmark labels alone. Existing runtime estimate tests also pass. |
| F2 | Probability bootstrap fixes eligible columns before resampling, requires identical model/baseline validity, rejects any draw missing a selected station/group occurrence, and records attempts, rejected draws/rate, maximum attempts, eligible columns and block count. Bounded exhaustion uses the shared `BootstrapExhaustedError`. The sparse two-station example now has CI [5.5, 5.5], versus the reproduced old [1, 10]. Both station and crossed-group exhaustion paths are tested. Forecast nonfiniteness on scored cells is rejected before score construction. |
| F3 | Formal calibration bootstraps coverage minus each nominal level (0.80/0.90/0.95); these results are named `coverage_minus_nominal`. Model-minus-Gaussian comparisons remain separately identified. Formal confirmation also requires the engineering coverage checks. Equal 0.50 coverage in model and Gaussian no longer confirms calibration, even with improving ES/VS/CRPS. |
| F4 | `product_lock` independently selects the full-C2019 CRPS winner for each available M2 arm from stable2019 records. Its distinct checkpoint hash is locked under `controls.m2_f_marginal` / `controls.m2_j_marginal`. Existing retrospective control dispatch evaluates those checkpoints on D2020 without reselection. The regression uses different point/CRPS winners while Bstar belongs to another model. Marginal controls contribute station CRPS, intervals and widths, and do not enter joint claims or functional grid export. |
| F5 | `StageSpec.observation_early_stopping` explicitly includes functional refinement and both actual M2 observation stage names. Existing minimum 5,000 updates and six stagnant evaluations remain configurable contracts. Improvement in either point or marginal product resets patience. Resume/pending evaluation is checked before another optimizer update. M0/mean_all explicitly retain their fixed budgets: this is the controller's interpretation of the observation-refinement stopping rule for these from-scratch mean stages. |
| F6 | Every formal summary retains primary seven-day inference and adds `sensitivity.14` from the same saved forecasts and fixed populations. Both lengths include fine and probability endpoints, product claims and eligible-population metadata. Changed product confirmations, gates and nominal-coverage confirmations are reported. Probability cells are computed once; the sensitivity does not retrain or predict. |
| F7 | Production stochastic refinement selects 2018 hours with at least four qualified B observations; deterministic refinement retains minimum one. The production-dispatch boundary regression covers counts [1,2,3,4] across det/Gaussian/EDM/GAN. The controller's real-pack audit (minimum 82 in positive 2018 hours) means this boundary correction leaves the current real eligible population unchanged; this wave did not repeat that audit. |

## Additional controller-requested integration corrections

- `package_digest` hashes actual package files by package-relative name, excluding bytecode caches, independently of source or wheel installation path. Pipeline dependency/lock identity now uses this digest. `verified_release_hashes` compares the executing package against `project/src/gnss_field` and rejects stale wheel/source combinations. Release-wide hashing of source/configs/Slurm remains intact. Tests exercise identical copies, changed package content, missing packages and stale executing-package rejection. The existing temporary-release test fixture now includes a real package copy.
- Pipeline construction invokes the shared runtime configuration before model/CUDA work, using the requested seed and CPU count. Tests first set TF32 on and deterministic algorithms off, then verify the real pipeline restores the registered settings and seed 17.
- Optional engine batch telemetry records actual supervised station-hours, their checkpointed cumulative total, functional-head NFE and primary-objective head invocations. Estimated-coefficient pretraining reports zero direct real-observation supervision. Basis telemetry applies its fitting-station mask. Network counts explicitly cover the primary functional objective and exclude adversarial update, critic, encoder and patch-network calls; these excluded calls are not represented as total network usage. CUDA elapsed card-hours are labeled process elapsed time, not Slurm billed accounting. CPU reports zero GPU card-hours. Resume tests verify cumulative counts across saved updates without changing cursor, losses or RNG.
- Mean profiling includes the configured microbatch size even when it exceeds two. A final partial chunk preserves exactly eight hours and its actual loss weight: micro=3 uses 3+3+2. A small differentiable profiling regression verifies the consumed chunks and sweep entries.

## Test-first evidence and commands

All commands ran from the standalone package root with:

`D:/anaconda3-2025.12-2/envs/pytorch/python.exe`

1. `-m pytest tests/test_final_review_fixes.py -q --tb=short`
   - Initial scientific regression run: **13 failed, 5 passed in 5.68 s**. Failures reproduced 32-versus-16/8 actual batches, admission of counts 1–3 for stochastic refinement, [1,10] sparse-bootstrap CI, unmatched validity, missing bounded-retry argument, false calibration confirmation, M2 continuing to budget, and missing marginal control lock.
   - An earlier run exposed an invalid test-only manifest digest; the fixture was corrected and the above run reached the actual missing marginal-lock failure.
2. `-m pytest tests/test_final_review_fixes.py::test_formal_summary_keeps_seven_and_fourteen_days -q --tb=short`
   - **1 failed in 2.46 s** before sensitivity implementation: actual calls [None], expected [7,14].
3. The expanded regression file ran before the package/configuration additions. It failed on TF32 still enabled, missing `package_digest`, and no rejection of stale executing-package bytes. After scientific corrections, this intermediate run was **3 failed, 21 passed in 5.41 s**.
4. Telemetry regressions were run before implementation: the existing engine rejected `batch_telemetry`, and the production engine lacked the telemetry callback. The later combined targeted verification passed **37 tests in 8.64 s** (`tests/test_final_review_fixes.py tests/test_runtime.py`).
5. `-m pytest tests/test_final_review_fixes.py::test_mean_micro_sweep_includes_custom_partial_batches -q --tb=short`
   - **1 failed in 2.02 s** before extracting the callable production profiler; `_profile_mean_updates` did not exist. After extraction/correction, the regression file passed its then-current 30 cases. A crossed-bootstrap exhaustion parametrization subsequently raised the count to 31.
6. `-m pytest -q --tb=short`
   - Intermediate whole-suite run: **4 failed, 188 passed in 23.74 s**. All four failures were existing functional-stage dispatch test fixtures whose minimal manifests lacked `masks()`, newly needed for observation telemetry. Their fixture now supplies its already-declared four valid targets. Pretraining telemetry also skips mask lookup entirely. These cases are included in the passing final scoped run below.
7. `-m pytest tests/test_final_review_fixes.py tests/test_pipeline.py tests/test_pipeline_review.py tests/test_training.py tests/test_evaluation.py tests/test_runtime.py -q --tb=short`
   - **101 passed in 15.16 s**, including all 31 new regression cases, the four corrected integration fixtures, existing exact recovery, prior three P1 regressions, evaluation and runtime checks. A subsequent text-only clarification makes the telemetry count scope explicitly exclude adversarial updates.

## Edited files and handoff

Production: `src/gnss_field/training/pipeline.py`, `training/engine.py`, `evaluation/selection.py`, `runtime.py`, `artifacts.py`.

Tests: new `tests/test_final_review_fixes.py`; fixture updates in `tests/test_pipeline.py` and `tests/test_runtime.py`.

Report: `docs/FINAL_FIX_REPORT.md`. Controller-owned monitor/UI changes were not edited.

The controller will run final whole-suite verification and independent scoped rereview before packaging. Real Tongji driver, Slurm signal delivery, accounting and budget acceptance remain the existing deployment gates. Installed-wheel execution should be tested outside the source checkout so source imports cannot mask a stale wheel.
