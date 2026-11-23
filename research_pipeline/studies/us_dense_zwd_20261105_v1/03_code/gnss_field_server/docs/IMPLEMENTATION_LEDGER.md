# SDD ledger — plan: APPROVED_PLAN.md

## Identity
Independent server package, approved 2026-11-17. Root project is not a Git checkout.
No legacy files, checkpoints, queue, or data may be modified. No paid Slurm job submission.

## Preflight and rulings
| Tasks/interfaces | Check | Resolution |
|---|---|---|
| Pure models / data | ContextBatch shapes and units must agree | See task-01-models.md; grid xy in metres, station geometry normalized |
| Data / inference | A only versus B/C/D labels | Separate read views and role/year gates |
| Representation / score | Estimated coefficients are not truth | Initialization only; final heads selected in observation refinement |
| Training / selection | Joint versus marginal probability | RMSE point, ES joint, CRPS M2 marginal; separate products |
| Slurm / resume | Shell-only signal would miss Python | USR1 sent to job steps; handler only sets flags |
| Budget / implementation | No remote credentials or budget yet | Build and test locally; generate scripts and <=4 card-hour benchmark plan only |
| Workspace / Git | No repository exists | Ruling: use approved new standalone directory and release hashes, no Git initialization |
| Training / data | Full-hour batches replace station-at-a-time | Report hours, labels, calls, wall time; fixed effective hour batches |

## Tasks
- [x] 1. Pure network extraction and generative models; legacy parity.
- [x] 2. Contracts, data pack, geometry and input-only construction.
- [x] 3. Production stage engine, real observations, exact ES, checkpoint recovery.
- [x] 4. Full evaluation, dual products, bootstrap and population contracts.
- [x] 5. Slurm campaign, Optuna, budget locks and monitoring.
- [x] 6. Real-data pack/smoke, portable release, independent reviews and handoff (local acceptance; remote benchmark remains gated).

## Verification policy
Run targeted numerical tests per task, then full tests and real-data CPU/CUDA checks.
Record actual commands and results below. Remote Slurm/driver/signal tests remain remote-required until observed.

## Execution findings
- Pure models: 22 targeted tests, M0/M2 numerical extraction and 35-call EDM gradients verified on CPU; independent review found no blocking fixed-geometry issue.
- Packed real data: 240 shards / 60 months, A174/B511/C110/D110; all shard hashes verified. Legacy data remains unchanged.
- Ruling: default single-GPU CPU allocation is 7, correcting approved draft8 — live Tongji jobSubmit.md rejects >7 CPUs/GPU — runtime-only change, scientific batches unchanged.
- Production engine: actual EDM 100+100 recovery equals continuous200; pending evaluation is replayed before another update.
- Task 4: fix round 1/5, original four findings addressed; scoped review clean for evaluation files. Pipeline sampling provenance integration passed to Task 3.
- Task 5: orchestration fix round 1/5, four findings addressed; scoped independent review clean (34 targeted tests).
- Task 3: fix round 1/5 implemented: current-only contexts and cache contracts, fresh allocation clock, per-family/product auxiliary configuration locks; 148 tests passed owner report, scoped rereview pending.
- Task 6 runtime review: direct benchmark CLI must verify actual campaign identity (P1); SSH connection failure must be visible in UI (P2). Fix round 1 in progress.
- Root integration: CLI/runtime ten tests passed; no-Slurm dry planning/status verified. Real CUDA profiling exercised M2 and det/Gaussian/EDM/GAN training/evaluation paths, cache IO and microbatch sweep; engineering-only output saved.
- Root data audit refreshed: all 240 monthly shard hashes verified; A174/B511/C110/D110 unchanged. Separate portable data ZIP CRC and all member SHA256 verified.
- Final-review attention: probability-specific pipeline bootstrap uses its own statistic; inspect fixed-denominator behavior and production 14-day sensitivity output, alongside approved evaluation spec.
- Task 3: scoped rereview clean; reviewer independently ran all 13 regression tests. All three P1 findings and sampling/cache integrations addressed.
- Fresh actual CUDA recovery after Task 3 fix passed: continuous200 versus100+resume100 bitwise equal; 35NFE finite gradients, query equivalence0.0000801086mm; `local_validation/final_real_cuda/smoke.json`.
- Task 6 runtime fix round 1/5 complete: both findings addressed; independent scoped rerun23passed and reviewclean. Full integration161passed22.57s before final scientific fix wave.
- Final broad review: seven verified findings F1–F7 in FINAL_REVIEW.md assigned together to one fix implementer. Production scientific batches and probability conclusions are gated until fixes and scoped rereview complete.
- Ruling: M0/M0-all retain the approved fixed maximum-update selection protocol; the >=5000-step/6-evaluation patience rule applies to observation refinement, including M2 — the server spec names observation refinement explicitly — cost if mistaken is additional M0 computation within its locked ceiling; checkpoint selection still uses development RMSE.
- Real qualification boundary: current2018positive supportedB minimum82; zero hours with1–3 observations. EnforcingESminimum4 changes no current2018sample count.
- Final fix wave: all F1–F7 addressed. Additional actual-package binding, production FP32/determinism setup, scoped telemetry and custom microbatch handling verified. Scoped independent review clean,31newregressions independently passed.
- Final controller suite:193passed31.54s, zero failures/errors/skips; JUnit saved. Wheel built and installed in a temporary target, run from outside source tree; all registered data hashes, executing package/source identity, resources, Pipeline import and CLI help verified.
- Final dry campaign generation: bootstrap4tasks/4worst-caseGPUcardhours; blueprint35tasks. Blueprint1204cardhours is conservative placeholder reservation including attempts, not an expected duration or cost prediction. Formal research requires measured explicit reservations.
- Monitor browser QA:工程验收banner, actual200-step logs, development-subset charts/best-checkpoint section and separate completedfull2019 table rendered. This is not an active scientific training window.
- Release-version CUDA test repeated after final engine corrections:77patches/860targets,35finite-gradient denoiser calls; continuous200 equals100+resume100 bitwise. Query discrepancy0.0000801086mm, allocated peak306.83MiB for this two-hour engineering probe. Not an L40/effective32 memory estimate.
- Old experiment rechecked live: M0step1021, phasepaused and PAUSEfilepresent. No old training was resumed. Formal server budget remains unlocked and no Slurm jobs were submitted.
