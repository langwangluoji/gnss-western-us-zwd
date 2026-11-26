# Whole-project independent review — 2026-11-17

**Disposition: changes required before scientific production.** Seven integration findings remain. Four affect the declared training workload or formal probability conclusions; the others omit registered products, stopping behavior, or uncertainty checks. The pending Tongji acceptance is a separate deployment gate.

Scope: the standalone package under `src`, its contracts, CLI, orchestration, evaluation and reporting integration. Binding reference: `APPROVED_PLAN.md`, read together with `IMPLEMENTATION_LEDGER.md` rulings. The documented seven-CPU correction is accepted. This review did not edit implementation, submit jobs, use SSH, or repeat the full suite. The report follows the requested clear, evidence-preserving writing principles.

## Findings

### F1 — [P1] Restore the approved effective batches in production basis and M2 stages

Locations: `src/gnss_field/training/pipeline.py:348` and `:697–700`; related estimate declarations at `src/gnss_field/runtime.py:401–414`.

Every basis CV/final run passes effective **32 hours**, although the approved basis batch is **16**. Every M2-F/J run also passes **32**, although the approved batch is **8**. `TrainEngine.run` consumes `spec.effective_hours` each update and normalizes accumulated microbatch losses by that total, so these are actual optimizer batches rather than reporting labels. The fixed update schedules consequently consume twice the intended basis hours and four times the intended M2 hours. This changes optimization and invalidates runtime estimates that explicitly use 16 and 8.

Set the production stage contracts to their approved values and test the actual `_train_basis`/`_pixel_run` engine arguments against runtime estimates. Preserve 32 for the functional-head stages. A regression asserting only benchmark metadata would miss this defect.

### F2 — [P1] Preserve the probability bootstrap's fixed scored population

Location: `src/gnss_field/training/pipeline.py:1309–1330`, especially `statistic` at `:1317–1320`.

When station/group availability is intermittent, a sampled time block can remove every observation for an otherwise eligible column. `means[counts > 0].mean()` silently removes that column and changes the macro denominator. With crossed group resampling it can likewise erase a selected group occurrence. These draws then feed CRPS, coverage, ES/VS intervals and selected-product p-values. The reviewed point/increment bootstrap has rejection logic, but this separate production implementation does not use it.

Independent CPU probe of the actual method: `hours=[0,168]`, model cells `[[1,NaN],[NaN,10]]`, baseline zero on the identical valid cells, seven-day blocks, seed 20261117 and 100 replicates. The observed two-station macro difference is **5.5**; the implementation emits replicate differences **1, 5.5, 10** and CI **[1,10]**. Under the registered fixed-station population, an accepted draw must retain both stations and gives 5.5 in this example.

Fix the original eligible population before resampling; require aligned model/baseline validity and preserve every selected station/group occurrence. Reject incomplete draws with a bounded attempt limit and rejection metadata, as in the existing point/increment implementation. Do not treat forecast nonfiniteness on declared scored cells as missing observations.

### F3 — [P1] Confirm coverage against nominal levels, rather than Gaussian coverage

Locations: `src/gnss_field/training/pipeline.py:1344–1349`; `src/gnss_field/evaluation/selection.py:210–215`.

`_formal_evaluation` passes CIs of **model coverage minus Gaussian coverage** into the formal calibration gate. The gate accepts an interval entirely within ±0.03 and, in formal mode, does not require the nominal engineering checks. Two equally miscalibrated distributions can therefore be declared calibrated. Equality to the baseline's coverage does not establish agreement with 80%, 90% or 95% nominal coverage.

Independent CPU probe of the actual gate: model coverage **0.50 at all three levels**, supplied coverage-difference CIs **[0,0]**, and negative ES/VS/CRPS differences. Result: all `engineering_coverage` flags are false, but `formal_coverage_status='confirmed'` and `calibrated_joint_pass=True`. The pipeline produces precisely that CI target whenever both forecast products have equal coverage cells.

Bootstrap each model coverage minus its corresponding nominal level on the fixed population, and use those intervals for calibration confirmation. Keep model-minus-Gaussian coverage differences as separately named comparisons if useful. Add a regression with equally undercovered model and baseline; it must not confirm calibration.

### F4 — [P1] Carry M2 marginal selections through locking and retrospective evaluation

Locations: `src/gnss_field/training/pipeline.py:1103–1137` (`product_lock`) and `:1201–1231` (`evaluate2020`); upstream marginal selection at `:265–269` and `:790–825`.

M2-F/J produce separate point and marginal candidates, and the full-2019 records retain CRPS selections. The downstream lock loops only over point/joint products of the advanced functional families. Its controls contain only Bstar and Gaussian. Retrospective evaluation reads those controls and locked products, so the chosen M2 marginal checkpoint is never evaluated on D2020 unless it happens also to be the Bstar point checkpoint. That coincidence is neither guaranteed nor the registered marginal selection rule.

Preserve the predeclared M2 marginal control selections independently of the functional family's fine/joint advancement. Bind their C2019-selected checkpoint hashes into the lock and evaluate their D2020 marginal metrics without selecting again. Keep these marginal controls out of joint-distribution claims. Test a case in which the M2 point and CRPS winners differ and M2 is not Bstar; the CRPS winner must still reach locked retrospective output. The controller confirmed there is no ruling excluding this product.

### F5 — [P2] Apply observation early stopping to M2 stages explicitly

Locations: `src/gnss_field/training/engine.py:270`; M2 stage construction at `src/gnss_field/training/pipeline.py:697`.

The engine checks early stopping only when the stage name contains `'refine'`. M2 uses `m2_f_observations`/`m2_j_observations`; it therefore continues to its budget even after at least 5,000 updates and six stagnant formal evaluations across its point/marginal products. The approved observation stopping rule covers all products. The same string predicate also excludes mean stages; any intentional fixed-budget exception for M0 should be stated explicitly rather than inferred from naming.

Represent stopping eligibility as a stage contract or an explicit stage predicate that includes M2 observation training. Test the real M2 stage names with stagnant metrics and confirm that improvement in either tracked product resets patience. Preserve pretraining behavior and resume state.

### F6 — [P2] Emit the registered 14-day sensitivity beside primary seven-day results

Locations: `src/gnss_field/training/pipeline.py:1262–1279`, `:1322–1329`, `:1332–1360`; `src/gnss_field/contracts.py:52`.

The formal production path runs one block length, `science.bootstrap_days` (default 7). No second 14-day invocation or sensitivity result is produced. The protocol explicitly registers 14-day sensitivity, and merely changing the global configuration from 7 to 14 replaces the primary calculation rather than providing both on the same locked forecast artifacts.

Run and label both block lengths for the registered formal endpoints, preserving seven-day results as primary and recording sensitivity intervals and any changed conclusions. Reuse saved station ensembles; no extra training or prediction is required. Verify that a standard retrospective summary contains both lengths and their declared populations.

### F7 — [P2] Enforce minimum four qualified B observations for stochastic ES refinement

Locations: `src/gnss_field/training/pipeline.py:575–576`; default selection at `src/gnss_field/data/manifests.py:18`; score guard at `src/gnss_field/objectives/scores.py:13–16`.

The functional refinement cursor uses `manifest.select([2018])`, whose minimum is one. The ES implementation rejects zero targets but accepts one to three. A future valid data pack with sparse supported hours therefore trains Gaussian/EDM/GAN ES below the approved four-observation eligibility threshold. Deterministic mean/MSE training has a different minimum-one rule and should retain it.

Independent CPU boundary probe: a manifest with valid counts `[1,2,3,4]` admits all four hours through the current selection; explicit minimum four admits just the fourth. The actual ES helper accepts three targets and returns a finite score (1.0 for the probe). The controller's fresh real-manifest audit reports 8,736 positive 2018 hours, minimum count **82**, and **zero** hours with one to three valid B targets. Thus this is a contract/boundary defect with no demonstrated impact on the current packed 2018 population.

Select minimum four for stochastic observation refinement and retain minimum one for deterministic refinement. Add the boundary test at the production cursor integration, not just in an unused helper.

## Verification and deployment boundary

The three independent CPU probes above ran with `D:/anaconda3-2025.12-2/envs/pytorch/python.exe`, importing the current source. Static call-path inspection establishes F1, F4, F5 and F6. No implementation edits were made during this review.

The controller reports a fresh full run of **161 passed in 22.57 s** and the runtime scoped rereview of **23 passed**. These are supplied verification results, not a second full-suite execution by this reviewer. The reviewed `local_validation/final_real_cuda/smoke.json` records the actual real-B 200 versus 100+100 bitwise-equal resume, 35 denoiser calls, 77 patches, 860 targets, finite gradients and query discrepancy 0.0000801086 mm. Those successful checks address numerical/recovery behavior; the integration findings above require additional targeted regressions.

Actual Tongji L40 driver, Slurm step-signal delivery, accounting and throughput remain pending by design. The smoke artifact correctly says `verified_for_tongji_budget=false`. This review does not classify that pending acceptance as a code defect or authorize remote execution. Production readiness requires resolving the actionable findings and then satisfying the already declared remote/budget gates.
