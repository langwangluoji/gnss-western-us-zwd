# Task 4 evaluation implementation

## Array contract

The public package is `gnss_field.evaluation`. Every ZWD value supplied to these functions is already in **millimetres**. Heights and projected x/y coordinates are metres. Distance weights use kilometres, obtained by dividing projected distances by 1000. The height reduction is reversed with `exp(height/3000)` before scoring Z0 increments.

```python
evaluate_arrays(hours, station_ids, truth, prediction, population_F,
                normal_support_S0, *, samples=None, std=None, metadata,
                coarsened_prediction=None, safe_T=None,
                coarsened_samples=None) -> dict
point_metrics(truth, prediction, population) -> dict
crps_ensemble(truth, samples, axis=-1)
energy_score_ensemble(truth, samples) -> float
variogram_score(truth, samples, distance_km) -> float
```

Truth, prediction and masks have shape `[T,N]`. Ensemble samples have shape `[T,K,N]`. The standalone CRPS routine accepts an explicit member axis. Standalone joint scores accept truth `[N]` and samples `[K,N]`. Metadata rows must match the supplied station order and contain `x`, `y`, `height`, `dense_group` and `cohort`. Station IDs are unique stable global indices.

Every truth value, prediction and ensemble member must be finite throughout F. Gaussian standard deviations must be finite and positive throughout F. Target/prediction NaNs outside F are allowed. S0 is a declared boolean subset of F. A failed prediction raises an error before any S0 filtering. A Gaussian marginal forecast supplies `std`. An empirical forecast supplies `samples`.

## Metric schema

`evaluate_arrays` returns `full`, `primary` and `unsupported` point/probability tables for F, S0 and F outside S0. Each table contains the following point keys.

- `station_macro_rmse_mm` is the mean of the per-station square roots of mean squared error.
- `micro_rmse_mm` is the square root across all declared station-hours.
- `station_macro_mae_mm`, `micro_mae_mm`, `station_macro_bias_mm` and `bias_mm` report absolute and signed errors.
- `p95_absolute_error_mm` uses the finite station-hour absolute-error population with linear quantiles.
- `n_stations` and `n_station_hours` report the scored population. Empty scopes retain zero counts and `None` scalar metrics.

Empirical CRPS uses the finite forecast distribution with the V-statistic pair term, including diagonal terms and denominator K squared. Gaussian CRPS uses its analytic expression. Probability tables add `station_macro_crps_mm`, `micro_crps_mm` and `intervals`. Gaussian tables also report station-macro and micro Gaussian NLL. Interval keys are `80`, `90` and `95`. Each contains station-macro coverage and mean width, micro coverage, covered count, station count and station-hour count. Empirical quantiles use NumPy `method='linear'`, Hyndman–Fan type 7, with the NumPy version recorded in the table.

The top-level `joint` table is present for empirical ensembles. Its primary keys are `group_macro_es_mm` and `group_macro_vs_mm`. Joint evaluation uses the same global member index at every site and at least four valid stations in a predeclared dense group at the same product hour. It averages within each group over its scored hours, then averages across groups. ES divides the L2 norm by sqrt(N). VS uses p=0.5 and upper-triangle weights `1/max(distance_km,25)` normalized to sum one. Both use forecast V-statistics. The training ES loss has its own U-statistic convention. Gaussian marginal standard deviations produce marginal scores.

`selection_values(bundle)` extracts point RMSE, marginal CRPS and available joint ES/VS as product-specific unrounded selection inputs.

## Fine-scale population and increments

The top-level `increments` table scores Z0 pair increments on S0 within each fixed `dense_group`. Both endpoints belong to the same published product hour. It includes `0-25`, `25-50`, `50-100`, `100-200` and primary `0-50` distance tables. Main 0–50 km pairs include distances at exactly 50 km. The adjacent descriptive bins use left-inclusive/right-exclusive edges, with 200 km included in the last bin. Each table reports a mean across group-specific pair RMSE values, pair-hour counts and a same-height surface-increment check restricted to height differences at most 100 m.

Coarsening requires explicit boolean `safe_T[N]` or `safe_T[T,N]`. The common population is exactly `S0 & safe_T`. The `fine` table contains `model`, `coarsened`, `model_increments` and `coarsened_increments` on this common population. Matching coarsened ensemble samples can add coarsened marginal CRPS/intervals. The caller supplies globally aligned coarsened fields and safe-stencil geometry. The scored fine population represents complete common query stencils.

`pair_increment_arrays` exposes fixed pair columns, group IDs, same-height flags, distances and same-hour masks for paired crossed bootstrap. Its arrays preserve the declared pair population.

## Paired uncertainty and decisions

```python
paired_block_bootstrap(hours, truth, prediction, baseline, population, *,
                      replicates=2000, block_days=7, seed=20261117,
                      return_indices=False, max_attempts=None)
crossed_increment_bootstrap(hours, truth, prediction, baseline, population,
                           group_ids, *, replicates=2000, block_days=7,
                           seed=20261117, max_attempts=None)
event_block_bootstrap(model_values, baseline_values, event_ids, *,
                     replicates=2000, seed=20261117)
direction_gate(comparisons, *, formal=True)
```

The point bootstrap resamples nonoverlapping calendar-time blocks with replacement and applies the same indices to all stations and both models. Each accepted replicate retains every originally active station and recomputes the station-macro RMSE. Draws missing an originally active station are rejected and redrawn. The trailing partial calendar block remains a whole block, so replicate station-hour lengths can vary. Numeric hours are nominal product-hour coordinates. NumPy datetime64 hours are also accepted. `block_days=14` implements the registered sensitivity analysis.

Point and crossed bootstrap outputs contain `difference_mm`, `ci95_mm` and `replicate_differences_mm`. Direction is model minus baseline. Crossed increment bootstrap establishes observed endpoint groups before sampling, then independently resamples those fixed group IDs and synchronized time blocks. Every accepted draw retains every selected group, including duplicates as repeated group weights. Both helpers report attempts and rejections and stop at a bounded attempt limit. Event bootstrap samples entire named weather processes and reports a paired observation-mean statistic with `difference` and `ci95`.

The fine gate requires exactly four preregistered comparisons. Pilot selection requires every difference strictly below zero. Formal selection requires every paired 95% CI upper bound strictly below zero. The caller supplies the point and increment comparisons against locked Bstar and self100km, all on the common T population.

`probability_gate` reports separate marginal improvement, joint ES/VS improvement, engineering coverage and formal coverage flags. Joint improvement also requires CRPS at most the baseline value. Engineering coverage checks nominal coverage within three percentage points at all three levels. Formal calibration requires every supplied coverage-minus-nominal CI to fit inside the same band. Missing or wider formal intervals produce `indeterminate`. `holm_adjust` adjusts a caller-specified preregistered claim family.

## Product selection and lock

```python
keep_top_candidates(records, products=('point', 'joint', 'marginal'), top=3,
                    *, require_full_year=False)
select_products(records, *, products=('point', 'joint', 'marginal'), top=3,
                require_full_year=False)
sampling_stability(samples64, samples128, *, point64_mm, point128_mm,
                   order64, order128, gates64, gates128, tolerance_mm=.005,
                   member_ids64=None, member_ids128=None,
                   prefix_tolerance_mm=.001, noise_provenance64=None,
                   noise_provenance128=None)
build_product_lock(products, *, config_sha256, manifest_sha256, stability,
                   protocol_sha256=None)
validate_retrospective_access(lock, *, year, role, config_sha256,
                              manifest_sha256, checkpoint_sha256,
                              product=None)
```

Selection uses point RMSE, joint ES and marginal CRPS independently. The record may hold the metric directly, under `metrics`, or under `selection_values[product]['value']`. Sorting uses the full floating-point value, then step, then checkpoint identifier. Pretraining stages and steps through 200 are excluded from learned-product candidates. Each product returns `selected`, `top_candidates`, `metric` and `full_year_required`.

`select_bstar(records, registered_controls, *, fixed_controls=('coarse_bilinear', 'A_IDW', 'A_kriging'))` selects the point-optimal registered control using complete 2019 results. The declared fixed nontrained controls can use step zero. Learned controls require steps greater than 200. Pretraining stages are excluded.

The stability helper verifies unique matching global member-ID prefixes and identical nonempty fixed-noise provenance. Forecast prefix differences can reach 0.001 mm for FP32 batch roundoff. This prefix tolerance is separate from the 0.005 mm point-RMSE change tolerance. Candidate order and gate decisions must also agree. Missing provenance produces an unstable result. Instability requests a uniform K128 rerank. The lock accepts a stable check or an explicit completed K128 rerank. Every selected product must have full-year 2019 metrics and a checkpoint SHA256, normalized to lowercase in the stored lock. The resulting JSON-compatible metadata contains product selections, config/manifest/checkpoint hashes, stability evidence and a self-validating lock digest. The caller persists this object through the existing immutable artifact writer.

Retrospective validation checks lock integrity, the fixed 2019 selection year, D2020 role, unchanged configuration/manifest hashes and membership of the requested checkpoint in the lock. The caller controls when the 2020 loader invokes this validation and supplies content-derived hashes.

## Image diagnostics

`ssim_diagnostic` uses uniform 7x7 windows, population covariance, K1=0.01 and K2=0.03. The caller supplies the fixed training-derived data range. Scoring uses common-mask windows containing all 49 valid pixels.

`psd_diagnostic` accepts a complete common rectangle, subtracts a least-squares plane, applies a symmetric two-dimensional Hann window and corrects power by mean squared window energy. It reports physical 50–100, 100–200, 200–400 and greater-than-400 km bands, radial power and phase-sensitive RMSE/error power. Members and ensemble means receive separate caller labels/calls. The anti-phase sinusoid test has matching truth/prediction power and positive phase-sensitive error.

## Verification

The tests were written before implementation and initially failed on the missing public module. Later regression tests exposed missing global member-ID validation and exclusion of legal fixed Bstar controls. Both failed behaviors were corrected and rerun.

At this task's final verification snapshot, evaluation tests passed 18 cases and the full package run passed **85 tests in 14.03 seconds** using `D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest -q`. The cases cover hand-computed macro/micro metrics, finite forecast CRPS, Gaussian analytic scores, dependence-sensitive ES/VS, finite-ensemble intervals, unchanged F despite prediction failure, common coarsening counts, height reversal, synchronized bootstrap indices, fresh RMSE per replicate, whole-event/crossed resampling, dual checkpoint selection, member prefixes, four-way gates, Holm correction, retrospective hash validation and phase-sensitive spectra.

Real-data 2019/2020 metrics and calibrated claim-family results are produced by the parent training/inference pipeline. These primitives operate on arrays and JSON-compatible metadata and perform their validation without opening current input observations or teacher data. Remote cluster execution, actual forecast checkpoint content, end-to-end retrospective loader enforcement and real-data bootstrap/calibration intervals remain integration-level verification items.

## Round 1 review corrections

All four reproduced review findings and the parent's FP32 stability regression have corresponding regression tests. The failing-test run reproduced variable station denominators, endpoint-empty group draws, learned startup Bstar eligibility, uppercase checkpoint lock mismatch and missing FP32 provenance support. The final prefix-bound test also confirmed that the numerical tolerance must stay at or below 0.001 mm.

The point helper records the fixed active station indices and station count. Crossed bootstrap records `eligible_group_ids`, `excluded_group_ids` and their fixed count. Both helpers report `attempts`, `rejected_draws`, `rejection_rate` and `max_attempts`. The default attempt budget is `max(1000,100*replicates)`. Exhaustion raises exported `BootstrapExhaustedError`, a RuntimeError carrying `attempts`, `rejected_draws`, `accepted_replicates`, `requested_replicates` and `rejection_rate`. A wholly empty time draw follows the same rejection path.

The stability provenance objects are JSON-compatible descriptions of the fixed-noise stream, such as its seed, sampler identity, checkpoint and noise construction version. K64 and K128 use the same provenance object describing their shared stream. The arrays of global member IDs must preserve the first 64 IDs. Matching output values alone, including a constant 53.5 mm field, provide insufficient provenance. The returned record includes `member_ids_match`, `noise_provenance_matches`, `provenance_matches`, `noise_provenance_sha256`, `prefix_max_abs_difference_mm` and `prefix_tolerance_mm` alongside the point-RMSE tolerance and difference.

Final Round 1 verification passed **34 evaluation tests in 0.80 seconds** and **127 full-package tests in 25.95 seconds**. This verification used the registered PyTorch environment and exercised synthetic/local tests. Real-data and remote integration checks retain the scope stated above.
