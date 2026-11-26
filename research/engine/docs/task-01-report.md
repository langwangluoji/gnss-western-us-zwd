# Task 01 — standalone neural models

Implemented the requested model package and 22 model tests. The complete package
suite passed: **31 passed in 6.00 s**, exit code 0, on 2026-11-17 using
`D:/anaconda3-2025.12-2/envs/pytorch/python.exe`. All reported verification ran on
CPU. No training, remote jobs, or pretrained scientific weights were used.

## Delivered interfaces

| Module | Behavior |
| --- | --- |
| `models.mean.MeanField` | Original M0 checkpoint keys and operations; accepts the legacy patch dictionary with a 6x6 baseline, returns `(field_z0_mm[B,1,24,24], None)`. |
| `models.pixel.ResidualField` | Original M2 parameterization and 32-channel features; preserves faithful/joint standard-deviation and loss gradients. `gaussian_losses` is exported from this module. |
| `models.condition.ConditionEncoder` | Station 17→64→64 encoder plus a 5→64 amplitude projection and a learned empty-set token; coarse 60→32→32 encoder; geometry-biased four-head attention; 103→64→64 fusion; adaptive 4x4 pooling and 1024→128 projection. Uses `context.mean` when supplied and `context.baseline` otherwise. |
| `models.basis.SpatialBasis` | Physical metre coordinates; three scaled coordinates plus 24 Fourier features at the six registered wavelengths; two width-256 hidden layers and a rank-wide output; column RMS normalization on the supplied fixed land mask. Configurable hidden widths must be at least 256 and support the requested rank. |
| `models.heads` | Deterministic normalized-unit coefficients, diagonal-Gaussian mean/log-standard-deviation with [-5,3] clamping, conditional GAN generator, and conditional critic. Coefficients have no imposed unit-vector constraint. |
| `models.diffusion.EDMHead` | Standard sigma-data=.5 EDM preconditioning; 64 sinusoidal noise features and condition FiLM in four fully connected width-256 residual blocks. `forward` aliases `denoise`; scalar and per-sample sigmas are supported. |
| `models.diffusion` functions | Five EDM coefficients, exact CPU keyed noise streams, and tensor-state differentiable Heun integration. The default 18 steps use 35 denoiser evaluations and retain their autograd graphs. |

`legacy_core.py` contains extracted pure neural classes from the original Europe
`model/network.py` and US `us_gan`, `cvae_pilot`, `station_mae`, `mae_gan_cvae`,
`moment_superresolution`, `regional_temporal_mae`, `architecture_search`, and
`residual_edm_v1/model.py` sources. Original constructor ordering is retained for
checkpoint compatibility. The residual core exposes its constructor and backbone;
historical training and no-gradient sampling methods were omitted. Production
modules import torch, standard-library modules, and the new package's contracts
and neural modules. Historical scripts and data loaders are never imported at
runtime. Legacy files were left untouched.

## Tests-first evidence

Before each implementation group, its tests ran and failed because the requested
standalone module was absent:

| Group | Initial run | First passing run |
| --- | --- | --- |
| Legacy parity and portability | 5 failed | 5 passed in 5.87 s |
| Condition, basis, coefficient heads | 7 failed | 7 passed in 1.65 s |
| Coefficient EDM and solver | 10 failed | 10 passed in 1.85 s |

The final complete-suite command, run from `gnss_field_server`, was:

```powershell
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -m pytest -q
```

Result: `31 passed in 6.00s`, process exit code 0, no failures or warnings.

## Test coverage and acceptance evidence

`tests/test_models_legacy.py` has five cases:

- `test_m0_exact_checkpoint_keys_and_nontrivial_output_parity` loads an actual
  historical MeanField state dictionary strictly, randomizes every parameter to
  exercise nonzero output paths, and compares ordinary, all-padded, and zero-length
  station sets with absolute tolerance 0.001 mm and zero relative tolerance.
- `test_m2_parameters_features_and_outputs_match_actual_legacy` checks parameter
  count, strict checkpoint loading, residuals, and features against the actual
  historical ResidualField. Output and feature tolerances are exactly zero.
- `test_faithful_loss_and_variance_gradient_parity[False/True]` checks both loss
  modes against the historical implementation. A nonzero variance projection
  exercises the detach behavior. Faithful mean gradients also match a separately
  derived MSE gradient; joint mode supplies feature gradients from uncertainty.
- `test_models_import_and_run_without_historical_modules` starts a fresh Python
  process that raises on imports of historical model modules, training modules,
  pandas, or h5py. It imports and executes every delivered model family, including
  coefficient EDM sampling. The child process succeeds. Historical parity cases
  explicitly skip when the historical source tree is absent from a standalone
  deployment; they all executed and passed in this local verification.

`tests/test_models_global.py` has seven cases:

- `test_condition_encoder_ignores_padded_values_and_station_order` verifies padded
  NaNs cannot change outputs and station permutation preserves the result within
  floating-point attention tolerance.
- `test_condition_encoder_empty_station_set_matches_all_padding_and_backpropagates`
  verifies finite null-token outputs for both empty representations and a nonzero
  learned null-token gradient.
- `test_condition_encoder_uses_supplied_mean_and_fixed_support_maps` checks mean
  override and baseline fallback, support-map sensitivity, and gradients into the
  station and coarse encoders.
- `test_spatial_basis_has_true_rank_128_and_fixed_land_rms` uses 640 physical
  coordinates in float64. The resulting matrix has numerical rank 128 and smallest
  singular value above 1e-5; each fixed-land column has mean square one. All basis
  parameters receive finite gradients.
- `test_spatial_basis_validates_rank_and_nonempty_fixed_land` rejects unsupported
  ranks and an empty geographical normalization mask.
- `test_coefficient_heads_shapes_condition_and_noise_gradients` exercises all
  head shapes and backpropagates to both conditions and GAN noise.
- `test_gaussian_head_predicts_nonzero_variance_and_clamps_extreme_log_std`
  verifies variance four from log-standard-deviation log(2), the empirical variance
  of 50,000 generated coefficient draws, and both clamp endpoints.

`tests/test_models_diffusion.py` has ten cases:

- `test_edm_coefficients_match_hand_computed_standard_formula` checks all five
  coefficients at sigma=.5 and 1 against independently calculated literals.
- `test_keyed_noise_is_exactly_prefix_batch_and_order_independent` checks bitwise
  prefix, reordering, and single-hour identity; seed sensitivity; output dtype;
  and preservation of the process-wide RNG state.
- `test_heun_generic_tensor_uses_35_calls_and_preserves_all_call_gradients` gives
  every solver evaluation a separate differentiable parameter. All 35 gradients
  are finite and nonzero.
- `test_heun_one_dimension_known_constant_clean_solution` checks the solver's
  result against a known clean vector.
- `test_edm_head_uses_preconditioning_and_supports_scalar_and_batch_sigma` checks
  the real head against independently calculated preconditioning results and
  rejects invalid sigma cardinality and zero sigma.
- `test_edm_samples_keep_prefixes_and_batch_identity` checks actual sampled
  coefficient prefixes and single-hour batch identity with FP32 tolerance
  `rtol=2e-5, atol=2e-5`. The underlying keyed initial noise is bitwise identical.
- `test_real_edm_sample_backpropagates_through_all_35_denoiser_calls` retains the
  real network output from every solver evaluation and backpropagates an
  ensemble-score-style objective. All 35 network outputs have finite nonzero
  gradients; conditions and every model parameter receive finite gradients.
- `test_edm_rejects_invalid_ensemble_or_solver_requests` has three parameterized
  cases for zero ensemble size, mismatched hours, and an invalid step count.

## Integration notes and remaining validation

No model implementation blocker remains. The patch-based M0 baseline uses the
original 6x6 convention; the global ContextBatch baseline/optional mean may use
the whole fine grid and is resized inside ConditionEncoder. M2 uncertainty expects
point features with the 32 features on the last axis, plus three geometry values,
as in the original model.

All numerical evidence above is CPU evidence. GPU behavior, full-data training,
scientific accuracy, calibration, and final training convergence belong to the
parent integration and subsequent registered experiments. Differentiable EDM
sampling retains 35 evaluation graphs per forward ensemble; training memory should
be measured with the intended server microbatch. Legacy parity tests require the
development-tree historical sources; standalone deployment/runtime tests do not.
