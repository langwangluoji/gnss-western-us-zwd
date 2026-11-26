# Task 1: full-field neural modules and observation mathematics

The new modules implement the published 37-grid/31-station contract, conditional VAE, latent GAN, latent EDM and surface-observation critic. All networks are newly defined in `generative/architecture.py`; the reused numerical functions are `keyed_noise` and `heun_sample` only. Existing network definitions and experiments were preserved.

## Verification

Interpreter: `D:/anaconda3-2025.12-2/envs/pytorch/python.exe`.

Command: `python -m pytest tests/test_full_generative_architecture.py tests/test_full_generative_observations.py -q`, with `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `PYTHONUTF8=1`.

- RED before implementation: collection reported the two new production modules missing. The first default-interpreter attempt had no torch; the intended pytorch interpreter then confirmed the expected missing-module state.
- First implementation run: 11 passed, 1 failed. The failing VAE posterior test exposed a method name collision with PyTorch's `_parameters` dictionary. Renaming the Gaussian splitter to `_gaussian_parameters` corrected it.
- GREEN after the correction: 12 passed in 2.67 seconds.
- Final strengthened verification: **13 passed in 3.80 seconds**, exit 0. Fixtures use CPU, 32-by-32 grids and latent dimension 8.

The tests cover encoder pyramid shapes; input station permutation; empty A sets; all-missing rows and masked NaNs; geometry-attention and null-token gradients; nonzero latent influence and gradients through every decoder parameter; posterior station permutation; inference label independence; empty posterior rejection; log-standard-deviation bounds; generator noise/condition gradients; critic and denoiser gradients; 35 EDM evaluations; independent keyed prefixes and batch independence; observation-pair gradients and permutation; exact bilinear interpolation followed by height restoration; hand-calculated Gaussian KL including unequal standard deviations; Gaussian normalization constant; common NLL/KL reference count; invalid-label sanitization before nonlinear arithmetic; and adding observation discrepancy exactly once.

A broad existing-suite run was interrupted on the parent agent's instruction before a final summary, after progress reached approximately 98%. It was started while the independently owned data/training modules were still under construction and displayed errors in their tests. It supplies no final suite-pass claim. The pytest failure cache also contained `tests/test_local_context_experiment.py::test_monitor_open_handle_allows_atomic_status_replacement`; its provenance was not isolated in this task. No unrelated code was changed.

## Measured trainable parameters

Counts below use the default 144-by-96 published grid and width 256 latent networks.

| Module | Parameters |
|---|---:|
| ConditionEncoder | 1,663,304 |
| FieldDecoder, latent 64 | 2,328,033 |
| ConditionalFieldVAE, latent 64 | 4,249,065 |
| FieldDecoder, latent 128 | 2,409,953 |
| ConditionalFieldVAE, latent 128 | 4,396,777 |
| LatentGenerator, latent 64 | 1,139,008 |
| LatentCritic, latent 64 | 164,609 |
| LatentDenoiser, latent 64 | 1,221,952 |
| ObservationCritic | 62,465 |

## API and numerical decisions

- All public neural and observation functions follow `INTERFACES.md`. `ConditionalFieldVAE.sample` returns clean full fields, shape `[B,K,1,H,W]`. `sample_surface` is the separate observation-discrepancy product.
- Both VAE log standard deviations are explicitly clamped to `[-6,2]`; they are log standard deviations, not log variances.
- `variational_loss` returns `loss`, `nll`, `kl`, `kl_per_dim`. `nll` and `kl` use the same scalar reference count; the raw average KL per dimension is a diagnostic. The optional reference-count default is the fixed query count N. Production must provide its locked training-only median eligible count.
- ObservationCritic consumes residual values already divided by 10 as specified. Its geometry uses projected XY divided by 150 km and height divided by 3000 m. Each bin selects at most 256 currently valid pairs in geometry-canonical order; pair selection depends on geometry and mask, never residual values. Empty bins contribute zero vectors.
- Station tokens masked out at the set boundary are sanitized before their MLP. A learned null token is enabled when no valid A token remains. Posterior training requires at least one valid station in every hour; the data protocol applies its stronger four-station eligibility gate upstream.
- FieldDecoder has a small nonzero final convolution and latent FiLM at all five scales. It adds the external background to a learned nonlinear residual and imposes no ensemble recentering.
- LatentDenoiser uses sigma_data 1, 18 positive levels from 80 to 0.002 with rho 7, followed by terminal zero: 17 Heun steps and one Euler step, 35 neural evaluations. Sampling retains gradients.

## Integration handoff

The root agent owns real-data/CUDA verification, scientific representation qualification, integrated trainer/evaluation checks, release hashes and any server submission. This Task 1 verification establishes small-grid behavior and mathematical reductions. It does not establish full-data convergence or representation qualification. Runtime measurement should include finite-value checks in the differentiable sampling chain. No server jobs were launched.
