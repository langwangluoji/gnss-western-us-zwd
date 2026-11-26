# Local generative architecture pilot

User authorization: 2026-11-22, implement the four discussed ablations locally and try Neural Diffusion Processes. This is an isolated research spike, not a production replacement. Native implementation; no server jobs or environment installation.

## Goal and design

Test whether spatial relation discrimination, structured multiscale noise, bottleneck condition attention, or magnitude-preserving blocks improve the existing GAN/EDM sampling parameterization. Separately test a coordinate-set observation-space diffusion model inspired by NDP; do not call it a faithful reproduction of the original paper or a continuous-field product.

Keep full 144x96 publication grids and widths 32/64/128/128. Six matched grid configurations: G0, G-pair, G-noise, D0, D-attn, D-MP. No combined variants. Both existing spatial folds, seed42. Same frozen M0+IDW mean, conditions69, source masks, Gaussian prior and keyed noise. Original and training-only marginally rescaled Gaussian controls are reported. All learned grid arms start at the rescaled Gaussian.

B2018 source: existing 576 training,48 inner and288 outer hours. Source base/prior and previous B outcomes are historically exposed: incremental development. Inner/outer short evaluation uses the first registered hour per month (12 each), selected by calendar only. Training interleaves months with fixed seeded order; no label-based selection. C2019/D2020 are not read. Do not pool the folds as one deployable field.

## Local budget

First benchmark a real full-grid 35-call backward. Initial short screen:48 updates per grid arm/fold, effective batch2 hours, micro1, K4. Save engineering check20, inner evaluation0/24/48 with K16, select minimum inner ES (then earlier step), final outer K32. Raw weights are primary; no unannounced EMA changes. Learning rate1e-4, AdamW weight decay1e-4, norm clip1. GAN critic1e-4 betas0/.9, GP10, adversarial weight0.1 matching actual previous training code (the old JSON says0.01 but code hardcodes0.1; disclose this mismatch rather than silently correcting it).

48 steps is a short mechanism screen, not two full passes, convergence evidence or a server promotion decision. Benchmark measurements may reduce the common budget before any scientific comparison; record this in a lock. Training/checkpoint logs per arm, resumable state and deterministic data cursor. GPU fits are sequential.

NDP-inspired branch: pointwise denoiser control versus two-block permutation-equivariant coordinate-set attention denoiser; widths64,480 DSM updates, batch8, both folds, same training labels. Predict noisy-observation residuals normalized by common hour amplitude and10mm, not latent clean weather fields. Independent base diffusion noise, observed-coordinate masks, standard EDM weighted denoising; no extra observation noise at sampling.18-step Heun. Evaluate the same station cohorts and report sampling mean error. No continuous grid export or query-subset consistency claim.

## Implementation plan (writing-plans / executing-plans)

- [ ] Add isolated `pilot_models.py`: invariant bounded station-pair critic, structured noise injection, condition attention, functional MP blocks, set denoiser. Tests: initial Gaussian equivalence; permutation/padding invariance; pair second-order GP; full35-call backward; live new-branch gradients; no parameter mutation in forward.
- [ ] Add `pilot_data.py`: build full-grid input-only69 cache using existing frozen M0 grids, verify cached compact67 and mean queries, expose only fitting labels to objectives. Cache per-hour checksums, source hashes and locked roles.
- [ ] Add `run_pilot.py`: benchmark, paired deterministic schedule, genuine optimization, finite checks, raw checkpoint selection on inner only, resume, outer scoring. Use repository ES/CRPS/VS evaluators; uncertainty and station-pair diagnostics retain population counts.
- [ ] Add NDP-inspired real-station DSM training and its pointwise control. Mask invalid values before all arithmetic. Inference never sees current truth or current target-valid mask.
- [ ] Run model/numerical tests; run local scientific pilot; save predictions/checkpoints/training curves and representative grid images. Document actual results and remaining uncertainty. Hash deliverables.

## Review focus

Held station labels must not affect inputs/prior/sampling. GP is differentiated with respect to original station residuals. Pair edges use physical geometry and identical true/fake masks. Functional MP normalization must survive repeated calls. Same-hour members remain stable when splitting batches. NDP mask handling must prevent missing-label leakage and avoid adding tau twice.

## Success criteria

Every requested arm actually updates weights and gives finite held-out metrics. A useful direction requires improvements relative to its matched new baseline in both folds, with CRPS/interval/mean-drift context. Tiny short-screen differences are hypotheses for longer training, not definitive rankings. Weather12-date uncertainty is exploratory; not full-week bootstrap.
