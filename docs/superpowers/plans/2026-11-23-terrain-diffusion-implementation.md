# Terrain temporal diffusion implementation plan

> **For agentic workers:** Use executing-plans inline, with a final independent review.

**Goal:** Build and locally validate the v2 direct-ZWD training/evaluation pipeline, with real learning curves and diagnostic figures.
**Architecture:** Dynamic training-pool context/target splits, geometry-aware attention, causal context Transformer, generated-history GRU, and direct v diffusion. Existing experiments remain immutable.
**Tech Stack:** Existing PyTorch/NumPy/Pandas/SciPy/Matplotlib environment; no new paid service or dependency.
**Spec:** ../specs/2026-11-23-terrain-temporal-diffusion-v2.md

## Constraints and rulings

- Reuse current checkout and data paths; new package and result directories isolate the implementation.
- Training410 / validation137 from the committed station proposal; confirmation138 and C/D labels remain unread during this implementation run.
- Parse raw files using only ZWD/T/P and timestamp/identity. Rebuild per-variable record QC. The inherited 905-station cohort was historically selected using water-vapour completeness; per-record QC does not erase station-selection history.
- P0/P1 and bounded real screening run in this turn; P4/P5 start only after scientific gates. No claim of full training or 8mm without results.
- Source/QC hashes, exact profile, data roles, sampling budget and checkpoint selection are recorded; large data/weights local, light evidence pushed.

## Tasks

- [x] 1. `tools/terrain_diffusion/data.py`: raw allowed-column parser, role-isolated cache, feature tensors and context masks. Tests first: forbidden-column independence and role guard.
- [x] 2. `model.py`: geometric attention, causal encoder, history GRU, v schedule and differentiable DDIM. Tests first: schedule inversion, permutation and gradient contracts.
- [x] 3. `train.py`: encoder transfer, deterministic/Gaussian/direct diffusion arms, EMA/selection/checkpoints and CSV logs; sampled trajectories and bounded local profile.
- [x] 4. `metrics.py`, `report.py`: common-population RMSE/CRPS/coverage/increments, terrain diagnostics, learning/quality/trajectory figures and report.
- [x] 5. Real raw preparation, tests, small overfit and full-chain backward profile; bounded two-seed screening with independent evaluation noise.
- [x] 6. Independent review, final verification, docs and GitHub sync.

## Review focus

- Hidden target values/physical aliases must not enter context; missing values must not become zero labels.
- Current-valid denoising subsets versus fixed generated trajectories are distinct.
- Evaluation is cold-start/self-history, not true target warm-start; causal features use past only.
- Finite-member and geographic support changes cannot alter common scoring population silently.
- Resume/selection/report artifacts must identify their exact data and source version.

## Execution ledger

Initial: user authorised implementation and training/evaluation deliverables. No new model training has run yet.

Task 1 complete: 547 raw files, allowed-column independent QC, 72×36×410×3 training cache; validation2018 only, confirmation/CD unread. Data/source hashes in immutable contract. Parser/role tests RED→GREEN.

Task 2 complete: cosine v inversion, node permutation, causal context, differentiable DDIM tests RED→GREEN. Mutable history dictionary checkpoint bug reproduced by temporal gradient test and fixed by frozen per-call dictionary.

Task 3–5 complete: 8 runs ×1000 updates; additional300-update pilot and300-update two-window overfit. Full4h K4 DDIM20 backward7.68s/1920.4MiB. Independent-noise24h B1/D1 seed42 K8. All8 saved main metrics recompute exactly; population identical. 12 targeted tests pass. Training arrays/weights local; CSV/config/metrics/plots publishable.

Ruling: bounded screen is1000 updates per arm, Jan/July4h/K4 selection; independent24h/K8 follow-up on seed42 — measure feasibility before formal P2 — costs: cannot claim full-calendar or two-seed24h scientific success.

Ruling: current model uses nearest12 context graph, mean/max global summaries, nearest support fallback; terrain nearest25km — minimal runnable graph prototype — costs: formal local/far geometry and learned global summarisation remain untested.

Ruling: batch1,Nq32,random82context,FP32,constant3e-4,EMA0.95; D1 transfers selected B1 encoder — bounded local engineering screen — costs: formal spatial sampling/large-query optimisation and separate MAE-head budget not reproduced; D1 has extra pretraining cost.

Ruling: publish CSV dynamic records rather than redundantJSONL; arbitrary-query interface uses dense masked attention and cached observation windows — lightweight reproducibility — costs: real-time input adapter and large high-resolution sparse graph deployment remain follow-up work.

Final review: four substantive findings fixed with RED→GREEN tests (nonfinite scoring, all-missing hour, source checkpoint receipt, configured report data). Final report also fixes Windows UTF8 path decoding. Older runs preserve original config and receive separately labelled postrun source receipts. No known Critical/Important item left open. See results/20261123_terrain_diffusion_v10/REVIEW.md.
