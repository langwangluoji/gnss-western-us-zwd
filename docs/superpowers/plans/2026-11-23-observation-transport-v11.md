# Observation transport v11 implementation plan

> Execute inline using executing-plans; one fresh-context review at completion.

**Goal:** Test the v10 diagnosis with real paired training before scaling diffusion.
**Architecture:** Preserve direct surface-ZWD output and existing causal graph backbone. Add condition-only multiscale current/past ZWD summaries; transport variant changes messages by relative elevation with a learned positive scale and a small weather/geometry edge correction. No interpolated output background or target teacher.
**Spec:** `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_terrain_diffusion_v10_diagnosis/ANALYSIS.md`, authorised by user's “实践”.
**Tech Stack:** Existing PyTorch/CUDA, NumPy, SciPy, Matplotlib.

## Locked execution choices

- Reuse existing checkout; new v11 modules/results isolate historical models and contracts. Training410/validation137 only; no confirmation/CD labels. Same72 training windows, allowed ZWD/T/P, inherited cohort limitation unchanged.
- A: base/raw/transport, seeds42/17,2000 updates each. No target history. Same random context82 and cycling query counts64/96/137/160, AdamW2e-4,100-step warmup/cosine to2e-5,EMA0.99. All arms share protocol; compare within v11, not attribute changed training protocol to the new encoder.
- Raw summaries: three spatial Gaussian scales50/150/300km × temporal offsets0/3/6h × value/std/log-support =27 dimensions. Missing values excluded before weighted sums.
- Transport uses the same27 summaries, but transforms each context ZWD to query height using exp((Hc-Hq)/L), L bounded1–6km initial3km. A zero-initialised weather/geometry edge MLP adjusts transported standardised values. No new trainable station identities; all features from context only.
- Selection: quarterly2018 month1/4/7/10 day1, first4h,3fixed layouts; deterministic macroRMSE. Final frozen A follow-up: all12month1 days×24h,3layouts, plus untuned surface/height3000IDW baselines; same truth population.
- B gate: transport/raw must improve both seeds over paired base on the selection screen; otherwise diagnose A and stop expansion. If passed, choose single encoder variant by two-seed mean macroRMSE, train Gaussian/direct diffusion3000 steps each/seed with same transferred conditioner. No target history, K16 fixed selection, CRPS checkpoint plus RMSE tradeoff recorded. Final probability audit on a broader calendar if feasible; any reduced calendar explicitly named.
- C gate: only candidates meeting B1+0.2mm point tolerance with probability improvement enter200–500step self-history temporal training. Otherwise retain direct generative model and report blocker rather than unqualified temporal scaling.
- Save gradient norms/clipping, learning rate, fixed training diagnostics, validation scores, exposure counts, checkpoints each500 updates, source/data hashes, curves and metrics. Raw arrays/weights local; lightweight results GitHub.

## Tasks and ledger

- [x] Tests: exact constant-field height transport, missing-value isolation, no hidden target/extra-query dependency of condition summaries; all-null support finite.
- [x] Implement isolated conditioner/model and paired trainer with history disabled.
- [x] Run six A experiments, all-calendar evaluation and gate decision.
- [x] Run B if gate passes; C only if its scientific gate passes.
- [x] Independently review, verify saved metrics, produce plots/report, sync GitHub.

Initial: user approved practical execution of the prior diagnostic route. No v11 result yet.

Task1/2: module import test RED, then15/15 targeted tests GREEN. Implemented condition-only27-feature summaries, bounded learned1–6km transport, weather/geometry edge correction; training follows same missing-history protocol for all arms. Immutable v10 results untouched. A runs active.

A training complete: six2000-step runs. Quarterly two-seed means base14.309146,raw12.073782,transport12.172332mm. Both enhancement variants pass paired improvement gate; single raw variant selected by the preregistered mean rule. Full12month24h A follow-up active; not used to select variant. B Gaussian/diffusion3000/seed active.

A expanded calendar complete: base14.307291/raw12.456102/transport12.231071mm two-seed means; fixed height-IDW12.065729mm. Full-calendar observations did not change the raw selection. Replay of actual A draws finds median nearest valid context39.69km vs validation77.93km; remaining spatial-support shift is recorded, not silently changed mid-run.

Independent review: no leakage/height-unit blocker; calendar label default could misdescribe quarterly scope. Reproduced RED then fixed by derived label and report manifest verification.17/17 targeted tests GREEN, including full conditioner transfer and checkpointed sampler gradient. Current actual A evaluations explicitly used all12months; historical metrics unchanged. Expanded artifact audit added and first13 saved artifacts recompute exactly.

Ruling: C probability gain means improvement over the paired Gaussian baseline in both seeds; refine its decision using analytic Gaussian moments/CRPS and K32 diffusion with independent noise, as suggested by review of the0.2mm finite-ensemble sensitivity. This interpretation is documented during B, not claimed as a fully preregistered detailed C rule. Cost: conservative gate may postpone a temporal-only improvement; such an experiment needs a separately named hypothesis. Frozen B checkpoints remain selected by originalK16 rule.


Final 2026-11-24: all10 runs complete,24,000 updates. All10 frozen checkpoints evaluated on12month-start days×24h×3layouts. B expanded means Gaussian12.285455mm/CRPS6.667626, diffusion12.339468mm/CRPS6.784056. C refined gate failed both seeds; C training intentionally not started. Expanded near-pair CRPS improves about5.0% with diffusion; marginal-preserving shuffle worsens diffusion by0.135/0.131mm, while50–150km shuffle improves it. Exploratory finding does not override gate.18tests pass;20artifact audits and4gate audits pass. Reports, curves and project navigation prepared for GitHub sync.
