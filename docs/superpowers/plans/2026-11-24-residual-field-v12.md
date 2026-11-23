# Residual field v12 implementation plan

> Use executing-plans inline; one independent review before the unattended queue.

Goal: serial, budgeted real-data comparison of pretrained multiscale fields, residual regression, correlated Gaussian and residual diffusion; retain diagnostics and publish lightweight results.
Spec: user approved all approaches in the preceding design discussion and requests about one hour per model, serial overnight execution. Existing main checkout and automatic GitHub sync remain authorised.

## Locked protocol

Training410/validation137 allowed-variable v10 data contract; confirmation138/C/D labels remain sealed. No prohibited water-column conditions. Twelve-hour causal context,82 conditions; mixed random/spatial-hole masking. Query-wise foundation readout uses no target labels or query-set interaction. CNN-LSTM is a radial multiscale spatial Conv1D encoder followed by a causal history LSTM, not a dense image truth model.

E1201 multiscale MLP field; E1202 radial CNN-LSTM field; E1203 XGBoost field; E1204 selected field + XGBoost residual; E1205 selected field + spatially correlated Gaussian residual; E1206 selected field + direct station residual diffusion; E1207 diffusion + common XGBoost correction. Two seeds42/17, nominal3600s budget per family including crossfit training, excluding final evaluation. Trees may early-stop sooner; no padding runtime. Foundation choice by average quarterly validation RMSE, frozen for dependent arms. Crossfit training uses spatial threefold groups inside410; exclude held-out groups from supervised fitting, context labels permitted as observable only. Final fitting held-out temporal windows monitor early stopping. Crossfit residuals feed correction/probability models; final base retrained on eligible training windows. All budgets and actual updates logged.

Selection quarterly2018 first4h x3layouts, expansion12month-start days x24h x3layouts; not blind validation. K16 selection, K32 final probability evaluation where feasible. CRPS selects probability checkpoints, RMSE also reported. Retain IDW12.065729 on identical full calendar. Missing-target mask required in all losses.

Residual diffusion operates in full target point values; no24/64-dimensional residual projection. Joint target graph may retain query dependence; foundation invariance tested, diffusion query sensitivity reported rather than silently assumed fixed. Correlated Gaussian supplies a non-diffusion spatial baseline. Common post correction shifts all members equally. Generated arrays/weights local, configurations/logs/figures/metrics/report GitHub.

## Tasks

- [x] Contract tests: fold isolation, masks, common correction preserves spread, foundation query invariance, finite diffusion gradient.
- [x] Cache multiscale allowed features and spatial-fold training packs; record hashes.
- [x] Implement foundation/crossfit and downstream models, serial queue and atomic receipts.
- [x] Smoke every arm on real data, independent review and fixes.
- [ ] Start full serial queue, verify first running stage; check periodically in this chat, no recurring automation.
- [ ] Finish all evaluations, audit saved metrics, write final report/figures, sync and verify GitHub.

Ruling: one-hour budget is per model family, split across two seeds and required fold fits; diagnostics/evaluation are additional. This keeps the whole suite practical, with actual budget reported. CNN-LSTM uses multiscale radial neighbourhoods to respect irregular GNSS geometry. No dense interpolation is treated as truth. Full suite may extend beyond07–08 depending on evaluation/runtime.


Implementation/review ledger: model and queue tests first failed on missing module, then passed. Independent review identified actual padded-query diffusion contamination, empty tiny-smoke calibration monitor, and Git failure aborting queue. Added query_valid independent of truth mask (RED/GREEN), expanded smoke to24OOF windows with nonempty-tree assertion, and sync_safely with network-failure regression. Full protocol now usesK16 for selection/expanded/calibration consistently to avoid ensemble-feature shift in E1207. Probability wall time includes extra evaluation beyond the nominal fitting budget. Source/cache hashes are checked before resuming. Original smoke interrupted before publication; smoke2 exercises fixed pipeline. Raw XGBoost weights excluded fromGit.

Verification:25 targeted tests passed; completed smoke stages independently recomputed against canonical validation labels. Expanded inference is part of the smoke run, so diffusion output validation adds several minutes even with an8-second fitting budget. No formal result is inferred from smoke.

2026-11-24 00:53CST: fixed smoke2 completed all7 stages in509.6s including full-calendar inference. Formal serial worker launched after independent canonical-truth/metric audit; budget3600s/family. Windows execution-state request prevents idle sleep only during worker lifetime.
