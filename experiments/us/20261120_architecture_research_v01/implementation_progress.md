# Execution ledger — local-context-model-research

Plan: docs/superpowers/plans/2026-11-20-local-context-model-research.md

- Authorization: user requested “开始吧”; execute the bounded local study before any server promotion.
- Ruling: this engineering project is a standalone non-Git directory. Use new tools/test files and a separate run directory; do not initialize Git, change old release snapshots, environments or weights.
- Shared interfaces: M0 grids -> A-only 99-channel features -> vertex-first query / fixed R64 projection -> three matched ES trainers -> locked B evaluation.
- Ruling: kernel = exp(-0.5*(distance/scale)^2), cutoff = 3*scale, scales75/150km; density is sum of unnormalized kernels. These are fixed before training.
- Tasks: (1) sparse M0 cache; (2) feature construction and numerical/leakage tests; (3) three-arm trainer and real resume; (4) bounded local experiments and locked B results; (5) review and report.
- Task 1 complete: sparse cache768h,327 reused441 generated;186.546s; frozen M0 hash unchanged.
- Task 2 complete: actual raw67+PCA32 cache3794 vertices includes3724 land vertices and every B query corner. Baseline round-trip0.0002823mm; PCA/amplitude/masks identical to archived B bundle.
- Ruling: normalize raw67 only on core hours/all retained vertices; retain frozen PCA32 unchanged. Historical B exposure remains explicitly labeled. Cost if wrong: geography weighting may affect optimization, to be tested in a separately registered study.
- Ruling: short-study startup=0, every100 update is a formal inner check; engineering200-step startup/recovery remains separate. This matches the short-study selection schedule and keeps step200 eligible.
- Task 3 complete: first seven invariant tests passed; fresh review found zero-incumbent patience and partial initial-save recovery issues. Added failing regression, fixed initialize_zero and zero incumbent; test verifies stop at300 for candidates worse than zero and checks evaluation at100/200/300.
- Actual resume: continuous200 ==100+resume100 for model/EMA/optimizer/RNG/cursor bitwise. Two implementation versions each used400 separate engineering updates;6000 is the formal scientific cap, not the engineering-update count.
- Local issue: normal Windows read handle reproducibly prevents atomic replacement (WinError5). First formal P0 stopped after saving step800. Resumed from that checkpoint; training source/science unchanged. Final monitor only reads append-only logs and once-published metadata, never status.json/last.pt or a replaceable snapshot. Regression covers ignoring unreadable status.json.
- Task 4 complete: all six formal runs finished1000updates; no retraining after B outer results; both architecture promotion gates failed. Original training source/checkpoints remain unchanged.
- Final review complete: independent reviewer checked features, fold/time separation, full-land projection, Gaussian factors, linear controls and final aggregation. Three selection/recovery findings fixed and regression-tested. Report clarifies identical7/14day anchor groups and cross-fitted pooled ES. Scalar/linear control has no EMA.
- Minor deferred: existing TrainEngine cumulative-supervision counter was not wired in this local adapter. Final results replay the exact seed42 HourCursor and masks to provide actual supervised station-hour counts; old zero log field is labeled uninstrumented. Cost: generic legacy monitor still displays this zero in raw-record details; no metric or checkpoint selection uses it.
- Additional field diagnostic: first fixed anchor has13 negative base-Z0 land cells, all outside current support;0 negative supported cells. No positivity modification was introduced into this scientific comparison.
- Final verification:268tests passed; all6000formal updates and report complete. C2019/D2020 not accessed for this new study; no server submissions. Local read-only monitor remains on127.0.0.1:8778.
