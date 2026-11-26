# Execution ledger — PLAN.md

- Started 2026-11-17. Existing experiments read-only; new package field_generation_v1.
- Ruling: workspace is not a Git checkout; use independent package/output directories,
  source hashes and durable ledger rather than Git commits/worktrees.
- Data preparation pending; model, tests, training and visualization pending.
- Source numeric audit complete:905 representatives (174input,511train,110development,110retrospective).
- Five initial geometry/model/checkpoint tests passed. Full real-data smoke pending.
- Ruling: use nearest valid SRTM pixel within12pixels for coarse-coastline/DEM coastal
  disagreement; record every substitution. Larger gaps remain fatal. This avoids
  invented zero elevations; cost is up to approximately1.1km terrain-location error.
- Superseding terrain ruling: coarse country polygons disagree with SRTM coastline by
  more than12pixels at a second point. At invalid coastal centers use valid-terrain
  mean in a301x301 SRTM window (approximately25km grid footprint), recorded per cell;
  cells with no terrain remain fatal. No scoring cell is removed. Cost: coastal
  grid terrain represents nearby cell land rather than an exact-center elevation.
- Terrain complete:33 official tiles,16 recorded coastal fallback cells; grid144x96,
  77 overlapping patches. No output cells removed for coastline disagreements.
- Independent review found pending-evaluation/rerank resume, pause boundaries,
  insufficient gates and cache identity gaps. Added explicit phases, phase checks,
  evaluation cursors, residual-scale progress, data/code/config contracts and launch gate.
- TF32 caused station-permutation feature differences; disabled TF32 and enabled
  deterministic GPU algorithms. Real data smoke passed, 7 numerical/state tests passed,
  actual ResidualField two-optimizer100+100 versus200 exact GPU/EMA gate passed.
- No registered training has started at this ledger checkpoint.

- 2026-11-17 09:27:37 queue launched: Python PID27700, launcher28428; monitor12620 on127.0.0.1:8766, visible log32504. M0 step200 reached and screen evaluation running. HTML-only monitor additions do not change training contract.

- Paused safely at M0 step389. Completed required reporting groups, field/100km paired differences, 7-day intervals, support coverage and fixed calendar grid exports. Reporting-only contract migration recorded; model/EMA/optimizers/RNG/cursor exactly unchanged. Real-data smoke and all7 unit tests passed again; resumed from389.

- Dense-group audit found27 mixed-role50km groups: dispersed-fill sampling had included unassigned dense cells. Archived all preliminary M0 artifacts under excluded_split_trial_20261117 (not eligible results); corrected fill pool to genuinely dispersed cells, reran preparation. New dense mixed-role count0. Development58dense/52dispersed; retrospective57/53. All9 tests passed. Rebuilding role-dependent training stats/support and restarting from seed42, without checkpoint reuse.

- User redirected objective to local direction screening followed by server training. Safely paused at M0 step1021; checkpoint readable with optimizer/EMA/RNG. First formal step1000 screen macroRMSE12.413321206365003mm,76stations/42540station-hours. Full queue remains paused. Proposed reduced budgets are not yet applied or launched.
