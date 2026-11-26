# Weather control summary verification

- Scope: `weather_controls_summary.py`, its tests, three `paired/summary.json` families, and `2026-11-26-weather-controls-final.md`.
- Three families contain 8 centered, 10 transport, and 6 supplementary-calendar rows. Fifteen model comparisons use shared month indices across the two fixed seeds.
- Fresh maintenance verification: **23 passed**. It includes four new tests for pooling, matched observation populations, fair mean MSE, and missing event targets.
- Primary marginal scores were independently recomputed from actual members in float64, then cast to their stored float32 dtype for exact comparison. Increment scores were recomputed from members. Gaussian analytic scores and centered analytic means were independently checked.
- All three summary code hashes, dependency hashes, and 24 per-row event threshold lock hashes match actual files. Calendar protocol, origin hours, query identity hashes and registered order are checked.
- An independent read-only reviewer exactly recomputed all paired values and month intervals, including **60 comparisons at the 20mm event threshold**. Final findings: **Critical 0 / Important 0 / Minor 0**.
- Report and all three performance figures were inspected. New local-document evidence links resolve.
- Interpretation: weather-GP near-pair gains reproduce on supplemental dates; Local adds small F6 marginal/pair gains over Affine with an increment cost. Centered shape does not establish a comprehensive advantage over Gaussian scale calibration. Current point prediction stays with the deterministic base.
- Supplemental G265 evidence does not validate the new Local/Affine on those dates; the old Local event audit is separately labeled. Month intervals condition on two fixed training seeds and are exploratory, not population-of-seeds or blind-year uncertainty.
- Git boundary: only code and lightweight protocols, curves, source seals, scores, reports and figures. No raw arrays, model weights or caches.
