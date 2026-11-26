# Weather GP / trajectory transport independent review

Scope: 0fcb892..37847b5; reviewer review_weather_gp_transport; read-only CPU review.

Independent14 tests passed. Direct active-Affine composition vs actual sampler: Heun0 max5.02e-7, DDIM40 max1.37e-6. No Critical mathematical issue found.

Important findings addressed prospectively:

- Lock formal transport to G265 and matching seed; generic GP query audit remains175/265.
- Bind last weights to final3000 checkpoint tensor-by-tensor, protocol/status/code SHA seal; formal consumers and summary verify it. Active GP computational files/locks unchanged; wrapper publishes receipts.
- Require complete metrics/lineage/array and matching model receipt, rather than metrics alone. Interrupted directories remain preserved and fail explicitly.
- Correct training data description: whole-block future label availability influences128 target IDs; fixed-query condition and development137-query inference use history only.

Old4-update transport source copied byte-for-byte under engineering_snapshots/weather_transport_v1. Sealed version independently reran local/affine4 updates1435 trajectories each, and137-query path probes. Native suite168 passed49.90s, default568 passed293.82s, maintenance12 passed0.45s.

Deferred minor: old GP codes omit native_weather_evaluation.py. Scoring source unchanged in this campaign; derived results checked from saved arrays. Separate scoring receipt remains follow-up.

Declined to judge: empirical formal GP performance, Local/Affine extra gains and confirmation performance; those require actual results and are not accepted as proven.
