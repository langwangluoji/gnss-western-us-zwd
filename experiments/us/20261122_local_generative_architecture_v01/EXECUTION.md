# Execution ledger

2026-11-22 local-only research spike. User approved the four prior architecture proposals and NDP-inspired local trial. Native implementation; no git checkout exists at this path. No server jobs, package installation or production-source edits.

- Full-grid912-hour input-only cache built169.1s; shared compact67 and mean query differences0.
- Full35-call CUDA backward measured; functional MP no parameter mutation.
- Independent reviewer local_architecture_review found three P2 issues: completed-result identity validation, cache-content validation at training startup, and shared time-conditioning initialization for MP. All fixed with new failing-then-passing tests.
- Final joint test run:343 passed57.09s (332 existing +11 prototype tests).
- Data acceptance: prior matrix/grid query5.722e-6mm, held-label perturbation0, initial generator/Gaussian7.629e-6mm.
- Original cache-builder source saved as cache_builder_snapshot.py and matched exact contract SHA256. Subsequent accessor/validation additions do not change the built input function.
- Ruling: short local screen uses48 grid updates/effective batch2 and480 observation-DSM updates/batch8, two folds. Full widths/grid/35NFE are retained; learning-rate schedule is explicitly local. Cost if insufficient: short-screen non-improvement cannot reject the architecture.
- Ruling: NDP branch is a finite-coordinate observation-space denoiser inspired by NDP, paired with a pointwise denoiser control. It is not original-paper reproduction or an accepted continuous-field product; no extra tau at inference.
- Ruling: GAN adversarial coefficient0.1 follows actual old training code; old JSON0.01 mismatch documented in PLAN. All new GAN arms use same0.1.
- Completed arms must validate metadata plus selected checkpoint before reuse; all input rows hash-checked before scientific training.

Training started after test success; see status.json and training.log. Stop with PAUSE file; restart same run_pilot.py command after removing PAUSE. Checkpoints retained at20/24/final grid updates and160-step DSM intervals. Earlier unsaved updates may be repeated on interruption; this is a research prototype.

## Final execution results

- Original16 fits completed; adaptive GP-noise follow-up added4 fresh480-step fits. Total20 actual trained models +4 Gaussian controls,4416 scientific updates +288 critic updates.
- Follow-up independent review found no model-math or label-isolation issue; sampling dependency hash binding was fixed with a new RED-to-GREEN regression. Final suite345 passed57.06s.
- First sampling audit was deliberately stopped during read-only postprocessing for dependency binding and inference batching. Training/checkpoints were not stopped or changed. v01 partial output and exact source snapshot retained; v02 authoritative audit passed checkpoint/model/factor hashes before and after all sampling.
- K32/128/512 fixed-prefix audit completed for both folds and all four NDP models plus matched Gaussian. Estimator-only control variate preserves full ensemble forecasts.
- Raw updated grid models all generated nonzero corrections; max member-chunk difference0.000102997mm.
- Final numerical audit reproduced every K32 point metric, validated all selected checkpoint hashes and actual log cursors. Completion.json records24 cohort counts.
- Scientific result: no stable server-promotion advantage in this local screen. Correlated noise improves VS versus independent noise, but not joint ES; point-estimate differences have exploratory intervals crossing0.
- Rendered and visually inspected training curves, K32 metric figure, fixed-checkpoint sampling estimator figure, field examples and supported-land base audit.
- All training and postprocessing finished. Production sources remain unchanged, and no server job or package installation occurred.
