# v12 overnight residual field results

Queue partial; status.json is authoritative.

Training410/validation137; allowed ZWD/T/P only. Frozen confirmation138 and C/D unread. Three spatial-fold foundation predictions train downstream models; final foundations are refitted. Same2018 development calendar, not a blind test. Radial CNN-LSTM uses12h observable history; stochastic temporal coherence is not trained.

| Experiment | Macro RMSE/mm | Fair CRPS/mm | 1h increment CRPS/mm |
|---|---:|---:|---:|
| E1201_mlp | 12.0406 | 9.2899 | 4.4011 |
| E1202_cnn_lstm | 12.4013 | 9.5572 | 4.2113 |
| E1203_xgboost | 12.0196 | 9.2065 | 4.7108 |

K1 deterministic CRPS is MAE; probability arms K16 at expansion (matches post-calibration ensemble features). Independent selection K16. Baseline is selected by two-seed quarterly RMSE. Model-family budgets include its own fold fitting; downstream dependencies are extra cost. Trees can early-stop. Fixed spatial Gaussian covariance is a meaningful correlated baseline, not a learned temporal model. Diffusion has full point residual support but retains query-graph dependence. Post-XGBoost common correction preserves ensemble spread. Historical PWV station-cohort selection remains.

![Comparison](comparison.png)

Source plan: docs/superpowers/plans/2026-11-24-residual-field-v12.md. Per-run folders contain metrics, curves, timing and checkpoints (weights/arrays local). Final interpretation to follow after frozen evaluations and diagnostic review.
