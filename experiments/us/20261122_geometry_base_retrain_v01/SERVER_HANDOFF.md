# Geometry-base A800 retraining handoff

Last verified: 2026-11-22 20:20:43 UTC+8

- Slurm job: 3688243; allocation: one A800; cumulative cap: 3600 seconds.
- Release: $HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/code/20261122_geometry_base_retrain_v01
- Run: /share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261122_us_dense_zwd_geometry_base_a800_v01
- Current stage: {'arm': 'gan', 'fold': 0, 'product': 'joint', 'stage': 'OUTER', 'updated': 1795350007.6752138}
- Engineering accepted: True; measured paths: batch8, continuous4 versus2+2, exact RNG/nonfloating state, floating tolerance1e-7, forecast tolerance0.001mm.
- Benchmark including1.5x margin: 3034.8074248172343 seconds. This is a runtime estimate, not a quality result.
- Sequential arms per fold: GAN72, diffusion72, independent-noise point NDP1440, GP-noise point NDP1440; folds0 and1. M0 and geometry mean remain frozen.
- Metrics/checkpoints: fold_0/<arm>/ and fold_1/<arm>/; training.jsonl, inner_*.json, result.json, candidate_*.pt, latest.pt, point_predictions.npz, joint_predictions.npz. Final aggregate: results.json.
- Selection: inner12 fixed monthly B2018 hours; point macroRMSE and joint groupES separately. Outer12 fixed monthly B2018 hours with128 members. No C2019 evaluation of these residual heads; D2020 remains sealed.
- Pause: create PAUSE inside the run directory. The current optimizer update finishes and a checkpoint is retained. Resume requires removing PAUSE and explicit server_submit.py --resume with the same release/run; previous allocation time is deducted and active jobs are deduplicated. No automatic requeue.
- Local SSH bridge refreshes server_status.json every20 seconds while active. It stays idle after terminal status; touching bridge/request_status triggers a refresh. No credentials are saved.
