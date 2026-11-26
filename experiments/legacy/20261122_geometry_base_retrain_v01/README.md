# 20261122_geometry_base_retrain_v01

当前公开来源：`experiments/us/20261122_geometry_base_retrain_v01`；[来源目录](../../us/20261122_geometry_base_retrain_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_geometry_base_retrain_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)，第1行：# Geometry-corrected frozen base retraining plan

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261122_geometry_base_retrain_v01/status.json](../../us/20261122_geometry_base_retrain_v01/status.json)，stage：ANNUAL_PASSED
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/status.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/status.json)，stage：RETRAIN

## 报告中的阶段声明

- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)，第13行：5. Fresh seed42 training, common new base and calibrated prior: plain GAN, plain diffusion, pointwise observation DSM with independent noise and pointwise DSM with calibrated GP correlation noise. No attention/MP/noise-injection architecture extensions. GAN/DF each72 updates, effective8 hours/micro1/K4: one complete pass over576 hours. NDP each1440 updates/batch8. Both folds independently. Preserve paired initialization and deterministic schedule. Old48-step model scores are context, not a causal matched retraining control.
- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)，第23行：- [ ] `train_new_base.py`: actual sequential optimization, locks, finite checks, complete optimizer/RNG/cursor checkpoints, inner-only products and outer evaluation.
- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)，第26行：Review focus: old generation-condition69 differs from ridge-readout69; residual supervision must subtract the new fold's mean; new mean can depend on historical fitted B labels only via frozen coefficients; Gaussian fit uses current training role only; no doubled observation noise; continuous field query must agree with point predictions; completed outputs require hash validation; no hidden reuse of old residual-head weights.

## 历史证据范围与信息条件

- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)，第10行：2. Full2019: compute all8760 hours, same110 metadata stations /76 scored stations /642258 station-hours, same archived population. Commit predictions before opening archived labels. Report coarse,IDW,M0,old mixture,newfold0,newfold1. Save M0 grids monthly for reuse. C is already exposed development; D2020 remains sealed. Gate for this retraining: both new fold means must improve full-year station-macro RMSE relative to old mixture and directIDW. If this fails, stop at the mean validation report; do not tune to C.

## 原始协议、结果与报告

- [experiments/us/20261122_geometry_base_retrain_v01/PLAN.md](../../us/20261122_geometry_base_retrain_v01/PLAN.md)（documentation；read）
- [experiments/us/20261122_geometry_base_retrain_v01/mean.lock.json](../../us/20261122_geometry_base_retrain_v01/mean.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/probability.lock.json](../../us/20261122_geometry_base_retrain_v01/probability.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/probability_v01.lock.json](../../us/20261122_geometry_base_retrain_v01/probability_v01.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/status.json](../../us/20261122_geometry_base_retrain_v01/status.json)（status；read）
- [experiments/us/20261122_geometry_base_retrain_v01/B_cache/lock.json](../../us/20261122_geometry_base_retrain_v01/B_cache/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/annual2019/lock.json](../../us/20261122_geometry_base_retrain_v01/annual2019/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/bridge/config.json](../../us/20261122_geometry_base_retrain_v01/bridge/config.json)（metadata；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v04_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v04_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v04_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v04_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v05_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v05_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v05_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/diffusion/local_v05_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v02_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v02_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v02_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v02_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_repeat/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_repeat/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v03_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v04_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v04_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v04_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v04_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v05_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v05_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/gan/local_v05_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/gan/local_v05_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v04_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v04_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v04_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v04_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v05_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v05_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v05_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_GP_point/local_v05_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v04_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v04_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v04_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v04_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v05_continuous/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v05_continuous/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v05_split/lock.json](../../us/20261122_geometry_base_retrain_v01/engineering/ndp_point/local_v05_split/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/allocation.lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/allocation.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/mean.lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/mean.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/probability.lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/probability.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/status.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/status.json)（status；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/training.lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/training.lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/diffusion/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/diffusion/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/gan/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/gan/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/ndp_GP_point/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/ndp_GP_point/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/ndp_point/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_0/ndp_point/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/diffusion/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/diffusion/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/gan/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/gan/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/ndp_GP_point/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/ndp_GP_point/lock.json)（protocol；read）
- [experiments/us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/ndp_point/lock.json](../../us/20261122_geometry_base_retrain_v01/received_20261122_212219/fold_1/ndp_point/lock.json)（protocol；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
