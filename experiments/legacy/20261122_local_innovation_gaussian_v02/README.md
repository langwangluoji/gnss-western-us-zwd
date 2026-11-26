# 20261122_local_innovation_gaussian_v02

当前公开来源：`experiments/us/20261122_local_innovation_gaussian_v02`；[来源目录](../../us/20261122_local_innovation_gaussian_v02)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_local_innovation_gaussian_v02`。

用途类别：`diagnostic`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_local_innovation_gaussian_v02/PROGRESS.md](../../us/20261122_local_innovation_gaussian_v02/PROGRESS.md)，第1行：# Local experiment v02

## 类别依据

- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)，第46行：## 空间与采样补充诊断

## 状态依据

- [experiments/us/20261122_local_innovation_gaussian_v02/status.json](../../us/20261122_local_innovation_gaussian_v02/status.json)，stage：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/constant/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/constant/continuous/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/constant/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/constant/split/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/free/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/free/continuous/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/free/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/free/split/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/gated/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/gated/continuous/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/gated/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/gated/split/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/constant/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/constant/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/free/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/free/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/gated/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/gated/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/constant/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/constant/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/free/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/free/status.json)，status：COMPLETE
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/gated/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/gated/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261122_local_innovation_gaussian_v02/PROGRESS.md](../../us/20261122_local_innovation_gaussian_v02/PROGRESS.md)，第8行：- Complete: all6 scientific arms84/84 updates; actual recovery3arms bitwise; corrected Gaussian bootstrap100/100 per fold; paired weather/group intervals2000; K64→128 Gaussian rank stable.

## 历史证据范围与信息条件

- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)，第3行：本轮完成了三个点式均值头的两折、两遍小训练，以及冻结均值后的Gaussian误差分配审计。小训练使用336小时，每折84次更新；内层为四个新天气块内41个缓存时刻和独立物理/50 km站组。外层为原12个注册天气周内288个缓存锚点，折0/1分别204/244站。这是B2018增量开发，不与上轮96小时或C2019全年数字直接排序。
- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)，第22行：所有概率模型使用同一入选均值，缩放独立项不会改变RMSE。两个尺度是统计残差分配，不是仪器噪声的物理测定。二维NLL及100次同步天气块×原200 km站群重采样只作小样本稳定性诊断；旧版50 km重采样改变了估计目标，已判无效并由reviewed结果替代。
- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)，第29行：12个天气簇各为缓存锚点，不是12个全小时完整周；原底座与概率先验有历史B暴露。C2019未读取评分，D2020保持封存。
- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)，第37行：本轮没有训练Diffusion/GAN，没有提交服务器作业。数学/查询和恢复验收用于确认工程路径；这些检查不替代独立站精度证据。

## 原始协议、结果与报告

- [experiments/us/20261122_local_innovation_gaussian_v02/PROGRESS.md](../../us/20261122_local_innovation_gaussian_v02/PROGRESS.md)（documentation；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/RESULTS.md](../../us/20261122_local_innovation_gaussian_v02/RESULTS.md)（report；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/experiment.lock.json](../../us/20261122_local_innovation_gaussian_v02/experiment.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/point_results.json](../../us/20261122_local_innovation_gaussian_v02/point_results.json)（metadata；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/status.json](../../us/20261122_local_innovation_gaussian_v02/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/constant/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/constant/continuous/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/constant/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/constant/split/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/free/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/free/continuous/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/free/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/free/split/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/gated/continuous/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/gated/continuous/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/engineering/gated/split/status.json](../../us/20261122_local_innovation_gaussian_v02/engineering/gated/split/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/point_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/point_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/probability_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/probability_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/constant/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/constant/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/free/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/free/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/gated/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/gated/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/point_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/point_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/probability_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/probability_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/probability_selection_v03.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_0/reviewed/probability_selection_v03.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/point_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/point_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/probability_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/probability_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/constant/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/constant/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/free/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/free/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/gated/status.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/gated/status.json)（status；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/point_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/point_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/probability_selection.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/probability_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/probability_selection_v03.lock.json](../../us/20261122_local_innovation_gaussian_v02/fold_1/reviewed/probability_selection_v03.lock.json)（protocol；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
