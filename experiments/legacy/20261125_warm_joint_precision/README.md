# 20261125_warm_joint_precision

当前公开来源：`experiments/us/20261125_warm_joint_precision`；[来源目录](../../us/20261125_warm_joint_precision)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_warm_joint_precision`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261125_warm_joint_precision/protocol.json](../../us/20261125_warm_joint_precision/protocol.json)，objective：Improve matched ~10mm best p14 fusion; preserve exact initialized reference

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261125_warm_joint_precision/status.json](../../us/20261125_warm_joint_precision/status.json)，status：COMPLETE_MATCHED_WARM_FUSION

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261125_warm_joint_precision/REPORT.md](../../us/20261125_warm_joint_precision/REPORT.md)，第3行：参考保留原GRU+p14全部分支与归一化。所有候选训练3000步，2016–2019连续12小时批、仅条件站抽样；原2020每月72小时×3布局×137登记开发站评价。该开发队列已暴露。
- [experiments/us/20261125_warm_joint_precision/combined_precision/REPORT.md](../../us/20261125_warm_joint_precision/combined_precision/REPORT.md)，第3行：基准使用原GRU模型保存的2020预测，逐项核对真值与源权重哈希。所有结果为相同12个月×72小时×3布局，登记137站、135站有真值、327330个有效布局-小时-站记录。2020属于已暴露开发队列。
- [experiments/us/20261125_warm_joint_precision/combined_precision/REPORT.md](../../us/20261125_warm_joint_precision/combined_precision/REPORT.md)，第38行：单位均为mm。神经候选均采用最终3000次梯度更新；I系列采用3000个连续块的解析拟合。Q/S/A/D各自的raw与EMA来自同一训练轨迹，不计为独立训练重复。T5_joint额外解冻旧时间模块原先固定为零的查询头518参数，应解释为“联合微调＋旧查询头激活”。
- [experiments/us/20261125_warm_joint_precision/combined_precision/REPORT.md](../../us/20261125_warm_joint_precision/combined_precision/REPORT.md)，第61行：## 输入、损失及创新历史的直接配对
- [experiments/us/20261125_warm_joint_precision/combined_precision/REPORT.md](../../us/20261125_warm_joint_precision/combined_precision/REPORT.md)，第92行：连续块内的一小时增量独立评分，跨月份不连差。所有五项主指标采用float64独立复算。

## 原始协议、结果与报告

- [experiments/us/20261125_warm_joint_precision/REPORT.md](../../us/20261125_warm_joint_precision/REPORT.md)（report；read）
- [experiments/us/20261125_warm_joint_precision/protocol.json](../../us/20261125_warm_joint_precision/protocol.json)（protocol；read）
- [experiments/us/20261125_warm_joint_precision/status.json](../../us/20261125_warm_joint_precision/status.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/summary.json](../../us/20261125_warm_joint_precision/summary.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/verification.json](../../us/20261125_warm_joint_precision/verification.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T2_frozen_s17/metrics.json](../../us/20261125_warm_joint_precision/T2_frozen_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T2_frozen_s17/progress.json](../../us/20261125_warm_joint_precision/T2_frozen_s17/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/T2_frozen_s42/metrics.json](../../us/20261125_warm_joint_precision/T2_frozen_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T2_frozen_s42/progress.json](../../us/20261125_warm_joint_precision/T2_frozen_s42/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/T5_frozen_s17/metrics.json](../../us/20261125_warm_joint_precision/T5_frozen_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T5_frozen_s17/progress.json](../../us/20261125_warm_joint_precision/T5_frozen_s17/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/T5_frozen_s42/metrics.json](../../us/20261125_warm_joint_precision/T5_frozen_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T5_frozen_s42/progress.json](../../us/20261125_warm_joint_precision/T5_frozen_s42/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/T5_joint_s17/metrics.json](../../us/20261125_warm_joint_precision/T5_joint_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T5_joint_s17/progress.json](../../us/20261125_warm_joint_precision/T5_joint_s17/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/T5_joint_s42/metrics.json](../../us/20261125_warm_joint_precision/T5_joint_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/T5_joint_s42/progress.json](../../us/20261125_warm_joint_precision/T5_joint_s42/progress.json)（status；read）
- [experiments/us/20261125_warm_joint_precision/combined_precision/REPORT.md](../../us/20261125_warm_joint_precision/combined_precision/REPORT.md)（report；read）
- [experiments/us/20261125_warm_joint_precision/combined_precision/comparison.json](../../us/20261125_warm_joint_precision/combined_precision/comparison.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/combined_precision/verification.json](../../us/20261125_warm_joint_precision/combined_precision/verification.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/reference_s17/metrics.json](../../us/20261125_warm_joint_precision/reference_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_warm_joint_precision/reference_s42/metrics.json](../../us/20261125_warm_joint_precision/reference_s42/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
