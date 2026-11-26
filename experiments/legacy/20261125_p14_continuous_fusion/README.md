# 20261125_p14_continuous_fusion

当前公开来源：`experiments/us/20261125_p14_continuous_fusion`；[来源目录](../../us/20261125_p14_continuous_fusion)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_p14_continuous_fusion`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261125_p14_continuous_fusion/REPORT.md](../../us/20261125_p14_continuous_fusion/REPORT.md)，第1行：# p14与新空间特征及时间模块：连续3000步融合实验

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261125_p14_continuous_fusion/protocol.json](../../us/20261125_p14_continuous_fusion/protocol.json)，status：RUNNING_MATCHED_CURRENT_FIELD_FUSION
- [experiments/us/20261125_p14_continuous_fusion/status.json](../../us/20261125_p14_continuous_fusion/status.json)，status：COMPLETE_CONTINUOUS_P14_FUSION

## 报告中的阶段声明

- [experiments/us/20261125_p14_continuous_fusion/REPORT.md](../../us/20261125_p14_continuous_fusion/REPORT.md)，第3行：状态：全部完成。原p14最佳两种子权重已恢复，旧2018合同指标精确复现。

## 历史证据范围与信息条件

- [experiments/us/20261125_p14_continuous_fusion/REPORT.md](../../us/20261125_p14_continuous_fusion/REPORT.md)，第8行：无随机日期采样：34993个独立目标小时完整遍历一次，再重复开头996小时，共35989目标小时呈现。窗口历史72小时，块内12小时连续。
- [experiments/us/20261125_p14_continuous_fusion/REPORT.md](../../us/20261125_p14_continuous_fusion/REPORT.md)，第22行：冻结基线与续训p14区分数据续训效应；features与续训p14比较新空间特征；时间组与features比较额外历史编码。

## 原始协议、结果与报告

- [experiments/us/20261125_p14_continuous_fusion/REPORT.md](../../us/20261125_p14_continuous_fusion/REPORT.md)（report；read）
- [experiments/us/20261125_p14_continuous_fusion/protocol.json](../../us/20261125_p14_continuous_fusion/protocol.json)（protocol；read）
- [experiments/us/20261125_p14_continuous_fusion/status.json](../../us/20261125_p14_continuous_fusion/status.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/summary.json](../../us/20261125_p14_continuous_fusion/summary.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/verification.json](../../us/20261125_p14_continuous_fusion/verification.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/features_s17/metrics.json](../../us/20261125_p14_continuous_fusion/features_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/features_s17/progress.json](../../us/20261125_p14_continuous_fusion/features_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/features_s42/metrics.json](../../us/20261125_p14_continuous_fusion/features_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/features_s42/progress.json](../../us/20261125_p14_continuous_fusion/features_s42/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/features_s42_interrupted_1795589658/progress.json](../../us/20261125_p14_continuous_fusion/features_s42_interrupted_1795589658/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/frozen_s17/metrics.json](../../us/20261125_p14_continuous_fusion/frozen_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/frozen_s42/metrics.json](../../us/20261125_p14_continuous_fusion/frozen_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/gru_s17/metrics.json](../../us/20261125_p14_continuous_fusion/gru_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/gru_s17/progress.json](../../us/20261125_p14_continuous_fusion/gru_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/gru_s42/metrics.json](../../us/20261125_p14_continuous_fusion/gru_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/gru_s42/progress.json](../../us/20261125_p14_continuous_fusion/gru_s42/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/lstm_s17/metrics.json](../../us/20261125_p14_continuous_fusion/lstm_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/lstm_s17/progress.json](../../us/20261125_p14_continuous_fusion/lstm_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/lstm_s42/metrics.json](../../us/20261125_p14_continuous_fusion/lstm_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/lstm_s42/progress.json](../../us/20261125_p14_continuous_fusion/lstm_s42/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/mlp_s17/metrics.json](../../us/20261125_p14_continuous_fusion/mlp_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/mlp_s17/progress.json](../../us/20261125_p14_continuous_fusion/mlp_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/mlp_s42/metrics.json](../../us/20261125_p14_continuous_fusion/mlp_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/mlp_s42/progress.json](../../us/20261125_p14_continuous_fusion/mlp_s42/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/p14_s17/metrics.json](../../us/20261125_p14_continuous_fusion/p14_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/p14_s17/progress.json](../../us/20261125_p14_continuous_fusion/p14_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/p14_s42/metrics.json](../../us/20261125_p14_continuous_fusion/p14_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/p14_s42/progress.json](../../us/20261125_p14_continuous_fusion/p14_s42/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/transformer_s17/metrics.json](../../us/20261125_p14_continuous_fusion/transformer_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/transformer_s17/progress.json](../../us/20261125_p14_continuous_fusion/transformer_s17/progress.json)（status；read）
- [experiments/us/20261125_p14_continuous_fusion/transformer_s42/metrics.json](../../us/20261125_p14_continuous_fusion/transformer_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_p14_continuous_fusion/transformer_s42/progress.json](../../us/20261125_p14_continuous_fusion/transformer_s42/progress.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
