# 20261126_directed_lag_v2

当前公开来源：`experiments/us/20261126_directed_lag_v2`；[来源目录](../../us/20261126_directed_lag_v2)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_directed_lag_v2`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_directed_lag_v2/paired_calendar15/REPORT.md](../../us/20261126_directed_lag_v2/paired_calendar15/REPORT.md)，第1行：# 有向站点×滞后传递：同源同预算的真实训练结果

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_directed_lag_v2/factorized_s17/status.json](../../us/20261126_directed_lag_v2/factorized_s17/status.json)，status：COMPLETE_DIRECTED_LAG_V2
- [experiments/us/20261126_directed_lag_v2/factorized_s42/status.json](../../us/20261126_directed_lag_v2/factorized_s42/status.json)，status：COMPLETE_DIRECTED_LAG_V2
- [experiments/us/20261126_directed_lag_v2/joint_s17/status.json](../../us/20261126_directed_lag_v2/joint_s17/status.json)，status：COMPLETE_DIRECTED_LAG_V2
- [experiments/us/20261126_directed_lag_v2/joint_s42/status.json](../../us/20261126_directed_lag_v2/joint_s42/status.json)，status：COMPLETE_DIRECTED_LAG_V2

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_directed_lag_v2/paired_calendar15/REPORT.md](../../us/20261126_directed_lag_v2/paired_calendar15/REPORT.md)，第3行：2016–2019原连续缓存，同源TF3000各新增3000，种子42/17；2020每月15日起72h三布局，已暴露开发。raw final3000为主，EMA并列。H_delta为匹配对照；factorized/joint增加同参数四头数值传递，joint使用历史key和空间/lag交互。三者损失均含0.5内部增量及原跨起点差分。
- [experiments/us/20261126_directed_lag_v2/paired_calendar8/REPORT.md](../../us/20261126_directed_lag_v2/paired_calendar8/REPORT.md)，第3行：2016–2019原连续缓存，同源TF3000各新增3000，种子42/17；2020每月8日起72h三布局，已暴露开发。raw final3000为主，EMA并列。H_delta为匹配对照；factorized/joint增加同参数四头数值传递，joint使用历史key和空间/lag交互。三者损失均含0.5内部增量及原跨起点差分。
- [experiments/us/20261126_directed_lag_v2/station_tail_audit/REPORT.md](../../us/20261126_directed_lag_v2/station_tail_audit/REPORT.md)，第3行：同两种子、raw final3000、已暴露2020抽样日期。站点RMSE按每布局分别对月份/小时计算后平均布局；宏RMSE再平均站点与种子，与原评价一致。micro/MAE/Bias/P90/P95先每种子汇总全部布局再平均种子。P90/P95为绝对误差分位数，单位mm。无标签站记NA，地图为站点值，不代表格网真值。
- [experiments/us/20261126_directed_lag_v2/station_tail_audit/REPORT.md](../../us/20261126_directed_lag_v2/station_tail_audit/REPORT.md)，第37行：完整137站、两日历和来源见summary.json。逐站描述不替代主报告的共同月份整体区间，当前结果不作完整年度或独立确认结论。

## 原始协议、结果与报告

- [experiments/us/20261126_directed_lag_v2/factorized_ema_dev_s17/manifest.json](../../us/20261126_directed_lag_v2/factorized_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_ema_dev_s42/manifest.json](../../us/20261126_directed_lag_v2/factorized_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_last_dev_s17/manifest.json](../../us/20261126_directed_lag_v2/factorized_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_last_dev_s42/manifest.json](../../us/20261126_directed_lag_v2/factorized_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s17/progress.json](../../us/20261126_directed_lag_v2/factorized_s17/progress.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s17/protocol.json](../../us/20261126_directed_lag_v2/factorized_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s17/status.json](../../us/20261126_directed_lag_v2/factorized_s17/status.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s17/eval_ema/metrics.json](../../us/20261126_directed_lag_v2/factorized_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s17/eval_last/metrics.json](../../us/20261126_directed_lag_v2/factorized_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s42/progress.json](../../us/20261126_directed_lag_v2/factorized_s42/progress.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s42/protocol.json](../../us/20261126_directed_lag_v2/factorized_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s42/status.json](../../us/20261126_directed_lag_v2/factorized_s42/status.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s42/eval_ema/metrics.json](../../us/20261126_directed_lag_v2/factorized_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/factorized_s42/eval_last/metrics.json](../../us/20261126_directed_lag_v2/factorized_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_ema_dev_s17/manifest.json](../../us/20261126_directed_lag_v2/joint_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_ema_dev_s42/manifest.json](../../us/20261126_directed_lag_v2/joint_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_last_dev_s17/manifest.json](../../us/20261126_directed_lag_v2/joint_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_last_dev_s42/manifest.json](../../us/20261126_directed_lag_v2/joint_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_s17/progress.json](../../us/20261126_directed_lag_v2/joint_s17/progress.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/joint_s17/protocol.json](../../us/20261126_directed_lag_v2/joint_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_directed_lag_v2/joint_s17/status.json](../../us/20261126_directed_lag_v2/joint_s17/status.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/joint_s17/eval_ema/metrics.json](../../us/20261126_directed_lag_v2/joint_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_s17/eval_last/metrics.json](../../us/20261126_directed_lag_v2/joint_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_s42/progress.json](../../us/20261126_directed_lag_v2/joint_s42/progress.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/joint_s42/protocol.json](../../us/20261126_directed_lag_v2/joint_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_directed_lag_v2/joint_s42/status.json](../../us/20261126_directed_lag_v2/joint_s42/status.json)（status；read）
- [experiments/us/20261126_directed_lag_v2/joint_s42/eval_ema/metrics.json](../../us/20261126_directed_lag_v2/joint_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/joint_s42/eval_last/metrics.json](../../us/20261126_directed_lag_v2/joint_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/paired_calendar15/REPORT.md](../../us/20261126_directed_lag_v2/paired_calendar15/REPORT.md)（report；read）
- [experiments/us/20261126_directed_lag_v2/paired_calendar15/summary.json](../../us/20261126_directed_lag_v2/paired_calendar15/summary.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/paired_calendar8/REPORT.md](../../us/20261126_directed_lag_v2/paired_calendar8/REPORT.md)（report；read）
- [experiments/us/20261126_directed_lag_v2/paired_calendar8/summary.json](../../us/20261126_directed_lag_v2/paired_calendar8/summary.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/station_tail_audit/REPORT.md](../../us/20261126_directed_lag_v2/station_tail_audit/REPORT.md)（report；read）
- [experiments/us/20261126_directed_lag_v2/station_tail_audit/summary.json](../../us/20261126_directed_lag_v2/station_tail_audit/summary.json)（metadata；linked_only_size_limit）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_ema_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/H_delta_ema_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_ema_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/H_delta_ema_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_ema_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/H_delta_ema_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_ema_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/H_delta_ema_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_last_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/H_delta_last_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_last_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/H_delta_last_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_last_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/H_delta_last_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/H_delta_last_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/H_delta_last_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_ema_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/factorized_ema_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_ema_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/factorized_ema_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_ema_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/factorized_ema_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_ema_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/factorized_ema_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_last_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/factorized_last_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_last_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/factorized_last_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_last_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/factorized_last_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/factorized_last_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/factorized_last_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_ema_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/joint_ema_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_ema_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/joint_ema_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_ema_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/joint_ema_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_ema_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/joint_ema_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_last_s17/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/joint_last_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_last_s17/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/joint_last_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_last_s42/embedding/manifest.json](../../us/20261126_directed_lag_v2/supplement/joint_last_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_directed_lag_v2/supplement/joint_last_s42/evaluation/metrics.json](../../us/20261126_directed_lag_v2/supplement/joint_last_s42/evaluation/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
