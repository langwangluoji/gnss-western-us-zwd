# 20261126_within_forecast_increment

当前公开来源：`experiments/us/20261126_within_forecast_increment`；[来源目录](../../us/20261126_within_forecast_increment)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_within_forecast_increment`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_within_forecast_increment/paired_calendar15/REPORT.md](../../us/20261126_within_forecast_increment/paired_calendar15/REPORT.md)，第1行：# 固定H架构：内部增量监督的真实训练结果

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_within_forecast_increment/H_delta_s17/status.json](../../us/20261126_within_forecast_increment/H_delta_s17/status.json)，status：COMPLETE_MEAN_INCREMENT
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/status.json](../../us/20261126_within_forecast_increment/H_delta_s42/status.json)，status：COMPLETE_MEAN_INCREMENT

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_within_forecast_increment/paired_calendar15/REPORT.md](../../us/20261126_within_forecast_increment/paired_calendar15/REPORT.md)，第3行：2016–2019原连续缓存，同源TF3000各新增3000，种子42/17；2020每月15日起72h三布局，已暴露开发。raw final3000为主，EMA并列。H_delta唯一训练变化是原query目标加0.5×同起点内部lead差分损失；原跨起点h0差分保留。
- [experiments/us/20261126_within_forecast_increment/paired_calendar8/REPORT.md](../../us/20261126_within_forecast_increment/paired_calendar8/REPORT.md)，第3行：2016–2019原连续缓存，同源TF3000各新增3000，种子42/17；2020每月8日起72h三布局，已暴露开发。raw final3000为主，EMA并列。H_delta唯一训练变化是原query目标加0.5×同起点内部lead差分损失；原跨起点h0差分保留。

## 原始协议、结果与报告

- [experiments/us/20261126_within_forecast_increment/H_delta_ema_dev_s17/manifest.json](../../us/20261126_within_forecast_increment/H_delta_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_ema_dev_s42/manifest.json](../../us/20261126_within_forecast_increment/H_delta_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_last_dev_s17/manifest.json](../../us/20261126_within_forecast_increment/H_delta_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_last_dev_s42/manifest.json](../../us/20261126_within_forecast_increment/H_delta_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s17/progress.json](../../us/20261126_within_forecast_increment/H_delta_s17/progress.json)（status；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s17/protocol.json](../../us/20261126_within_forecast_increment/H_delta_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s17/status.json](../../us/20261126_within_forecast_increment/H_delta_s17/status.json)（status；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s17/eval_ema/metrics.json](../../us/20261126_within_forecast_increment/H_delta_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s17/eval_last/metrics.json](../../us/20261126_within_forecast_increment/H_delta_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/progress.json](../../us/20261126_within_forecast_increment/H_delta_s42/progress.json)（status；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/protocol.json](../../us/20261126_within_forecast_increment/H_delta_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/status.json](../../us/20261126_within_forecast_increment/H_delta_s42/status.json)（status；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/eval_ema/metrics.json](../../us/20261126_within_forecast_increment/H_delta_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/H_delta_s42/eval_last/metrics.json](../../us/20261126_within_forecast_increment/H_delta_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/paired_calendar15/REPORT.md](../../us/20261126_within_forecast_increment/paired_calendar15/REPORT.md)（report；read）
- [experiments/us/20261126_within_forecast_increment/paired_calendar15/summary.json](../../us/20261126_within_forecast_increment/paired_calendar15/summary.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/paired_calendar8/REPORT.md](../../us/20261126_within_forecast_increment/paired_calendar8/REPORT.md)（report；read）
- [experiments/us/20261126_within_forecast_increment/paired_calendar8/summary.json](../../us/20261126_within_forecast_increment/paired_calendar8/summary.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_ema_s17/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_delta_ema_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_ema_s17/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_delta_ema_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_ema_s42/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_delta_ema_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_ema_s42/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_delta_ema_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_last_s17/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_delta_last_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_last_s17/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_delta_last_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_last_s42/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_delta_last_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_delta_last_s42/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_delta_last_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_ema_s17/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_ema_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_ema_s17/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_ema_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_ema_s42/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_ema_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_ema_s42/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_ema_s42/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_last_s17/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_last_s17/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_last_s17/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_last_s17/evaluation/metrics.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_last_s42/embedding/manifest.json](../../us/20261126_within_forecast_increment/supplement/H_last_s42/embedding/manifest.json)（metadata；read）
- [experiments/us/20261126_within_forecast_increment/supplement/H_last_s42/evaluation/metrics.json](../../us/20261126_within_forecast_increment/supplement/H_last_s42/evaluation/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
