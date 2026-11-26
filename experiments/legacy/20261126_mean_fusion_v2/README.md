# 20261126_mean_fusion_v2

当前公开来源：`experiments/us/20261126_mean_fusion_v2`；[来源目录](../../us/20261126_mean_fusion_v2)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_mean_fusion_v2`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_mean_fusion_v2/paired/REPORT.md](../../us/20261126_mean_fusion_v2/paired/REPORT.md)，第1行：# 同历史逐站时间编码：双种子配对结果

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_mean_fusion_v2/C_s17/status.json](../../us/20261126_mean_fusion_v2/C_s17/status.json)，status：COMPLETE_MEAN
- [experiments/us/20261126_mean_fusion_v2/C_s42/status.json](../../us/20261126_mean_fusion_v2/C_s42/status.json)，status：COMPLETE_MEAN
- [experiments/us/20261126_mean_fusion_v2/H_s17/status.json](../../us/20261126_mean_fusion_v2/H_s17/status.json)，status：COMPLETE_MEAN
- [experiments/us/20261126_mean_fusion_v2/H_s42/status.json](../../us/20261126_mean_fusion_v2/H_s42/status.json)，status：COMPLETE_MEAN
- [experiments/us/20261126_mean_fusion_v2/S_s17/status.json](../../us/20261126_mean_fusion_v2/S_s17/status.json)，status：COMPLETE_MEAN
- [experiments/us/20261126_mean_fusion_v2/S_s42/status.json](../../us/20261126_mean_fusion_v2/S_s42/status.json)，status：COMPLETE_MEAN

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_mean_fusion_v2/paired/REPORT.md](../../us/20261126_mean_fusion_v2/paired/REPORT.md)，第1行：# 同历史逐站时间编码：双种子配对结果
- [experiments/us/20261126_mean_fusion_v2/paired/REPORT.md](../../us/20261126_mean_fusion_v2/paired/REPORT.md)，第3行：2016–2019连续训练；2020抽样开发日历、137注册查询站。每模型同72h有效历史、同站点抽样、同3000块。raw与EMA分别报告。C/H/S如列出则均从相同3000步源继续3000步。
- [experiments/us/20261126_mean_fusion_v2/paired/REPORT.md](../../us/20261126_mean_fusion_v2/paired/REPORT.md)，第37行：区间共享同月份抽样；它衡量已暴露开发日历的比较，模型选择未经多重比较校正。Transformer概率候选沿用seed42预先冻结的架构/last政策，seed17结果用于复核。

## 原始协议、结果与报告

- [experiments/us/20261126_mean_fusion_v2/C_ema_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/C_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_ema_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/C_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_last_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/C_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_last_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/C_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_s17/progress.json](../../us/20261126_mean_fusion_v2/C_s17/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/C_s17/protocol.json](../../us/20261126_mean_fusion_v2/C_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/C_s17/status.json](../../us/20261126_mean_fusion_v2/C_s17/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/C_s17/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/C_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_s17/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/C_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_s42/progress.json](../../us/20261126_mean_fusion_v2/C_s42/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/C_s42/protocol.json](../../us/20261126_mean_fusion_v2/C_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/C_s42/status.json](../../us/20261126_mean_fusion_v2/C_s42/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/C_s42/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/C_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/C_s42/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/C_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_ema_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/H_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_ema_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/H_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_last_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/H_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_last_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/H_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_s17/progress.json](../../us/20261126_mean_fusion_v2/H_s17/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/H_s17/protocol.json](../../us/20261126_mean_fusion_v2/H_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/H_s17/status.json](../../us/20261126_mean_fusion_v2/H_s17/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/H_s17/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/H_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_s17/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/H_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_s42/progress.json](../../us/20261126_mean_fusion_v2/H_s42/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/H_s42/protocol.json](../../us/20261126_mean_fusion_v2/H_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/H_s42/status.json](../../us/20261126_mean_fusion_v2/H_s42/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/H_s42/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/H_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/H_s42/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/H_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_ema_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/S_ema_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_ema_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/S_ema_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_last_dev_s17/manifest.json](../../us/20261126_mean_fusion_v2/S_last_dev_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_last_dev_s42/manifest.json](../../us/20261126_mean_fusion_v2/S_last_dev_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_s17/progress.json](../../us/20261126_mean_fusion_v2/S_s17/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/S_s17/protocol.json](../../us/20261126_mean_fusion_v2/S_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/S_s17/status.json](../../us/20261126_mean_fusion_v2/S_s17/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/S_s17/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/S_s17/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_s17/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/S_s17/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_s42/progress.json](../../us/20261126_mean_fusion_v2/S_s42/progress.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/S_s42/protocol.json](../../us/20261126_mean_fusion_v2/S_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_mean_fusion_v2/S_s42/status.json](../../us/20261126_mean_fusion_v2/S_s42/status.json)（status；read）
- [experiments/us/20261126_mean_fusion_v2/S_s42/eval_ema/metrics.json](../../us/20261126_mean_fusion_v2/S_s42/eval_ema/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/S_s42/eval_last/metrics.json](../../us/20261126_mean_fusion_v2/S_s42/eval_last/metrics.json)（metadata；read）
- [experiments/us/20261126_mean_fusion_v2/paired/REPORT.md](../../us/20261126_mean_fusion_v2/paired/REPORT.md)（report；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
