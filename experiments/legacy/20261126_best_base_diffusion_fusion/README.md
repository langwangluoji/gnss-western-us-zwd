# 20261126_best_base_diffusion_fusion

当前公开来源：`experiments/us/20261126_best_base_diffusion_fusion`；[来源目录](../../us/20261126_best_base_diffusion_fusion)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_best_base_diffusion_fusion`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_best_base_diffusion_fusion/s17/affine/protocol.json](../../us/20261126_best_base_diffusion_fusion/s17/affine/protocol.json)，objective：marginal seven-lead v score; GP-coupled query-local deterministic transport, not full joint score

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_best_base_diffusion_fusion/s17/G265/status.json](../../us/20261126_best_base_diffusion_fusion/s17/G265/status.json)，status：COMPLETE_WEATHER_GP
- [experiments/us/20261126_best_base_diffusion_fusion/s17/S175/status.json](../../us/20261126_best_base_diffusion_fusion/s17/S175/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_best_base_diffusion_fusion/s17/affine/status.json](../../us/20261126_best_base_diffusion_fusion/s17/affine/status.json)，status：COMPLETE_WEATHER_TRANSPORT
- [experiments/us/20261126_best_base_diffusion_fusion/s17/local/status.json](../../us/20261126_best_base_diffusion_fusion/s17/local/status.json)，status：COMPLETE_WEATHER_TRANSPORT
- [experiments/us/20261126_best_base_diffusion_fusion/s42/G265/status.json](../../us/20261126_best_base_diffusion_fusion/s42/G265/status.json)，status：COMPLETE_WEATHER_GP
- [experiments/us/20261126_best_base_diffusion_fusion/s42/S175/status.json](../../us/20261126_best_base_diffusion_fusion/s42/S175/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_best_base_diffusion_fusion/s42/affine/status.json](../../us/20261126_best_base_diffusion_fusion/s42/affine/status.json)，status：COMPLETE_WEATHER_TRANSPORT
- [experiments/us/20261126_best_base_diffusion_fusion/s42/local/status.json](../../us/20261126_best_base_diffusion_fusion/s42/local/status.json)，status：COMPLETE_WEATHER_TRANSPORT
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_G265/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_G265/status.json)，status：ENGINEERING_SMOKE_COMPLETE
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_S175/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_S175/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_affine/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_affine/status.json)，status：ENGINEERING_SMOKE_COMPLETE
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_local/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_local/status.json)，status：ENGINEERING_SMOKE_COMPLETE

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_best_base_diffusion_fusion/paired/REPORT.md](../../us/20261126_best_base_diffusion_fusion/paired/REPORT.md)，第3行：2016–2019连续训练，两种子每臂3000块；2020主/补日历均为开发。点输出逐值等于原joint v2；raw K32与独立不重叠双场差分K16分别比较。概率Gaussian残差中心与确定性点输出分开。
- [experiments/us/20261126_best_base_diffusion_fusion/paired/REPORT.md](../../us/20261126_best_base_diffusion_fusion/paired/REPORT.md)，第46行：完整事件、条件站距离/海岸/高程/地形/季节、有限成员覆盖和来源见summary.json。季节区间出现空月份抽样时标记NA。strict_oof=false；旧候选池未来可用性与历史质控限制保留，确认及原C/D未读。

## 原始协议、结果与报告

- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/G265_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/G265_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/G265_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/G265_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/affine_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/affine_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/affine_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/affine_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/local_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/local_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/local_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/local_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_G265_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_G265_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_G265_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_G265_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_affine_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_affine_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_affine_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_affine_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_local_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_local_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar15/paired_local_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar15/paired_local_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/G265_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/G265_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/G265_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/G265_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/affine_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/affine_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/affine_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/affine_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/local_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/local_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/local_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/local_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_G265_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_G265_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_G265_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_G265_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_affine_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_affine_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_affine_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_affine_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_local_s17/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_local_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/calendar8/paired_local_s42/metrics.json](../../us/20261126_best_base_diffusion_fusion/calendar8/paired_local_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/embedding_train_s17/manifest.json](../../us/20261126_best_base_diffusion_fusion/embedding_train_s17/manifest.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/embedding_train_s42/manifest.json](../../us/20261126_best_base_diffusion_fusion/embedding_train_s42/manifest.json)（metadata；read）
- [experiments/us/20261126_best_base_diffusion_fusion/paired/REPORT.md](../../us/20261126_best_base_diffusion_fusion/paired/REPORT.md)（report；read）
- [experiments/us/20261126_best_base_diffusion_fusion/paired/summary.json](../../us/20261126_best_base_diffusion_fusion/paired/summary.json)（metadata；linked_only_size_limit）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/G265/progress.json](../../us/20261126_best_base_diffusion_fusion/s17/G265/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/G265/protocol.json](../../us/20261126_best_base_diffusion_fusion/s17/G265/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/G265/status.json](../../us/20261126_best_base_diffusion_fusion/s17/G265/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/S175/protocol.json](../../us/20261126_best_base_diffusion_fusion/s17/S175/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/S175/status.json](../../us/20261126_best_base_diffusion_fusion/s17/S175/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/affine/progress.json](../../us/20261126_best_base_diffusion_fusion/s17/affine/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/affine/protocol.json](../../us/20261126_best_base_diffusion_fusion/s17/affine/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/affine/status.json](../../us/20261126_best_base_diffusion_fusion/s17/affine/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/local/progress.json](../../us/20261126_best_base_diffusion_fusion/s17/local/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/local/protocol.json](../../us/20261126_best_base_diffusion_fusion/s17/local/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s17/local/status.json](../../us/20261126_best_base_diffusion_fusion/s17/local/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/G265/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/G265/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/G265/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/G265/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/G265/status.json](../../us/20261126_best_base_diffusion_fusion/s42/G265/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/S175/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/S175/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/S175/status.json](../../us/20261126_best_base_diffusion_fusion/s42/S175/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/affine/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/affine/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/affine/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/affine/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/affine/status.json](../../us/20261126_best_base_diffusion_fusion/s42/affine/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/local/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/local/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/local/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/local/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/local/status.json](../../us/20261126_best_base_diffusion_fusion/s42/local/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_G265/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_G265/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_G265/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_G265/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_G265/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_G265/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_S175/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_S175/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_S175/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_S175/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_affine/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_affine/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_affine/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_affine/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_affine/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_affine/status.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_local/progress.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_local/progress.json)（status；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_local/protocol.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_local/protocol.json)（protocol；read）
- [experiments/us/20261126_best_base_diffusion_fusion/s42/smoke_local/status.json](../../us/20261126_best_base_diffusion_fusion/s42/smoke_local/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
