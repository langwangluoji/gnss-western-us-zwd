# 20261126_inducing_latent_flow

当前公开来源：`experiments/us/20261126_inducing_latent_flow`；[来源目录](../../us/20261126_inducing_latent_flow)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_inducing_latent_flow`。

用途类别：`diagnostic`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s17/protocol.json](../../us/20261126_inducing_latent_flow/affine_frequency_s17/protocol.json)，objective：joint proper endpoint scores: marginal+.25increment+.25pair+.1energy

## 类别依据

- [experiments/us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md](../../us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md)，第1行：# 有限成员覆盖率与物理支撑诊断
- [experiments/us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md](../../us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md)，第1行：# 有限成员覆盖率与物理支撑诊断

## 状态依据

- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s17/status.json](../../us/20261126_inducing_latent_flow/affine_frequency_s17/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s42/status.json](../../us/20261126_inducing_latent_flow/affine_frequency_s42/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/affine_time_s17/status.json](../../us/20261126_inducing_latent_flow/affine_time_s17/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/affine_time_s42/status.json](../../us/20261126_inducing_latent_flow/affine_time_s42/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s17/status.json](../../us/20261126_inducing_latent_flow/odd_frequency_s17/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s42/status.json](../../us/20261126_inducing_latent_flow/odd_frequency_s42/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/odd_time_s17/status.json](../../us/20261126_inducing_latent_flow/odd_time_s17/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/odd_time_s42/status.json](../../us/20261126_inducing_latent_flow/odd_time_s42/status.json)，status：COMPLETE_INDUCING_FLOW
- [experiments/us/20261126_inducing_latent_flow/smoke_s42/status.json](../../us/20261126_inducing_latent_flow/smoke_s42/status.json)，status：ENGINEERING_SMOKE_COMPLETE

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md](../../us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md)，第3行：K32 IID条件下，连续且交换的真值/成员总体，rank3–30参考覆盖率27/33=81.818%，rank4–29为25/33=75.758%。以0.7/0.3独立随机选择两区间，其期望覆盖率为80%；表内报告该随机化的期望，不对随机硬区间另抽一次噪声。原经验10%–90%覆盖率保留为描述性量。
- [experiments/us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md](../../us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md)，第3行：K32 IID条件下，连续且交换的真值/成员总体，rank3–30参考覆盖率27/33=81.818%，rank4–29为25/33=75.758%。以0.7/0.3独立随机选择两区间，其期望覆盖率为80%；表内报告该随机化的期望，不对随机硬区间另抽一次噪声。原经验10%–90%覆盖率保留为描述性量。
- [experiments/us/20261126_inducing_latent_flow/nonlinearity_probe/REPORT.md](../../us/20261126_inducing_latent_flow/nonlinearity_probe/REPORT.md)，第3行：每种子选固定四个训练历史的首起点，不读GNSS目标值进行拟合。用1024个Gaussian噪声拟合latent线性代理，再用独立1024噪声检验残差能量。场中分母包括完整剩余GP方差，分子只计算非线性代理残差经实际解码后的能量；它是受所选条件和Monte Carlo影响的诊断，不是精确高阶累积量或预测收益。
- [experiments/us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md](../../us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md)，第25行：空间锚点不截断完整GP；训练Q32为几何/固定随机子集，评价Q137固定全人口。原Q128未来可用性选站、历史PWV筛选来源、末91块复用等限制继续保留，确认及原C/D标签未读取。
- [experiments/us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md](../../us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md)，第27行：主指标为odd_frequency相对affine_frequency六小时fairCRPS；相对Gaussian及时间/频率、增量、站对、事件和地形分层分别归因。区间条件于两个种子及已暴露日期，未经多重探索校正。

## 原始协议、结果与报告

- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s17/progress.json](../../us/20261126_inducing_latent_flow/affine_frequency_s17/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s17/protocol.json](../../us/20261126_inducing_latent_flow/affine_frequency_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s17/status.json](../../us/20261126_inducing_latent_flow/affine_frequency_s17/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s42/progress.json](../../us/20261126_inducing_latent_flow/affine_frequency_s42/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s42/protocol.json](../../us/20261126_inducing_latent_flow/affine_frequency_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/affine_frequency_s42/status.json](../../us/20261126_inducing_latent_flow/affine_frequency_s42/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s17/progress.json](../../us/20261126_inducing_latent_flow/affine_time_s17/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s17/protocol.json](../../us/20261126_inducing_latent_flow/affine_time_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s17/status.json](../../us/20261126_inducing_latent_flow/affine_time_s17/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s42/progress.json](../../us/20261126_inducing_latent_flow/affine_time_s42/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s42/protocol.json](../../us/20261126_inducing_latent_flow/affine_time_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/affine_time_s42/status.json](../../us/20261126_inducing_latent_flow/affine_time_s42/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md](../../us/20261126_inducing_latent_flow/baseline_calibration/REPORT.md)（report；read）
- [experiments/us/20261126_inducing_latent_flow/baseline_calibration/summary.json](../../us/20261126_inducing_latent_flow/baseline_calibration/summary.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/affine_frequency_s17/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/affine_frequency_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/affine_frequency_s42/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/affine_frequency_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/affine_time_s17/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/affine_time_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/affine_time_s42/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/affine_time_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/gaussian_time_s17/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/gaussian_time_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/gaussian_time_s42/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/gaussian_time_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/odd_frequency_s17/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/odd_frequency_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/odd_frequency_s42/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/odd_frequency_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/odd_time_s17/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/odd_time_s17/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calendar8/odd_time_s42/metrics.json](../../us/20261126_inducing_latent_flow/calendar8/odd_time_s42/metrics.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md](../../us/20261126_inducing_latent_flow/calibration_calendar8/REPORT.md)（report；read）
- [experiments/us/20261126_inducing_latent_flow/calibration_calendar8/summary.json](../../us/20261126_inducing_latent_flow/calibration_calendar8/summary.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/nonlinearity_probe/REPORT.md](../../us/20261126_inducing_latent_flow/nonlinearity_probe/REPORT.md)（report；read）
- [experiments/us/20261126_inducing_latent_flow/nonlinearity_probe/summary.json](../../us/20261126_inducing_latent_flow/nonlinearity_probe/summary.json)（metadata；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s17/progress.json](../../us/20261126_inducing_latent_flow/odd_frequency_s17/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s17/protocol.json](../../us/20261126_inducing_latent_flow/odd_frequency_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s17/status.json](../../us/20261126_inducing_latent_flow/odd_frequency_s17/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s42/progress.json](../../us/20261126_inducing_latent_flow/odd_frequency_s42/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s42/protocol.json](../../us/20261126_inducing_latent_flow/odd_frequency_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/odd_frequency_s42/status.json](../../us/20261126_inducing_latent_flow/odd_frequency_s42/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s17/progress.json](../../us/20261126_inducing_latent_flow/odd_time_s17/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s17/protocol.json](../../us/20261126_inducing_latent_flow/odd_time_s17/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s17/status.json](../../us/20261126_inducing_latent_flow/odd_time_s17/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s42/progress.json](../../us/20261126_inducing_latent_flow/odd_time_s42/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s42/protocol.json](../../us/20261126_inducing_latent_flow/odd_time_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/odd_time_s42/status.json](../../us/20261126_inducing_latent_flow/odd_time_s42/status.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md](../../us/20261126_inducing_latent_flow/paired_calendar8_v2/REPORT.md)（report；read）
- [experiments/us/20261126_inducing_latent_flow/paired_calendar8_v2/summary.json](../../us/20261126_inducing_latent_flow/paired_calendar8_v2/summary.json)（metadata；linked_only_size_limit）
- [experiments/us/20261126_inducing_latent_flow/smoke_s42/progress.json](../../us/20261126_inducing_latent_flow/smoke_s42/progress.json)（status；read）
- [experiments/us/20261126_inducing_latent_flow/smoke_s42/protocol.json](../../us/20261126_inducing_latent_flow/smoke_s42/protocol.json)（protocol；read）
- [experiments/us/20261126_inducing_latent_flow/smoke_s42/status.json](../../us/20261126_inducing_latent_flow/smoke_s42/status.json)（status；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
