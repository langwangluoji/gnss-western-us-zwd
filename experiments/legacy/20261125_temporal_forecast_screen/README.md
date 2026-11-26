# 20261125_temporal_forecast_screen

当前公开来源：`experiments/us/20261125_temporal_forecast_screen`；[来源目录](../../us/20261125_temporal_forecast_screen)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_temporal_forecast_screen`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)，第1行：# 连续时空预报：本机配对筛查

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261125_temporal_forecast_screen/protocol.json](../../us/20261125_temporal_forecast_screen/protocol.json)，status：LOCAL_FORECAST_SCREEN_NOT_ANNUAL
- [experiments/us/20261125_temporal_forecast_screen/status.json](../../us/20261125_temporal_forecast_screen/status.json)，status：COMPLETE_LOCAL_FORECAST_SCREEN

## 报告中的阶段声明

- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)，第3行：状态：真实本机训练与2020抽样开发评价；非全年正式实验。

## 历史证据范围与信息条件

- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)，第5行：输入为条件站截至锚点的ZWD/T/P、地形与查询点海岸距离；隐藏目标站的所有历史不进入模型。
- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)，第30行：同人口周块配对bootstrap区间见 `paired_block_intervals.json`；1000次重采样按天气周分块，种子指标取均值，未将种子当作独立天气样本。
- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)，第32行：开发数据已暴露，后续调参须标记开发回看。168小时、多分辨率、完整年度及diffusion仍按计划独立推进。

## 原始协议、结果与报告

- [experiments/us/20261125_temporal_forecast_screen/REPORT.md](../../us/20261125_temporal_forecast_screen/REPORT.md)（report；read）
- [experiments/us/20261125_temporal_forecast_screen/protocol.json](../../us/20261125_temporal_forecast_screen/protocol.json)（protocol；read）
- [experiments/us/20261125_temporal_forecast_screen/status.json](../../us/20261125_temporal_forecast_screen/status.json)（status；read）
- [experiments/us/20261125_temporal_forecast_screen/summary.json](../../us/20261125_temporal_forecast_screen/summary.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/verification.json](../../us/20261125_temporal_forecast_screen/verification.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h12_s17/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h12_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h12_s42/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h12_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h48_s17/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h48_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h48_s42/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h48_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h72_s17/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h72_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/gru_h72_s42/metrics.json](../../us/20261125_temporal_forecast_screen/gru_h72_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h12_s17/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h12_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h12_s42/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h12_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h48_s17/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h48_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h48_s42/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h48_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h72_s17/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h72_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/lstm_h72_s42/metrics.json](../../us/20261125_temporal_forecast_screen/lstm_h72_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h12_s17/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h12_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h12_s42/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h12_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h48_s17/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h48_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h48_s42/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h48_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h72_s17/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h72_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/mlp_h72_s42/metrics.json](../../us/20261125_temporal_forecast_screen/mlp_h72_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/none_h12_s17/metrics.json](../../us/20261125_temporal_forecast_screen/none_h12_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/none_h12_s42/metrics.json](../../us/20261125_temporal_forecast_screen/none_h12_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h12_s17/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h12_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h12_s42/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h12_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h48_s17/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h48_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h48_s42/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h48_s42/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h72_s17/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h72_s17/metrics.json)（metadata；read）
- [experiments/us/20261125_temporal_forecast_screen/transformer_h72_s42/metrics.json](../../us/20261125_temporal_forecast_screen/transformer_h72_s42/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
