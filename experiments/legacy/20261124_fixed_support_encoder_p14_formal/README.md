# 20261124_fixed_support_encoder_p14_formal

当前公开来源：`experiments/us/20261124_fixed_support_encoder_p14_formal`；[来源目录](../../us/20261124_fixed_support_encoder_p14_formal)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_fixed_support_encoder_p14_formal`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)，第1行：# p14 固定支撑基础场正式实验与扩散架构对照

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261124_fixed_support_encoder_p14_formal/protocol_formal.json](../../us/20261124_fixed_support_encoder_p14_formal/protocol_formal.json)，status：COMPLETE_BASE_FOUNDATION_2_SEEDS

## 报告中的阶段声明

- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)，第23行：p14 基础场是确定性模型，CRPS在本轮等同绝对误差，区间覆盖率为 0。项目已完成的同区域残差扩散对照中，相关高斯宏 RMSE 为 11.880 mm，残差扩散为 12.055 mm；v11 直接扩散扩展宏 RMSE 为 12.339 mm。这些是历史扩散实验，不是 p14 基础场上重新训练的扩散。

## 历史证据范围与信息条件

- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)，第7行：固定支撑编码器按两种子、每种 600 秒预算完成训练。每个种子约 2.8 万次更新，使用 410 个训练站、137 个验证站、12 个月×24 小时×3 布局扩展评价。确认站及 C/D 标签未读取。
- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)，第23行：p14 基础场是确定性模型，CRPS在本轮等同绝对误差，区间覆盖率为 0。项目已完成的同区域残差扩散对照中，相关高斯宏 RMSE 为 11.880 mm，残差扩散为 12.055 mm；v11 直接扩散扩展宏 RMSE 为 12.339 mm。这些是历史扩散实验，不是 p14 基础场上重新训练的扩散。
- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)，第32行：带星号的扩散结果来自已核验历史实验，不代表 p14＋扩散新配对结果。

## 原始协议、结果与报告

- [experiments/us/20261124_fixed_support_encoder_p14_formal/REPORT.md](../../us/20261124_fixed_support_encoder_p14_formal/REPORT.md)（report；read）
- [experiments/us/20261124_fixed_support_encoder_p14_formal/protocol.json](../../us/20261124_fixed_support_encoder_p14_formal/protocol.json)（protocol；read）
- [experiments/us/20261124_fixed_support_encoder_p14_formal/protocol_formal.json](../../us/20261124_fixed_support_encoder_p14_formal/protocol_formal.json)（protocol；read）
- [experiments/us/20261124_fixed_support_encoder_p14_formal/summary.json](../../us/20261124_fixed_support_encoder_p14_formal/summary.json)（metadata；read）
- [experiments/us/20261124_fixed_support_encoder_p14_formal/s17/expanded/metrics.json](../../us/20261124_fixed_support_encoder_p14_formal/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_fixed_support_encoder_p14_formal/s42/expanded/metrics.json](../../us/20261124_fixed_support_encoder_p14_formal/s42/expanded/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
