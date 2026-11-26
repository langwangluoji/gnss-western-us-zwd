# 20261126_native_weather_probability

当前公开来源：`experiments/us/20261126_native_weather_probability`；[来源目录](../../us/20261126_native_weather_probability)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_native_weather_probability`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)，第1行：# 时空域与频域扩散：同C64条件双种子3000步

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/status.json)，status：COMPLETE_PROBABILITY
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/status.json)，status：COMPLETE_PROBABILITY

## 报告中的阶段声明

- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)，第3行：七组训练均完成双种子各3000连续块；Gaussian另有同权重copula推断，预条件两组各有DDIM/Heun采样器。2020主日历已暴露开发，非全年、非严格OOF。

## 历史证据范围与信息条件

- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)，第3行：七组训练均完成双种子各3000连续块；Gaussian另有同权重copula推断，预条件两组各有DDIM/Heun采样器。2020主日历已暴露开发，非全年、非严格OOF。
- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)，第43行：全部模型最后权重已与checkpoint3000逐tensor核对；预测、truth、冻结均值及计分来源已验收。旧协议缺少显式条件维度时，另由权重输入维度核验C64。部分旧计分源码版本不同，原始源码哈希保留，并从实际成员独立复算。频域original/commuted后端据实际来源记录，不混写。频域与joint参数约38.4k/38.5k，预条件组额外冻结5070参数Gaussian。Gaussian本体5070参数，并非容量匹配对照。
- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)，第44行：主要判断应比较同信息、相近容量的频域与joint，同时检查Gaussian copula参考。新S175/G265使用更丰富的直连天气信息，其差异不能归因为频域。内部增量收益也不能从独立Gaussian单独推断为非Gaussian优势。

## 原始协议、结果与报告

- [experiments/us/20261126_native_weather_probability/paired_spectral/REPORT.md](../../us/20261126_native_weather_probability/paired_spectral/REPORT.md)（report；read）
- [experiments/us/20261126_native_weather_probability/paired_spectral/summary.json](../../us/20261126_native_weather_probability/paired_spectral/summary.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/frequency_tokens/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/eval_copula/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/eval_copula/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/gaussian/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/graph/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/joint/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/eval_heun/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_frequency_tokens/eval_heun/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/eval_heun/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/preconditioned_joint/eval_heun/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s17/spatial/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/frequency_tokens/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/eval_copula/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/eval_copula/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/gaussian/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/graph/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/joint/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/eval_heun/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_frequency_tokens/eval_heun/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/eval_ddim/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/eval_heun/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/preconditioned_joint/eval_heun/metrics.json)（metadata；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/protocol.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/protocol.json)（protocol；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/status.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/status.json)（status；read）
- [experiments/us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/eval_ddim/metrics.json](../../us/20261126_native_weather_probability/transformer72_physics_aux_last_s42/spatial/eval_ddim/metrics.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
