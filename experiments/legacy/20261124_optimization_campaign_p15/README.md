# 20261124_optimization_campaign_p15

当前公开来源：`experiments/us/20261124_optimization_campaign_p15`；[来源目录](../../us/20261124_optimization_campaign_p15)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_optimization_campaign_p15`。

用途类别：`uncertain`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261124_optimization_campaign_p15/REPORT.md](../../us/20261124_optimization_campaign_p15/REPORT.md)，第1行：# P15 空间支撑采样筛查\n\n实验臂：random、hole100、hole150、distance_stratified。每臂 seed42/17，各 120 秒；模型保持 p14 固定支撑编码器，确认/C/D 未读取。\n\n| sampling | seed42 | seed17 | mean |\n|---|---:|---:|---:|\n| distance_stratified | 7.856 | 7.709 | 7.782 |\n| hole100 | 6.523 | 6.529 | 6.526 |\n| hole150 | 7.209 | 7.101 | 7.155 |\n| random | 6.361 | 6.211 | 6.286 |\n\n结果是固定时间监控筛查，不是扩展开发成绩。random 平均 6.29 mm，hole100 6.53 mm，hole150 7.16 mm，distance_stratified 7.78 mm。当前证据支持先保留随机支撑作为控制，同时加入 hole100 作为空间外推增强臂；150 km 空洞和四档距离分层在当前 120 秒预算下学习较慢，不能直接判定无效。\n\nP2 显式边消息、P3 真正可学习高程尺度、P4 查询一致性审计和 P5 p14 折外残差扩散需要独立工程接口与完整扩展评价，未将本筛查结果冒充这些实验的完成结果。

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261124_optimization_campaign_p15/REPORT.md](../../us/20261124_optimization_campaign_p15/REPORT.md)，第1行：# P15 空间支撑采样筛查\n\n实验臂：random、hole100、hole150、distance_stratified。每臂 seed42/17，各 120 秒；模型保持 p14 固定支撑编码器，确认/C/D 未读取。\n\n| sampling | seed42 | seed17 | mean |\n|---|---:|---:|---:|\n| distance_stratified | 7.856 | 7.709 | 7.782 |\n| hole100 | 6.523 | 6.529 | 6.526 |\n| hole150 | 7.209 | 7.101 | 7.155 |\n| random | 6.361 | 6.211 | 6.286 |\n\n结果是固定时间监控筛查，不是扩展开发成绩。random 平均 6.29 mm，hole100 6.53 mm，hole150 7.16 mm，distance_stratified 7.78 mm。当前证据支持先保留随机支撑作为控制，同时加入 hole100 作为空间外推增强臂；150 km 空洞和四档距离分层在当前 120 秒预算下学习较慢，不能直接判定无效。\n\nP2 显式边消息、P3 真正可学习高程尺度、P4 查询一致性审计和 P5 p14 折外残差扩散需要独立工程接口与完整扩展评价，未将本筛查结果冒充这些实验的完成结果。
- [experiments/us/20261124_optimization_campaign_p15/REPORT.md](../../us/20261124_optimization_campaign_p15/REPORT.md)，第4行：原表中的 P15 数值是固定时间 monitor RMSE，仅用于采样臂筛查，不能与 p14 正式扩展宏 RMSE直接比较。随后从各臂 `best.pt` 继续训练 600 秒，并使用 p14 同一合同：410 个训练站、137 个验证站、12 个月×24 小时、3 个布局，确认站与 C/D 标签不读取。

## 原始协议、结果与报告

- [experiments/us/20261124_optimization_campaign_p15/REPORT.md](../../us/20261124_optimization_campaign_p15/REPORT.md)（report；read）
- [experiments/us/20261124_optimization_campaign_p15/distance_stratified/protocol.json](../../us/20261124_optimization_campaign_p15/distance_stratified/protocol.json)（protocol；read）
- [experiments/us/20261124_optimization_campaign_p15/distance_stratified/summary.json](../../us/20261124_optimization_campaign_p15/distance_stratified/summary.json)（metadata；read）
- [experiments/us/20261124_optimization_campaign_p15/hole100/protocol.json](../../us/20261124_optimization_campaign_p15/hole100/protocol.json)（protocol；read）
- [experiments/us/20261124_optimization_campaign_p15/hole100/summary.json](../../us/20261124_optimization_campaign_p15/hole100/summary.json)（metadata；read）
- [experiments/us/20261124_optimization_campaign_p15/hole150/protocol.json](../../us/20261124_optimization_campaign_p15/hole150/protocol.json)（protocol；read）
- [experiments/us/20261124_optimization_campaign_p15/hole150/summary.json](../../us/20261124_optimization_campaign_p15/hole150/summary.json)（metadata；read）
- [experiments/us/20261124_optimization_campaign_p15/random/protocol.json](../../us/20261124_optimization_campaign_p15/random/protocol.json)（protocol；read）
- [experiments/us/20261124_optimization_campaign_p15/random/summary.json](../../us/20261124_optimization_campaign_p15/random/summary.json)（metadata；read）

## 待核查项

- 无明确状态证据；不由目录名称或指标文件推断完成。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
