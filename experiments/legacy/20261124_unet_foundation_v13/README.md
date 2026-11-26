# 20261124_unet_foundation_v13

当前公开来源：`experiments/us/20261124_unet_foundation_v13`；[来源目录](../../us/20261124_unet_foundation_v13)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_unet_foundation_v13`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261124_unet_foundation_v13/README.md](../../us/20261124_unet_foundation_v13/README.md)，第1行：# v13 同输入 U-Net 基础场实验

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261124_unet_foundation_v13/status.json](../../us/20261124_unet_foundation_v13/status.json)，state：completed
- [experiments/us/20261124_unet_foundation_v13/verification.json](../../us/20261124_unet_foundation_v13/verification.json)，status：passed
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/status.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/status.json)，state：completed
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/status.json](../../us/20261124_unet_foundation_v13/E1305_unet48/status.json)，state：completed
- [experiments/us/20261124_unet_foundation_v13/hybrid_smoke/status.json](../../us/20261124_unet_foundation_v13/hybrid_smoke/status.json)，state：completed
- [experiments/us/20261124_unet_foundation_v13/large_smoke/status.json](../../us/20261124_unet_foundation_v13/large_smoke/status.json)，state：completed

## 报告中的阶段声明

- [experiments/us/20261124_unet_foundation_v13/README.md](../../us/20261124_unet_foundation_v13/README.md)，第3行：五组架构、十次拟合及最终诊断已完成，包括用户追加的大宽度 U-Net48/96/192。
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第3行：2026-11-24，五种架构、每种两个种子全部完成。当前最值得继续验证的是点特征 MLP 与小 U-Net 的融合：平均站宏 RMSE 为 **10.828 mm**，比同输入 MLP 降低 0.347 mm（约 3.1%）。用户追加的大宽度 U-Net 为 **11.800 mm**，本轮加宽没有带来进一步收益。

## 历史证据范围与信息条件

- [experiments/us/20261124_unet_foundation_v13/README.md](../../us/20261124_unet_foundation_v13/README.md)，第9行：沿用当前允许变量及 410/137/138 站角色；2016/2017 年现有 72 片段展开 2,880 个拟合事件。2018 年每月 1 日、三个布局评价，确认及原 C/D 标签未读取。每次拟合预算 600 秒，实际累计约 100 分钟。历史约 9 mm 结果使用不同协议。
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第5行：## 同口径结果
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第17行：同评价人口的固定高程 IDW 为 12.066 mm，v12 相关高斯解析均值为 11.625 mm。历史约 9 mm U-Net 使用另一套输入、数据和划分，不能据此判断本轮退步。当前未达到 8 mm。
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第23行：沿用当前允许变量数据合同：410 训练站、137 验证站、138 确认站；确认站及原 C/D 标签未读取。训练来自 2016/2017 年既有 72 个 36 小时片段，每片段展开 24 个有效因果预测时刻及两个条件布局，共 3,456 事件。按行号模 6 留出时间监控后，2,880 事件参与拟合，包含 1,440 个不同训练时刻、554,083 个不同有效站时标签及 923,640 个布局—站时监督条目。本轮增加的是既有日期的利用率，没有新增天气日期。
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第27行：各模型使用同一套 113 维特征：41 维静态、时间、气象与支撑信息，加上 24 个历史变量的末值、均值和首末差。历史窗口为因果 12 小时，空间尺度为 50/150/300 km；背景为高差输运后的 150 km 高斯核 ZWD。没有 PWV、IWV、TCWV、Tm 输入或教师。历史站队列曾经过 PWV 完整率筛选，这一队列来源限制沿用现有合同。
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)，第59行：0–50 km 的 720 个站对差值 RMSE 从 7.758 降至 7.500 mm；50–150 km 的 519 个站对从 14.162 降至 13.121 mm，两种子方向一致。这支持已观测站对上的空间差值改善，尚无稠密真值验证整张场的细节。小时增量 RMSE 从约 5.839 降至 5.762 mm；本轮确定性历史条件模型尚未检验随机生成序列的联合时空一致性。
- [experiments/us/20261124_unet_foundation_v13/REVIEW.md](../../us/20261124_unet_foundation_v13/REVIEW.md)，第1行：# 独立只读审查记录
- [experiments/us/20261124_unet_foundation_v13/REVIEW.md](../../us/20261124_unet_foundation_v13/REVIEW.md)，第11行：采样修正用三轮2880事件的固定种子回放核验：原循环的每轮唯一事件数为2880/2879/2879，修正后每轮均2880。E1304独立循环已采用正确次序，E1305继承修正后的主方法；未修改先前权重。

## 原始协议、结果与报告

- [experiments/us/20261124_unet_foundation_v13/README.md](../../us/20261124_unet_foundation_v13/README.md)（documentation；read）
- [experiments/us/20261124_unet_foundation_v13/RESULTS.md](../../us/20261124_unet_foundation_v13/RESULTS.md)（report；read）
- [experiments/us/20261124_unet_foundation_v13/REVIEW.md](../../us/20261124_unet_foundation_v13/REVIEW.md)（report；read）
- [experiments/us/20261124_unet_foundation_v13/protocol.json](../../us/20261124_unet_foundation_v13/protocol.json)（protocol；read）
- [experiments/us/20261124_unet_foundation_v13/status.json](../../us/20261124_unet_foundation_v13/status.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/summary.json](../../us/20261124_unet_foundation_v13/summary.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/verification.json](../../us/20261124_unet_foundation_v13/verification.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1300G_grid_background/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1300G_grid_background/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1300P_point_background/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1300P_point_background/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s17/progress.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s17/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s17/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s17/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s17/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s42/progress.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s42/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s42/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1301_mlp/s42/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1301_mlp/s42/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s17/progress.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s17/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s17/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s17/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s17/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s42/progress.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s42/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s42/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1302_unet16/s42/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1302_unet16/s42/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s17/progress.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s17/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s17/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s17/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s17/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s42/progress.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s42/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s42/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1303_unet24/s42/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1303_unet24/s42/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/protocol.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/protocol.json)（protocol；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/status.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/status.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s17/progress.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s17/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s17/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s17/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s17/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s42/progress.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s42/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s42/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1304_point_unet/s42/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1304_point_unet/s42/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/protocol.json](../../us/20261124_unet_foundation_v13/E1305_unet48/protocol.json)（protocol；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/status.json](../../us/20261124_unet_foundation_v13/E1305_unet48/status.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s17/progress.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s17/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s17/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s17/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s17/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s17/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s42/progress.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s42/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s42/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/E1305_unet48/s42/last_diagnostic/expanded/metrics.json](../../us/20261124_unet_foundation_v13/E1305_unet48/s42/last_diagnostic/expanded/metrics.json)（metadata；read）
- [experiments/us/20261124_unet_foundation_v13/hybrid_smoke/protocol.json](../../us/20261124_unet_foundation_v13/hybrid_smoke/protocol.json)（protocol；read）
- [experiments/us/20261124_unet_foundation_v13/hybrid_smoke/status.json](../../us/20261124_unet_foundation_v13/hybrid_smoke/status.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/hybrid_smoke/s42/progress.json](../../us/20261124_unet_foundation_v13/hybrid_smoke/s42/progress.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/large_smoke/protocol.json](../../us/20261124_unet_foundation_v13/large_smoke/protocol.json)（protocol；read）
- [experiments/us/20261124_unet_foundation_v13/large_smoke/status.json](../../us/20261124_unet_foundation_v13/large_smoke/status.json)（status；read）
- [experiments/us/20261124_unet_foundation_v13/large_smoke/s42/progress.json](../../us/20261124_unet_foundation_v13/large_smoke/s42/progress.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
