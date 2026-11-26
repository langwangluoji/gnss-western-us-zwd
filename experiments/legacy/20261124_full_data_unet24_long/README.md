# 20261124_full_data_unet24_long

当前公开来源：`experiments/us/20261124_full_data_unet24_long`；[来源目录](../../us/20261124_full_data_unet24_long)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_full_data_unet24_long`。

用途类别：`uncertain`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261124_full_data_unet24_long/REPORT.md](../../us/20261124_full_data_unet24_long/REPORT.md)，第1行：# 20261124 全训练日 A+B + 宽 U-Net 长训复核

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261124_full_data_unet24_long/REPORT.md](../../us/20261124_full_data_unet24_long/REPORT.md)，第7行：- 输入：ZWD、温度、气压、现有地形与站点几何；未读取 PWV/IWV/TCWV 或 C/D 标签。
- [experiments/us/20261124_full_data_unet24_long/REPORT.md](../../us/20261124_full_data_unet24_long/REPORT.md)，第22行：这次是真正的扩充数据长训，纠正了此前 72 序列/60 秒筛查的口径问题。结果不支持直接把全量日历数据接入现有 U-Net；下一步保留扩充数据包，加入显式残差目标和时序/空间边消息，并重新检查训练/验证分布、归一化和事件采样权重。所有后续比较继续使用同一 formal 合同。

## 原始协议、结果与报告

- [experiments/us/20261124_full_data_unet24_long/REPORT.md](../../us/20261124_full_data_unet24_long/REPORT.md)（report；read）
- [experiments/us/20261124_full_data_unet24_long/protocol.json](../../us/20261124_full_data_unet24_long/protocol.json)（protocol；read）
- [experiments/us/20261124_full_data_unet24_long/summary.json](../../us/20261124_full_data_unet24_long/summary.json)（metadata；read）

## 待核查项

- 无明确状态证据；不由目录名称或指标文件推断完成。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
