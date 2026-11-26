# 20261126_leaveout_innovation_panel

当前公开来源：`experiments/us/20261126_leaveout_innovation_panel`；[来源目录](../../us/20261126_leaveout_innovation_panel)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_leaveout_innovation_panel`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261126_leaveout_innovation_panel/paired/REPORT.md](../../us/20261126_leaveout_innovation_panel/paired/REPORT.md)，第1行：# 留站创新面板：尺度与可预测信号分开检验

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261126_leaveout_innovation_panel/s17/status.json](../../us/20261126_leaveout_innovation_panel/s17/status.json)，status：COMPLETE_INFORMATION_PANEL
- [experiments/us/20261126_leaveout_innovation_panel/s42/status.json](../../us/20261126_leaveout_innovation_panel/s42/status.json)，status：COMPLETE_INFORMATION_PANEL

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261126_leaveout_innovation_panel/paired/REPORT.md](../../us/20261126_leaveout_innovation_panel/paired/REPORT.md)，第3行：两种子各48个固定月块，每块连续12h；2016–2018拟合线性探针，2019检验探针迁移。来源p14/TF已拟合2016–2019，因此不是基础模型独立时间留出。原查询由缓存既有规则确定；本轮没有新神经训练或2020评价。
- [experiments/us/20261126_leaveout_innovation_panel/paired/REPORT.md](../../us/20261126_leaveout_innovation_panel/paired/REPORT.md)，第30行：留出在全部背景历史GNSS/T/P上生效，单块有真实扰动不变性核验；前向留站不等于训练折外，拼接LOSO不自动形成统一Gaussian误差过程。旧未来可用性选目标与历史PWV筛选限制保留，确认及原C/D标签未读。

## 原始协议、结果与报告

- [experiments/us/20261126_leaveout_innovation_panel/paired/REPORT.md](../../us/20261126_leaveout_innovation_panel/paired/REPORT.md)（report；read）
- [experiments/us/20261126_leaveout_innovation_panel/paired/summary.json](../../us/20261126_leaveout_innovation_panel/paired/summary.json)（metadata；read）
- [experiments/us/20261126_leaveout_innovation_panel/s17/progress.json](../../us/20261126_leaveout_innovation_panel/s17/progress.json)（status；read）
- [experiments/us/20261126_leaveout_innovation_panel/s17/protocol.json](../../us/20261126_leaveout_innovation_panel/s17/protocol.json)（protocol；read）
- [experiments/us/20261126_leaveout_innovation_panel/s17/status.json](../../us/20261126_leaveout_innovation_panel/s17/status.json)（status；read）
- [experiments/us/20261126_leaveout_innovation_panel/s42/progress.json](../../us/20261126_leaveout_innovation_panel/s42/progress.json)（status；read）
- [experiments/us/20261126_leaveout_innovation_panel/s42/protocol.json](../../us/20261126_leaveout_innovation_panel/s42/protocol.json)（protocol；read）
- [experiments/us/20261126_leaveout_innovation_panel/s42/status.json](../../us/20261126_leaveout_innovation_panel/s42/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
