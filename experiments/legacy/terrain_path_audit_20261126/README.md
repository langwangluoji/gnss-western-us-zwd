# terrain_path_audit_20261126

当前公开来源：`research/terrain_path_audit_20261126`；[来源目录](../../../research/terrain_path_audit_20261126)。

历史路径：`maintenance/terrain_path_audit_20261126`。

用途类别：`uncertain`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [research/terrain_path_audit_20261126/README.md](../../../research/terrain_path_audit_20261126/README.md)，第1行：# 地形路径屏障：开发回看审计

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [research/terrain_path_audit_20261126/README.md](../../../research/terrain_path_audit_20261126/README.md)，第1行：# 地形路径屏障：开发回看审计
- [research/terrain_path_audit_20261126/REPORT.md](../../../research/terrain_path_audit_20261126/REPORT.md)，第3行：seed42，2020每月8–10日已暴露开发日历，137开发站、3布局。I1与冻结Transformer physics_aux均使用既有预测；本报告解释当前重建误差的空间关联，不新增ZWD精度或因果结论。
- [research/terrain_path_audit_20261126/REPORT.md](../../../research/terrain_path_audit_20261126/REPORT.md)，第11行：每条记录为布局×起点×查询站，3布局复用观测，不是独立小时。注册137开发站中，135站有当前有效标签；无当前标签站为P691, P695，缺测标签记录共27,774条。
- [research/terrain_path_audit_20261126/REPORT.md](../../../research/terrain_path_audit_20261126/REPORT.md)，第75行：本次证据强度偏弱，不建议现在为路径屏障新增独立3k训练臂。原始分层显示高屏障与较大误差共现；固定距离、月份和query高程后，TF差值区间跨零。加入现有局地高差、roughness、坡度和海岸信息的固定空间块诊断中，两个屏障量没有增加当前bias或RMSE的预测解释力。这个结果适用于本次25km DEM、最近可见站proxy与开发日历，不推广为所有地形机制无效。

## 原始协议、结果与报告

- [research/terrain_path_audit_20261126/README.md](../../../research/terrain_path_audit_20261126/README.md)（documentation；read）
- [research/terrain_path_audit_20261126/REPORT.md](../../../research/terrain_path_audit_20261126/REPORT.md)（report；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 无明确状态证据；不由目录名称或指标文件推断完成。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
