# 20261119_local_basis_diagnosis

当前公开来源：`experiments/us/20261119_local_basis_diagnosis`；[来源目录](../../us/20261119_local_basis_diagnosis)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261119_local_basis_diagnosis`。

用途类别：`diagnostic`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)，第1行：# 本机空间基诊断结果

## 类别依据

- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)，第1行：# 本机空间基诊断结果

## 状态依据

- [experiments/us/20261119_local_basis_diagnosis/summary.json](../../us/20261119_local_basis_diagnosis/summary.json)，status：COMPLETE
- [experiments/us/20261119_local_basis_diagnosis/confirmation/summary.json](../../us/20261119_local_basis_diagnosis/confirmation/summary.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)，第19行：单位均为mm。原始权重与EMA都出现相同方向的误差增加，EMA滞后不足以解释这次失败。原配置在拟合组的逐小时均方误差平均后开方约为3.5–3.6 mm，留出组对应指标约为11.4–13.6 mm。该统计使用全部合格留出位置，和表中的共同细尺度位置站宏RMSE分别定义。
- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)，第69行：时刻集中于各月中部，实际只有12个有观测的7日块。区间解释限定为同一物理站组、后选择的探索性诊断。现有数据尚不足以确认细尺度收益，方向未通过与统计证据未定应分别报告。原门槛失败记录保持可追溯。

## 历史证据范围与信息条件

- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)，第39行：补充时刻和探索时刻不重叠，但使用同一批物理站组和已拟合空间基。它检验同一2018实验内的时间稳定性。2019年C站的条件预测与2020年D站的最终回顾评价仍承担独立产品评价。

## 原始协议、结果与报告

- [experiments/us/20261119_local_basis_diagnosis/RESULTS.md](../../us/20261119_local_basis_diagnosis/RESULTS.md)（report；read）
- [experiments/us/20261119_local_basis_diagnosis/summary.json](../../us/20261119_local_basis_diagnosis/summary.json)（metadata；read）
- [experiments/us/20261119_local_basis_diagnosis/confirmation/candidate.lock.json](../../us/20261119_local_basis_diagnosis/confirmation/candidate.lock.json)（protocol；read）
- [experiments/us/20261119_local_basis_diagnosis/confirmation/summary.json](../../us/20261119_local_basis_diagnosis/confirmation/summary.json)（metadata；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
