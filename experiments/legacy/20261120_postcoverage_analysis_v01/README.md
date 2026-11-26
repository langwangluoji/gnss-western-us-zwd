# 20261120_postcoverage_analysis_v01

当前公开来源：`experiments/us/20261120_postcoverage_analysis_v01`；[来源目录](../../us/20261120_postcoverage_analysis_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261120_postcoverage_analysis_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)，第1行：# 覆盖试验后的本机复核

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json)，stage：observable_edm_refine
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json)，status：COMPLETE
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json)，stage：observable_mean_refine
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)，第23行：B开发集的mean EMA/raw ES为5.432114/5.405626 mm，EDM为5.442473/5.416517 mm。该差距证明现有EMA与实时权重表现不同，不能单独证明更长训练会改善空间泛化。下一轮应锁定EMA产品、记录raw诊断，先进行有上限的收敛验证；停止和推进依据使用B，C在本轮权重锁定后评价。

## 历史证据范围与信息条件

- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)，第5行：C2019全年8760小时、76个可评分站、642258个站时刻。概率评价为6个站组；C用于开发，D2020继续封存。
- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)，第15行：EDM原有全噪声128成员均值的RMSE为9.977926 mm；独立积分只估计潜变量变换的均值位移，以解析方式去掉固定高斯余项及观测噪声。独立种子20261121，128成员，64前缀RMSE为9.941590 mm；位移估计的标准误RMS为0.009084 mm。两种均值估计需分开报告。
- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)，第19行：高斯基础场加EDM独立均值位移后ES为6.855129 mm；扣除位移的EDM为6.924617 mm。EDM相对mean的ES差为+0.003572 mm，7日时间块与站组交叉重采样的95%区间为[-0.002183,0.010268] mm。形状归因区间同样跨零，14日敏感性结论一致。现阶段可观察收益主要表现为均值校正，尚未检出额外分布形状收益。

## 原始协议、结果与报告

- [experiments/us/20261120_postcoverage_analysis_v01/RESULTS.md](../../us/20261120_postcoverage_analysis_v01/RESULTS.md)（report；read）
- [experiments/us/20261120_postcoverage_analysis_v01/analysis.lock.json](../../us/20261120_postcoverage_analysis_v01/analysis.lock.json)（protocol；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/edm/C2019_evaluation.lock.json](../../us/20261120_postcoverage_analysis_v01/artifacts/edm/C2019_evaluation.lock.json)（protocol；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/edm/selection.lock.json](../../us/20261120_postcoverage_analysis_v01/artifacts/edm/selection.lock.json)（protocol；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/edm/status.json)（status；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/mean/C2019_evaluation.lock.json](../../us/20261120_postcoverage_analysis_v01/artifacts/mean/C2019_evaluation.lock.json)（protocol；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/mean/selection.lock.json](../../us/20261120_postcoverage_analysis_v01/artifacts/mean/selection.lock.json)（protocol；read）
- [experiments/us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json](../../us/20261120_postcoverage_analysis_v01/artifacts/mean/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
