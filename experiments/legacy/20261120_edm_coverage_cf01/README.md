# 20261120_edm_coverage_cf01

当前公开来源：`experiments/us/20261120_edm_coverage_cf01`；[来源目录](../../us/20261120_edm_coverage_cf01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261120_edm_coverage_cf01`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第1行：# 单A800 EDM覆盖对照：完成结果

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第3行：本次正式EDM配对训练及全年C2019评价均正常完成。训练作业3610617在09:27:28结束，评价作业3610618在09:29:50结束，两者均COMPLETED、ExitCode 0:0。额外GPU占卡时间合计269秒，约0.07472卡时；这是Slurm占卡时长，实际费用以平台账单为准。

## 历史证据范围与信息条件

- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第3行：本次正式EDM配对训练及全年C2019评价均正常完成。训练作业3610617在09:27:28结束，评价作业3610618在09:29:50结束，两者均COMPLETED、ExitCode 0:0。额外GPU占卡时间合计269秒，约0.07472卡时；这是Slurm占卡时长，实际费用以平台账单为准。
- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第23行：EDM当前站宏RMSE为9.977926 mm，来自完整观测分布的有限128成员均值；mean的9.939550 mm是解析均值。两种点产品的采样误差不同，下一步需对EDM使用独立潜在噪声池积分后再比较期望均值精度。固定Gaussian余项及观测噪声可解析去均值，不必将这些零均值项重复采样引入点误差。
- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第29行：1. 取回两臂入选权重与所需小型评分统计，在本机完成独立均值积分；不得用C观测残差拟合均值位移。
- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第30行：2. 对Gaussian、Gaussian＋EDM独立位移、EDM、去位移EDM及已训练mean作匹配评价，区分位置与分布形状。
- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)，第33行：本次任务已经结束，没有自动增加训练时长、种子、Optuna或GAN作业。D2020继续封存；25 km仍是输出间距。本轮为冻结底座上的增量覆盖开发证据，不是锁定后的回顾评价。

## 原始协议、结果与报告

- [experiments/us/20261120_edm_coverage_cf01/RESULTS.md](../../us/20261120_edm_coverage_cf01/RESULTS.md)（report；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
