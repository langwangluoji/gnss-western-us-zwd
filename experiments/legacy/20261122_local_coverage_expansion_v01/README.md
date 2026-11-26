# 20261122_local_coverage_expansion_v01

当前公开来源：`experiments/us/20261122_local_coverage_expansion_v01`；[来源目录](../../us/20261122_local_coverage_expansion_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_local_coverage_expansion_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_local_coverage_expansion_v01/PLAN.md](../../us/20261122_local_coverage_expansion_v01/PLAN.md)，第1行：# 本机覆盖扩展：线性对照与非线性均值头

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261122_local_coverage_expansion_v01/status.json](../../us/20261122_local_coverage_expansion_v01/status.json)，stage：COMPLETE
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json)，stage：coverage_refine
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json)，status：COMPLETE
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/split/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/split/status.json)，stage：coverage_refine
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/split/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/split/status.json)，status：COMPLETE
- [experiments/us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json)，stage：coverage_refine
- [experiments/us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json)，status：COMPLETE
- [experiments/us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json)，stage：coverage_refine
- [experiments/us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261122_local_coverage_expansion_v01/PLAN.md](../../us/20261122_local_coverage_expansion_v01/PLAN.md)，第13行：输入缓存逐小时校验、保存进度；完整新增M0场保留以便发布复核。原文件及环境不变。训练保存模型/EMA/优化器/RNG/小时游标，测试恢复和暂停门槛。
- [experiments/us/20261122_local_coverage_expansion_v01/PLAN.md](../../us/20261122_local_coverage_expansion_v01/PLAN.md)，第17行：交付：锁定清单、数据计数、两臂指标、训练曲线、完整格网查询复核、checkpoint和暂停/恢复命令。阶段：设计/数值测试→缓存准备→线性拟合→两折非线性训练→锁定产品后外层复核。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第3行：已完成预登记的两折训练：新增12个月度内层天气块，576训练小时、48内层评分小时、原288外层缓存锚点。空间站组和小网络保持不变；新内层与旧41小时人口不同，因此只在本轮共同人口内比较，不把数值变化当作跨轮精度提升。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第55行：本实验已完成。运行python tools/local_coverage_train.py只返回ALREADY_COMPLETE，不追加更新；python tools/local_coverage_report.py复核预测并重建报告。工程根目录使用原pytorch环境及PYTHONPATH=src;tools。

## 历史证据范围与信息条件

- [experiments/us/20261122_local_coverage_expansion_v01/PLAN.md](../../us/20261122_local_coverage_expansion_v01/PLAN.md)，第15行：晋级：两折空间×天气内层相对零改正均改善≥0.5%，原外层同向；同时检查相对线性头的收益、配对天气/站群区间及短距离站间差值。通过后才制定C2019的48小时开发确认，D2020封存；本地门槛不是收费作业自动提交授权。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第14行：非线性模型在两折空间×天气内层均优于线性，但外层没有相应优势：折0线性更低，折1两者接近。两折内层相对底座的改善约0.434%/0.236%，未通过各≥0.5%的方向门槛。保留模型和证据，本轮不触发服务器或C2019新评价。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第25行：- 所有目标仅用于拟合与固定人口评分；新时间清单先锁定。历史底座/先验及旧试验有B暴露，本轮称增量开发。D2020封存。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第32行：外层12天气簇各含缓存锚点，不是完整逐小时周评分；同时给出50 km站组重采样区间，不按单个小时独立抽样。
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)，第57行：实际训练源码保存在source_run/，与原锁一致。审阅后的断点恢复修复与完整依赖锁适用于新实验；不回填原运行锁。新实验使用新ID、在输入构建前锁定tools/package源码和packed manifest，移除PAUSE后可按相同入口恢复，依赖变化会拒绝继续。本轮独立审阅及后验工程修复说明见FINAL_REVIEW.md。

## 原始协议、结果与报告

- [experiments/us/20261122_local_coverage_expansion_v01/PLAN.md](../../us/20261122_local_coverage_expansion_v01/PLAN.md)（documentation；read）
- [experiments/us/20261122_local_coverage_expansion_v01/RESULTS.md](../../us/20261122_local_coverage_expansion_v01/RESULTS.md)（report；read）
- [experiments/us/20261122_local_coverage_expansion_v01/experiment.lock.json](../../us/20261122_local_coverage_expansion_v01/experiment.lock.json)（protocol；read）
- [experiments/us/20261122_local_coverage_expansion_v01/results.json](../../us/20261122_local_coverage_expansion_v01/results.json)（metadata；read）
- [experiments/us/20261122_local_coverage_expansion_v01/status.json](../../us/20261122_local_coverage_expansion_v01/status.json)（status；read）
- [experiments/us/20261122_local_coverage_expansion_v01/training.lock.json](../../us/20261122_local_coverage_expansion_v01/training.lock.json)（protocol；read）
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/verification.json](../../us/20261122_local_coverage_expansion_v01/engineering/verification.json)（metadata；read）
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/continuous/status.json)（status；read）
- [experiments/us/20261122_local_coverage_expansion_v01/engineering/split/status.json](../../us/20261122_local_coverage_expansion_v01/engineering/split/status.json)（status；read）
- [experiments/us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/selection.lock.json](../../us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_0/nonlinear/status.json)（status；read）
- [experiments/us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/selection.lock.json](../../us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json](../../us/20261122_local_coverage_expansion_v01/fold_1/nonlinear/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
