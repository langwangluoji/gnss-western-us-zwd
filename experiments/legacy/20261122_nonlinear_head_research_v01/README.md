# 20261122_nonlinear_head_research_v01

当前公开来源：`experiments/us/20261122_nonlinear_head_research_v01`；[来源目录](../../us/20261122_nonlinear_head_research_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_nonlinear_head_research_v01`。

用途类别：`diagnostic`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_nonlinear_head_research_v01/run.lock.json](../../us/20261122_nonlinear_head_research_v01/run.lock.json)，purpose：local research spike

## 类别依据

- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第61行：## 5. GAN判别器诊断

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/PLAN.md](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/PLAN.md)，第1行：Fixed optimizer control after Pro advice: same69 inputs and32-width residual hidden features, same ridge candidates and masks; paired zero/ridge readout;14complete epochs588updates, constantLR1e-4, raw-only selection each42updates, EMAhalf-life42 diagnostic. No architecture/data changes relative to preceding spike. No server submission. Keep zero and exact ridge initial candidates. Local0.5% two-fold inner gate unchanged.

## 历史证据范围与信息条件

- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第5行：本轮完成两组本机研究spike，共8次神经小训练、4752次更新，并完成线性/非线性解析读出及归档GAN判别器诊断。生产代码、旧实验、环境和服务器作业均未修改。没有新增C2019评分，D2020封存。
- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第9行：已核验的C2019全年共同人口为76站、642258站时刻：粗场双线性13.3640 mm，A站IDW格网10.2785 mm，冻结M0为10.3446 mm。M0相对粗场的站宏RMSE下降约22.6%；M0与直接A站IDW的排名则不同。已有基础链条的收益应保留，生成式分支须在此之上证明额外价值。
- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第41行：每折仍使用336个独立训练小时、41个内层缓存时刻。拟合站为195/160，空间内层站为49/44；外层为原12个注册天气周内288个缓存锚点、204/244站。历史冻结底座/先验有B暴露，全部属于增量开发。
- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第85行：旧5472小时池中，在排除已有768缓存时刻、新内层天气周及距所有336训练锚点18小时以内时刻后，可用候选为0。不能把锚点附近的新小时称为独立天气验证。下一覆盖扩展必须先重划天气分工；它仍称增量开发。增加小时数和增加独立天气过程分别计数。
- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)，第87行：若后续通过门槛，才启用提前锁定的C2019开发48小时确认（每月7/21日00/12小时）；这不是最终盲测。只有新覆盖、配对区间与线性/常数对照共同支持，才登记单张A800全合格时间池训练。约6 mm仍为待验证目标。

## 原始协议、结果与报告

- [experiments/us/20261122_nonlinear_head_research_v01/RESULTS.md](../../us/20261122_nonlinear_head_research_v01/RESULTS.md)（report；read）
- [experiments/us/20261122_nonlinear_head_research_v01/confirmation_hours.lock.json](../../us/20261122_nonlinear_head_research_v01/confirmation_hours.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/results.json](../../us/20261122_nonlinear_head_research_v01/results.json)（metadata；read）
- [experiments/us/20261122_nonlinear_head_research_v01/run.lock.json](../../us/20261122_nonlinear_head_research_v01/run.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/status.json](../../us/20261122_nonlinear_head_research_v01/status.json)（status；read）
- [experiments/us/20261122_nonlinear_head_research_v01/fold_0/selection.lock.json](../../us/20261122_nonlinear_head_research_v01/fold_0/selection.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/fold_1/selection.lock.json](../../us/20261122_nonlinear_head_research_v01/fold_1/selection.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/PLAN.md](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/PLAN.md)（documentation；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/results.json](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/results.json)（metadata；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/run.lock.json](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/run.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/status.json](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/status.json)（status；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/fold_0/selection.lock.json](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/fold_0/selection.lock.json)（protocol；read）
- [experiments/us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/fold_1/selection.lock.json](../../us/20261122_nonlinear_head_research_v01/pro_optimizer_control_v01/fold_1/selection.lock.json)（protocol；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
