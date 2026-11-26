# 20261121_grid_processor_pair_v01

当前公开来源：`experiments/us/20261121_grid_processor_pair_v01`；[来源目录](../../us/20261121_grid_processor_pair_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261121_grid_processor_pair_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第1行：# 单张A800配对训练交接

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json)，stage：grid_processor_refine
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json)，status：COMPLETE
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json)，stage：grid_processor_refine
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json)，status：COMPLETE
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json)，stage：grid_processor_refine
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json)，status：COMPLETE
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json)，stage：grid_processor_refine
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第43行：## 暂停与恢复
- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第45行：暂停：
- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第51行：更新边界保存恢复状态；评价在小时边界暂停。等待作业终止，再恢复：
- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第59行：恢复沿用实验ID、数据游标、模型/EMA/优化器及随机状态，自动清除PAUSE。controller使用sacct累计扣除既往分配时间，总额14400秒；记账未知时拒绝追加预算，不重复提交活动任务。达到预算上限保存暂停，不自动增加卡时。
- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第70行：作业3633416在gpu8006持续RUNNING，正式fold_0/mlp69已完成200步。training.jsonl由100步增长至200步，累计真实监督376538站时刻；当步训练ES 5.4652533531 mm。第200步启动检查：RMSE 7.3987451451 mm、MAE 5.6151130003 mm、bias -0.1437865614 mm、ES 5.3744170070 mm，共244站、22421站时刻。该检查不参与选模，仅为B2018内层工程启动验收。

## 历史证据范围与信息条件

- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)，第23行：本轮仅B2018增量开发验证；既有底座有B历史暴露。C2019与D2020不读取。EDM/GAN不自动启动。

## 原始协议、结果与报告

- [experiments/us/20261121_grid_processor_pair_v01/HANDOFF.md](../../us/20261121_grid_processor_pair_v01/HANDOFF.md)（documentation；read）
- [experiments/us/20261121_grid_processor_pair_v01/data.lock.json](../../us/20261121_grid_processor_pair_v01/data.lock.json)（protocol；read）
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/verification.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/verification.json)（metadata；read）
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/continuous/status.json)（status；read）
- [experiments/us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/engineering_resume/split/status.json)（status；read）
- [experiments/us/20261121_grid_processor_pair_v01/scientific/data.lock.json](../../us/20261121_grid_processor_pair_v01/scientific/data.lock.json)（protocol；read）
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/verification.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/verification.json)（metadata；read）
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/continuous/status.json)（status；read）
- [experiments/us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json](../../us/20261121_grid_processor_pair_v01/scientific/engineering_resume/split/status.json)（status；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
