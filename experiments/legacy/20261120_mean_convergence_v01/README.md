# 20261120_mean_convergence_v01

当前公开来源：`experiments/us/20261120_mean_convergence_v01`；[来源目录](../../us/20261120_mean_convergence_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261120_mean_convergence_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第1行：# mean收敛试验：服务器交接

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261120_mean_convergence_v01/local_engineering/status.json](../../us/20261120_mean_convergence_v01/local_engineering/status.json)，stage：observable_mean_benchmark
- [experiments/us/20261120_mean_convergence_v01/local_engineering/status.json](../../us/20261120_mean_convergence_v01/local_engineering/status.json)，status：PAUSED
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json)，stage：observable_mean_benchmark
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json)，status：PAUSED
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json)，stage：observable_mean_benchmark
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json)，status：PAUSED
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/status.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/status.json)，status：COMPLETE
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json)，stage：observable_mean_refine
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第23行：文件尚未生成表示对应阶段尚未完成，不据此认定失败。基准及预算错误查看`logs/benchmark_3617451.out`、`logs/budget_3617452.out`；训练查看`logs/mean_作业号.out`。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第25行：## 暂停与恢复
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第29行：恢复前确认原训练作业已经退出，保留原checkpoint和预算锁，移除本轮PAUSE标志，再以同一发布、运行目录和解释器重新提交`slurm/convergence_mean.sh`。恢复消费剩余预算，不重置累计时间；评价依赖要绑定新的训练作业号。不要直接重跑bootstrap提交脚本，不修改合同或以旧失败作业号恢复依赖。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第35行：文件和导入路径已现场核实；作业已提交。已确认真实权重更新与有限开发指标。mean在4700步早停，选中3700步EMA，B门槛通过，全年C2019评价已完成。资源硬上限：基准5分钟、训练15分钟、B门槛与条件C评价15分钟，GPU并发1。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第39行：原CPU预算作业3617452因将冷启动平均时间外推而退出，未开始科学训练。修正作为独立资源附录部署，绑定真实A800基准及200步恢复checkpoint哈希，保持训练发布代码和900秒上限不变。CPU修正作业3618413完成；mean作业3618417完成（100秒Slurm墙钟），B/C评价作业3618418已完成（124秒Slurm墙钟）。
- [experiments/us/20261120_mean_convergence_v01/RESULTS.md](../../us/20261120_mean_convergence_v01/RESULTS.md)，第3行：本轮训练与全年2019开发评价均完成，全部新作业退出0。mean在4700步触发预登记早停，入选第3700步EMA。没有追加训练分支。
- [experiments/us/20261120_mean_convergence_v01/RESULTS.md](../../us/20261120_mean_convergence_v01/RESULTS.md)，第18行：训练曲线见training_curve.png；入选权重SHA256为`d1096574109d89bdcc2b318f6d2d3f764095279fa9f99a07412633bbb6725a6c`，权重保留在服务器本轮运行目录。当前SSH连接保持打开；关闭本机连接不影响已完成产物。

## 历史证据范围与信息条件

- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/metrics.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/metrics.json)，scope：C2019 exposed development, not D2020 retrospective
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第9行：本轮只训练mean头。5000步余弦日程，seed42，batch8，ES K4，EMA=.999；每100步B评价K16并同时记录raw；1600步以后patience10。EMA唯一选模，零候选保留。独立K64、种子20261122的B整周门槛通过后，才运行一次全年C2019评价。D2020封存。EDM、GAN、搜索与追加种子均不自动运行。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第35行：文件和导入路径已现场核实；作业已提交。已确认真实权重更新与有限开发指标。mean在4700步早停，选中3700步EMA，B门槛通过，全年C2019评价已完成。资源硬上限：基准5分钟、训练15分钟、B门槛与条件C评价15分钟，GPU并发1。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第39行：原CPU预算作业3617452因将冷启动平均时间外推而退出，未开始科学训练。修正作为独立资源附录部署，绑定真实A800基准及200步恢复checkpoint哈希，保持训练发布代码和900秒上限不变。CPU修正作业3618413完成；mean作业3618417完成（100秒Slurm墙钟），B/C评价作业3618418已完成（124秒Slurm墙钟）。
- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)，第41行：mean在4700步触发早停，唯一选中3700步EMA。B独立K64复核：新EMA ES5.400692 mm，旧EMA5.432911 mm，差-0.032218 mm，12开发周配对区间[-0.065840,-0.005165] mm。raw诊断差的区间跨零，现阶段更支持EMA追赶；该开发区间受选模影响。全年C精度另列，D2020仍封存。
- [experiments/us/20261120_mean_convergence_v01/RESULTS.md](../../us/20261120_mean_convergence_v01/RESULTS.md)，第12行：同一76站、642258站时刻下，RMSE点估计改善0.031259 mm（0.314%）。组宏ES为6.803722 mm，站宏CRPS为5.594384 mm。95%区间覆盖率为90.975%，区间宽度34.451 mm。C是已暴露开发年；本轮C的配对区间尚未计算，D2020继续封存。
- [experiments/us/20261120_mean_convergence_v01/RESULTS.md](../../us/20261120_mean_convergence_v01/RESULTS.md)，第14行：B独立K64门槛通过，新EMA与旧EMA的ES差为-0.032218 mm，开发周配对区间[-0.065840,-0.005165] mm。新旧raw差区间跨零，曲线显示EMA追赶是本轮收益的重要解释，不能据此认定参数本身需要持续加训。开发区间受选模影响。

## 原始协议、结果与报告

- [experiments/us/20261120_mean_convergence_v01/HANDOFF.md](../../us/20261120_mean_convergence_v01/HANDOFF.md)（documentation；read）
- [experiments/us/20261120_mean_convergence_v01/RESULTS.md](../../us/20261120_mean_convergence_v01/RESULTS.md)（report；read）
- [experiments/us/20261120_mean_convergence_v01/local_engineering/status.json](../../us/20261120_mean_convergence_v01/local_engineering/status.json)（status；read）
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_continuous/status.json)（status；read）
- [experiments/us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json](../../us/20261120_mean_convergence_v01/local_engineering/resume_split/status.json)（status；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/budget.lock.json](../../us/20261120_mean_convergence_v01/server_artifacts/budget.lock.json)（protocol；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/evaluation.lock.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/evaluation.lock.json)（protocol；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/metrics.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/metrics.json)（metadata；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/status.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_full2019/status.json)（status；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/B_gate.lock.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/B_gate.lock.json)（protocol；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/selection.lock.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/selection.lock.json)（protocol；read）
- [experiments/us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json](../../us/20261120_mean_convergence_v01/server_artifacts/mean_seed42/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
