# mean收敛试验：服务器交接

服务器总目录：`$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`。

代码：`code/20261120_observable64_mean_convergence_v01`。
运行：`runs/20261120_us_dense_zwd_mean_convergence_v01`。
真实基准作业3617451，CPU预算验收3617452。后续mean与B/C评价的作业号分别写入运行目录的`mean_job_id`、`evaluation_job_id`。实际训练状态以`mean_seed42/status.json`为准。

本轮只训练mean头。5000步余弦日程，seed42，batch8，ES K4，EMA=.999；每100步B评价K16并同时记录raw；1600步以后patience10。EMA唯一选模，零候选保留。独立K64、种子20261122的B整周门槛通过后，才运行一次全年C2019评价。D2020封存。EDM、GAN、搜索与追加种子均不自动运行。

## 查看进度

```bash
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
run_root="$project_root/runs/20261120_us_dense_zwd_mean_convergence_v01"
squeue -u "$USER" -o '%.18i %.12P %.26j %.10T %.10M %R'
cat "$run_root/mean_seed42/status.json"
tail -n 5 "$run_root/mean_seed42/development.jsonl"
cat "$run_root/mean_seed42/B_gate.json"
cat "$run_root/mean_full2019/metrics.json"
```

文件尚未生成表示对应阶段尚未完成，不据此认定失败。基准及预算错误查看`logs/benchmark_3617451.out`、`logs/budget_3617452.out`；训练查看`logs/mean_作业号.out`。

## 暂停与恢复

训练期间在该运行目录创建`PAUSE`文件，会在一次完整更新后保存checkpoint，并以75退出，后续afterok作业不放行。也可向已确认的训练srun步骤发送USR1。不要对其他项目作业发送信号。

恢复前确认原训练作业已经退出，保留原checkpoint和预算锁，移除本轮PAUSE标志，再以同一发布、运行目录和解释器重新提交`slurm/convergence_mean.sh`。恢复消费剩余预算，不重置累计时间；评价依赖要绑定新的训练作业号。不要直接重跑bootstrap提交脚本，不修改合同或以旧失败作业号恢复依赖。

CPU任务`wrf_seg_0000`属于原有任务，本轮没有更改它。

## 交付状态

文件和导入路径已现场核实；作业已提交。已确认真实权重更新与有限开发指标。mean在4700步早停，选中3700步EMA，B门槛通过，全年C2019评价已完成。资源硬上限：基准5分钟、训练15分钟、B门槛与条件C评价15分钟，GPU并发1。

## 本次修复与验收结果

原CPU预算作业3617452因将冷启动平均时间外推而退出，未开始科学训练。修正作为独立资源附录部署，绑定真实A800基准及200步恢复checkpoint哈希，保持训练发布代码和900秒上限不变。CPU修正作业3618413完成；mean作业3618417完成（100秒Slurm墙钟），B/C评价作业3618418已完成（124秒Slurm墙钟）。

mean在4700步触发早停，唯一选中3700步EMA。B独立K64复核：新EMA ES5.400692 mm，旧EMA5.432911 mm，差-0.032218 mm，12开发周配对区间[-0.065840,-0.005165] mm。raw诊断差的区间跨零，现阶段更支持EMA追赶；该开发区间受选模影响。全年C精度另列，D2020仍封存。

当前认证SSH保持打开，供后续读写项目命令使用，不保存密码；关闭连接不影响Slurm。训练曲线及已验SHA的小型服务器日志见本目录server_artifacts与training_curve.png。
