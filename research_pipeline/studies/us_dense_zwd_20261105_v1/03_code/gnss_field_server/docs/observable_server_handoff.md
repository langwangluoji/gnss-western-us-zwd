# 单A800扩天气覆盖试验交接

## 本轮结论与作业范围

本机800步EDM与均值Gaussian对照已完成，全年2019及独立位移归因表明EDM的ES收益主要来自均值位置变化。服务器默认只运行短基准、均值头扩覆盖训练及全年开发评价。EDM最多保留同数据同预算的一次覆盖反证，单独登记原因；不排入长程、GAN、Optuna或自动多种子。

训练池为6288个合格B2018小时，排除了12个完整开发7日块及两侧18小时缓冲；固定288个B开发锚点。每次8小时、最多两遍，更新上限1572。保留现有M0、IDW混合权重、g(A)、PCA、V、协方差与tau。底座曾使用过开发B时刻，结果称冻结底座上的增量覆盖验证。

默认硬件为1张A800、2 CPU、16 GB内存。基准5分钟上限，训练和全年评价分别30分钟上限。本机新入口峰值显存约114 MB；训练使用已缓存背景和站端矩阵，该耗时不包含重新训练M0或全域格网采样。A800真实基准决定正式预算，RTX记录无法解锁。

## 上传与静态检查

上传`20261120_observable64_v02.zip`至既有项目`uploads`，然后在服务器执行。所有新文件落在既有总项目内的独立release和run目录，不安装包或修改Conda环境。

```bash
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
python_bin="$HOME/.conda/envs/ZhouSiyu/bin/python"
"$python_bin" -m zipfile -e "$project_root/uploads/20261120_observable64_v02.zip" "$project_root/code"
export OBS_RELEASE="$project_root/code/20261120_observable_subspace_v02"
export OBS_RUN="$project_root/runs/20261120_us_dense_zwd_observable64_coverage_v02"
mkdir -p "$OBS_RUN/logs"
PYTHONPATH="$OBS_RELEASE/src" PYTHONNOUSERSITE=1 "$python_bin" -m gnss_field.training.observable --help
sbatch --test-only "$OBS_RELEASE/slurm/observable_benchmark_a800.sh"
sbatch --test-only "$OBS_RELEASE/slurm/observable_mean_a800.sh"
sbatch --test-only "$OBS_RELEASE/slurm/observable_evaluate_a800.sh"
```

## 真实基准与预算锁

以下第一条会申请一张A800。基准含20步测速及连续200对100+恢复100的420次工程更新，A800另加约10步真实Slurm信号暂停/恢复，不产生科学候选。工程与正式检查点使用不同stage签名，不能把工程终点冒充零步。

```bash
bench_id=$(sbatch --parsable --output="$OBS_RUN/logs/benchmark_%j.out" "$OBS_RELEASE/slurm/observable_benchmark_a800.sh")
squeue -j "$bench_id"
```

确认基准作业完成且`benchmark.json`通过后生成预算锁。锁绑定实际A800、EDM保守测速上界、两模型与入口源码、B数据、seed42和run目录；默认每臂训练最多0.5卡时，科学步数仍最多1572。基准及全年评价另记账，不能把墙钟余量改成额外更新。

```bash
PYTHONPATH="$OBS_RELEASE/src" PYTHONNOUSERSITE=1 "$python_bin" "$OBS_RELEASE/tools/lock_observable_budget.py" \
  --benchmark "$OBS_RUN/benchmark/benchmark.json" --bundle "$OBS_RELEASE/bundle" \
  --output "$OBS_RUN/budget.lock.json" --per-arm-cap 0.5
```

## 默认顺序队列

```bash
mean_id=$(sbatch --parsable --dependency="afterok:$bench_id" \
  --output="$OBS_RUN/logs/mean_%j.out" "$OBS_RELEASE/slurm/observable_mean_a800.sh")
eval_id=$(sbatch --parsable --dependency="afterok:$mean_id" \
  --output="$OBS_RUN/logs/evaluate_mean_%j.out" "$OBS_RELEASE/slurm/observable_evaluate_a800.sh")
squeue -j "$mean_id,$eval_id"
```

训练入口只接受`bundle`中的B2018；评价入口只接受独立`evaluation`中的C2019，拒绝D2020。评价要求B选定的`selection.lock.json`且核验权重与基础产物哈希。每月保存站点集合预测，全年另存指标。均值Gaussian用解析均值；EDM如运行，128成员均值明确单列。

追加EDM覆盖反证前登记`OBS_EDM_COVERAGE_REASON`，沿同一个锁定训练池、种子、优化器和1572上限运行。不要直接把它接到默认队列。若两臂差异仍可由均值解释，不延长EDM。

## 日志、暂停与恢复

```bash
tail -f "$OBS_RUN/mean_seed42/training.jsonl"
tail -f "$OBS_RUN/mean_seed42/development.jsonl"
scancel --signal=USR1 "$mean_id"
```

Python在更新边界保存模型、EMA、优化器、RNG、游标后以75退出；`afterok`不会放行。评价在月边界保存，恢复重用已完成月份。重新提交同一stage和目录会恢复，工程目录不能用于正式训练。

同一run目录有原子GPU任务锁，防止重复作业同时写入。硬杀进程可能留下锁，需先核实锁内Slurm作业已退出，再处理该锁文件。没有自动requeue。跨项目作业由账户自行调度，本锁只约束此试验。

本机已完成同一portable入口和数据的更新、恢复及单位核验；Linux Slurm、A800驱动、真实信号及分区权限仍由服务器短预检确认。本交接包尚未提交任何作业。

## v02交付前补强

真实基准、预算锁、训练检查点及评价共用完整release.json哈希。启动时逐一核验包内源码、共享引擎、目标函数、随机流、配置及数据；任一依赖变化必须重新基准。年度成员分片以原子写入提交，避免中断留下半写文件。原v01包保留追溯。

第0步已保存但尚无last.pt的中断会恢复initial.pt，保留零步候选。A800预算锁还必须核对real_slurm_signal_resume=true，Windows工程结果不能替代该检查。

## 自动顺序交接

上传核验后可将CPU预算门控作业设为afterok依赖真实benchmark。门控核验基准、信号和完整发布哈希后锁定预算，再依次登记mean和C评价；mean依赖门控成功，C评价依赖mean正常完成。仅此一次默认队列，不包含EDM。作业ID保存在run根目录benchmark_job_id、gate_job_id、mean_job_id、evaluation_job_id。

默认流程的Slurm墙钟上限分列为工程5分钟、正式训练30分钟、全年评价30分钟，合计65分钟/1.0833卡时上限；.5卡时是训练阶段额度。CPU门控单列，不占GPU。实际收费按服务器账单记录，不按此上限当作已消耗。

评价现在每8小时批次检查暂停；未提交的当月可以重算，完整月片绑定evaluation.lock哈希及自身SHA，恢复不重复压缩已验证月份。预算耗尽时，pending评价和零步评价也不能越过暂停门槛。
