# 单张A800配对训练交接

实验：20261121_us_dense_zwd_grid69_processor_pair_v01

## 运行位置

服务器项目总目录：
`/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`

- 代码：`code/20261121_grid_processor_pair_v01`
- 运行：`runs/20261121_us_dense_zwd_grid69_processor_pair_v01`
- Slurm作业：3633416，1张A800，节点gpu8006。
- 环境：原ZhouSiyu环境，未安装依赖或改动其他环境。
- 本机监控：http://127.0.0.1:8781/；可见窗口GNSS Grid69 Live Training Log。
- 快照归档：本目录server_live.log；断开SSH不影响服务器训练。

## 队列与产物

fold_0/mlp69 → fold_0/resunet69 → fold_1/mlp69 → fold_1/resunet69。

每组最多2736次更新、有效batch8、最多4遍；共10944次科学更新上限。5472训练小时、96内层锚点、2016外层完整周小时。M0、IDW混合和Gaussian均冻结，只训练处理69通道条件的均值修正网络。

本轮仅B2018增量开发验证；既有底座有B历史暴露。C2019与D2020不读取。EDM/GAN不自动启动。

每组保留：training.jsonl、development.jsonl、initial_metrics.json、status.json、checkpoint、selection.lock.json、outer_predictions.npz、results.json。

第200步只作启动检查。按内层ES选择EMA并保留零修正候选；锁定后外层K64评价。RMSE、MAE、bias与ES同步记录。训练损失为站端ES，不是RMSE，初始/启动数值不能作为最终结论。

## 只读检查

```bash
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
export GRID_RELEASE="$project_root/code/20261121_grid_processor_pair_v01"
export GRID_RUN="$project_root/runs/20261121_us_dense_zwd_grid69_processor_pair_v01"
export GRID_PYTHON="$HOME/.conda/envs/ZhouSiyu/bin/python"
squeue -j 3633416
cat "$GRID_RUN/queue_status.json"
tail -n 5 "$GRID_RUN/fold_0/mlp69/training.jsonl"
tail -n 5 "$GRID_RUN/fold_0/mlp69/development.jsonl"
sacct -j 3633416 -o JobID,State,ExitCode,Elapsed,AllocTRES
```

## 暂停与恢复

暂停：

```bash
touch "$GRID_RUN/PAUSE"
```

更新边界保存恢复状态；评价在小时边界暂停。等待作业终止，再恢复：

```bash
PYTHONPATH="$GRID_RELEASE/src" "$GRID_PYTHON" \
  "$GRID_RELEASE/tools/grid_pair_submit.py" \
  --root "$GRID_RUN" --release "$GRID_RELEASE" --python "$GRID_PYTHON" --resume
```

恢复沿用实验ID、数据游标、模型/EMA/优化器及随机状态，自动清除PAUSE。controller使用sacct累计扣除既往分配时间，总额14400秒；记账未知时拒绝追加预算，不重复提交活动任务。达到预算上限保存暂停，不自动增加卡时。

## 验证记录

- 286项回归测试通过，证据pytest_final_verified.log。
- 最终release本机400次工程更新通过，100＋100恢复与连续200逐位一致，真实权重变化；scientific/engineering_resume/verification.json。
- 服务器release哈希一致，实测预算和工程恢复门槛已放行。
- 科学第200步启动检查与实时指标见server_live.log及本机监控，以实际追加记录为准。

## 启动验收完成（2026-11-21 14:48）

作业3633416在gpu8006持续RUNNING，正式fold_0/mlp69已完成200步。training.jsonl由100步增长至200步，累计真实监督376538站时刻；当步训练ES 5.4652533531 mm。第200步启动检查：RMSE 7.3987451451 mm、MAE 5.6151130003 mm、bias -0.1437865614 mm、ES 5.3744170070 mm，共244站、22421站时刻。该检查不参与选模，仅为B2018内层工程启动验收。

同一冻结输入与固定评价时刻下，初始RMSE 7.4028736265 mm，200步EMA的RMSE已改变，结合实际优化更新记录确认预测权重已变化；当前正式最优候选仍保留step0，首次正式选模评价在342步。

本机页面已观察到第200步启动RMSE及“未参与选模”标记；SSH快照、日志镜像、HTTP接口持续更新。完整证据startup_acceptance.json和latest_server_snapshot.json。

本次人工启动监督到此结束。保留Slurm单卡顺序队列、SSH快照与本机只读监控；不等待全队列完成，不扩展预算，不自动启动EDM/GAN。
