# 本轮运行及交接

服务器项目根保持为 `~/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`。

- 独立发布：`code/20261122_generative_full_v02`。
- 实验运行：`runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v01`。
- 解释性模型与逐层参数：`protocol/ARCHITECTURE.md`。
- 配置：发布目录 `configs/full_generative_a800_v01.json`。
- 权重与逐阶段日志：运行目录 `stages/<stage>/`，`progress.json`、`events.jsonl`、`last.pt`、正式候选文件。
- 作业链：CPU数据审计 → 单A800预检（最多1卡时） → CPU核验 → 单A800全量阶段串行 → CPU图表与置信区间导出。
- 单卡总额24卡时。实测完整预算超限会写 `preflight_status.json` 的 `STOP_BUDGET`，不会自动缩减epoch或扩大额度。超时、错误、科学门槛失败不盲目重投。
- 开发指标和点值/联合概率候选各自保存；2020 D标签没有随新发布包上传。

查看状态（以下命令均在已激活ZhouSiyu环境、发布目录下执行）：

```bash
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
release="$project_root/code/20261122_generative_full_v02"
run="$project_root/runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v01"
export PYTHONNOUSERSITE=1 PYTHONPATH="$release/src:$release/tools"
python "$release/tools/run_full_generative_campaign.py" --config "$release/configs/full_generative_a800_v01.json" status
squeue -u "$USER"
```

受控暂停：先创建暂停标记，阻止CPU派发器启动下一张卡，再向当前正式GPU作业发送USR1。将下方job_id替换为状态中当前GPU编号。不要向无关CPU或其他实验作业发送信号。

```bash
touch "$run/pause.request"
scancel --batch --signal=USR1 job_id
```

恢复：确认当前GPU作业已退出，移除本轮单个暂停标记，再通过同一配置恢复。训练器核对模型、优化器、EMA、RNG、游标、候选与依赖哈希。预检预算不足或程序错误应先诊断，不能把它们当普通暂停恢复。

```bash
rm -- "$run/pause.request"
python "$release/tools/run_full_generative_campaign.py" --config "$release/configs/full_generative_a800_v01.json" submit
```

CPU结果任务导出 `evaluation/figures/` 与 `evaluation/calendar_cases/`，入口 `tools/export_full_generative_results.py`；完成记录 `results_export.json`。场景固定为2019年1/4/7/10月15日12时，不按模型误差选择。场图16成员用于解释，不替代全年K128独立站评分。

本机长连接持续保存 `server_status.json` 和 `bridge/live.log`，退出SSH不会终止Slurm训练。触发只读刷新可创建本机 `bridge/request_status`；下载本轮结果可写 `bridge/request_download.json`，内容为 `{"relative":"evaluation.json"}`。凭据只由OpenSSH控制台处理，不写入文件。
