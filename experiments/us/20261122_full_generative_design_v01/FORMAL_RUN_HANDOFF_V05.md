# 正式全量队列交接（2026-11-23）

项目根：`~/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`。

当前发布：`code/20261122_generative_full_v05`。
当前配置：发布目录 `configs/full_generative_a800_v02.json`。
当前实验：`runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03`。

服务器实查：A800作业3699476已RUNNING，节点gpu8011，目前全量条件缓存准备；依赖派发器3699477正在等待。尚无正式模型更新指标。之前v02的STOP_BUDGET是历史状态，不代表当前v03。

环境、来源和预算验收已通过。实测预算23.4856卡时，包含此前633秒基准计费，总上限24卡时。全部科学epoch保留；固定576小时开发子集锁定10个产品后，各做一次全年2019评价，不以全年结果再选择checkpoint。2020 D站保持锁定。

自动顺序：缓存准备→B站内部诊断→确定性对照10 epoch→VAE表示30 epoch→后验缓存→GAN与Diffusion各20 epoch预训练→三类共享表示ES各3 epoch→三类完整模型ES各3 epoch→固定产品全年评价→CPU结果导出。科学门槛、异常或预算不通过会停，不无限重投。最多同时占用1张A800。

SSH和本机关闭不影响Slurm。预计总计算约一天，排队时间另计，早晨未必全部完成。

在服务器查询：

```bash
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
release="$project_root/code/20261122_generative_full_v05"
run="$project_root/runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v03"
export PYTHONNOUSERSITE=1 PYTHONPATH="$release/src:$release/tools"
"$HOME/.conda/envs/ZhouSiyu/bin/python" "$release/tools/run_full_generative_campaign.py" --config "$release/configs/full_generative_a800_v02.json" status
squeue -u "$USER"
```

日志在当前run根目录`<stage>-<jobid>.out/.err`；阶段权重及指标在`stages/<stage>/`。最终逐站预测与指标在`evaluation/`；图表由CPU导出阶段写出。只检查本实验作业，不操作其他任务。

本机当前实查快照：`runtime_v03/formal_status_verified.json`。旧长连接的默认`server_status.json`仍指向历史v02，应使用本快照或通过`actions/current_formal_status.py`刷新，不能误读旧快照。认证信息未写入本交接文件。
