# 本机三模型顺序训练交接

当前有效生成批次v03，编码器对照产物保留在同名v01目录。原始训练候选864小时中，2018-04-29T00没有合格拟合B站；统一剔除该不满足训练资格的小时，三个模型均使用863小时。评价人口未变化。v01失败的VAE尝试不作为本轮结果，不继承其权重；既往编码器和失败尝试228.56秒已计入本轮60分钟总墙钟。

执行顺序为VAE（ELBO600＋ES1200步）、GAN1600步、全格网Diffusion1000步。扩散6步Heun、11次调用完整反传，训练/评价相同；本轮是观测评分训练，不宣称标准稠密场去噪。各模型独立完整训练，均值初始化来自本轮加宽编码器seed42检查点。

运行信息：`runtime/process.json`；当前状态：`status.json`；日志：`runtime/training_stdout.log`与`training_stderr.log`。全部结果自动保存到各模型子目录，点值和联合产品分别为`point_best.pt`和`joint_best.pt`，恢复状态为`last.pt`。正式配置：`generators.lock.json`，实际资格小时：`training_qualification.json`。

在工程目录打开PowerShell后，可用以下命令查看、暂停与恢复：

```powershell
Get-Content results/20261123_enhanced_encoder_generative_v03/status.json
New-Item -ItemType File -Force results/20261123_enhanced_encoder_generative_v03/PAUSE
# 确认status为PAUSED、原进程已退出后，删除本实验暂停标记并恢复：
Remove-Item -LiteralPath results/20261123_enhanced_encoder_generative_v03/PAUSE
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' tools/train_independent_generators.py --stage all
```

跨进程文件锁防止重复启动。预算耗尽单独记录，不自动扩预算或无限重启。评价中断后保留待评价检查点及已完成的人口产物，不重复提交活动训练。当前不提交服务器作业，不改本机Python环境。

三模型均完成后，后处理入口`tools/report_independent_generators.py`生成结果表、训练曲线、校准与区间宽度图。任务结束后不会自动启动新训练。

CRPS目前是站时刻微平均；覆盖率和宽度为站宏平均。VAE集合是自由先验解码场分布，未加固定3 mm似然噪声，不能称为ELBO完整观测后验预测。本轮C2019为已暴露48小时开发子集，D2020封存。


v03修复空间48小时的选取：从未过滤的864小时原始日历固定选择。v02已训练的VAE1800步、GAN1600步全部参数、优化器、随机状态和游标逐项验证一致后迁移。原空间评价重新计算，其他人口经核验后保留。累计预算原样继承，不重复训练。见migration.json。

