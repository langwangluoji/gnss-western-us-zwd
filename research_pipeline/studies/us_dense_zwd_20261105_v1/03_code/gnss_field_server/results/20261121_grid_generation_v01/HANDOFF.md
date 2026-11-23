# 随机残差、Diffusion、GAN实施交接

当前状态：已核验旧作业COMPLETED0:0、完整结果及下载SHA256，服务器新训练待部署。外层U-Net未晋级，共同均值明确锁定M0+IDW，旧U-Net权重只作审计。

## 研究与队列

设计：docs/superpowers/specs/2026-11-21-grid-generation-v01.md
实施计划：docs/superpowers/plans/2026-11-21-grid-generation-v01.md
锁定配置草案：configs/grid_generation_v01.json

两折共同冻结已核验M0+IDW基础场。新增独立随机ResU-Net、Diffusion采样参数化、GAN。共同69条件+相关先验场+白化状态，共71通道。Gaussian先验旁路使第0步三臂一致；tau仅在站端加入一次。Diffusion明确采用真实站ES监督的18步Heun采样，不声称独立HR真值去噪预训练。

原5472训练小时完整覆盖一遍，每组684更新，batch8，运行时8小时并行；两折三臂上限4104更新。内层96小时K16，外层预登记96小时K32；C/D封存。单A800，累计分配上限4卡时，基准不容纳时停止，不自动追加预算。全部运行保持独立目录，不修改旧代码发布、旧缓存或环境。

## 已完成验证

- 303项全工程测试通过：pytest_final.log。
- 三分支恢复：生成器、EMA、GAN critic、双优化器、RNG和数据游标逐位一致。
- 已补测pending_evaluation中断后的三个选模产品恢复。
- 实际144x96、K4完整采样链在RTX4060通过有限非零梯度；Diffusion峰显存355.37MiB，完整35次前向和重计算反传；这是本机合成输入工程结果，不是A800预算。
- 既有Gaussian空间因子查询核验最大差0.00000159mm。
- 独立审阅完成，基准恢复误计时问题已修复并回归，未解决Critical/Important=0。

## 服务器连接和下载

当前下载入口：C:/Users/user/AppData/Local/Temp/gnss_gen3_fetch/fetch.cmd
日志：同目录fetch.log；仅当出现DOWNLOAD_SHA256与完整DOWNLOAD_BASE64后才能验核并解包。

连接加入本任务专用Python TCP传输，保留StrictHostKeyChecking=yes与原服务器身份验证，不保存密码，不改全局SSH/VPN设置。仅需用户在可见SSH窗口输入密码。

下载成功后：解码zip→核对SHA256及所有路径→保存到results/20261121_grid_generation_v01/previous→核对每折selected_ema.pt的source_sha256与旧results.json.source_model→记录外层结果（不据此重新挑内层checkpoint）。

## 后续命令

本机使用现有pytorch环境及PYTHONPATH=src：

```powershell
python tools/grid_generation_release.py
python tools/prepare_grid_generation_submission.py
```

发布构建器在缺少已核验旧结果与审计时拒绝打包。准备完成后把submission目录的submit.cmd、submit.stdin与ssh_tcp_transport.py一起复制到ASCII临时目录，通过可见SSH窗口执行。

服务器新运行目录：
`$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261121_us_dense_zwd_gen3_unet_df_gan_v01`

新代码目录：`.../code/20261121_grid_generation_v01`。

提交程序核验旧队列COMPLETE、旧入选checkpoint哈希、新发布每个文件，申请单A800后先基准，再真实100+100对连续200恢复验收，再顺序训练fold0随机/DF/GAN及fold1随机/DF/GAN。任何失败保留checkpoint，不重复提交活动任务。

暂停：在新GRID_RUN创建PAUSE；恢复：

```bash
PYTHONPATH="$GRID_RELEASE/src" "$GRID_PYTHON" "$GRID_RELEASE/tools/grid_generation_submit.py" \
 --root "$GRID_RUN" --release "$GRID_RELEASE" --python "$GRID_PYTHON" --resume
```

恢复从sacct扣除已分配墙钟，总额14400秒。SSH断开不影响Slurm。新监控建议端口8782，沿用grid_pair_monitor.py并指定新root和新SSH submit.log。观察首组真实权重变化、200步有限启动评价和动态页面后结束人工监督。

## 队列完成产物

每组双产品候选与锁、逐站集合、RMSE/MAE/bias/ES/VS/CRPS、覆盖率及宽度、均值MC标准误、Gaussian配对差值、成员置乱对照；预登记时刻的公共高程场均值/标准差及成员。汇总summary.json、RESULTS.md、training_curves.svg、calibration.svg及joint_field_case.svg。
