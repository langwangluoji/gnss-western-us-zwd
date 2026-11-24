# GNSS ZWD 场重建工程

> 当前主线入口：先读仓库根目录的 [快速上手](../../../../../docs/GETTING_STARTED.md)、[项目架构](../../../../../docs/PROJECT_ARCHITECTURE.md) 和 [方法验证规则](../../../../../docs/METHOD_VALIDATION.md)。

## 当前实现状态

本目录同时承载两类代码：`tools/terrain_diffusion/` 是 2026-11-23 之后的本机基础场、观测编码和残差场探索；`src/`、`configs/`、`slurm/` 及其余工具是早期服务器化生成式工程。两者共享数据合同和评价原则，但结果必须按实验目录和协议解释。当前最新基础场结果见 `results/20261124_unet_foundation_v13/RESULTS.md`。

当前最强开发候选是点特征 MLP＋小型 U-Net 空间修正，双种子平均宏 RMSE 10.828 mm；大宽度 U-Net 实验已完成，平均 11.800 mm。后续先做连续数据审计和点—图—查询编码，不把增加 U-Net 宽度作为默认优化。

## 当前工具入口

| 目标 | 入口 | 产物 |
|---|---|---|
| 基础场训练/评价 | `tools/terrain_diffusion/unet_foundation.py` | `results/20261124_unet_foundation_v13/` |
| 点—空间融合 | `tools/terrain_diffusion/unet_hybrid.py` | E1304 结果 |
| 大宽度对照 | `tools/terrain_diffusion/unet_large.py` | E1305 结果 |
| 诊断图表 | `tools/terrain_diffusion/unet_diagnostics.py` | terrain/month/spatial CSV 和 PNG |
| 结果审计 | `tools/terrain_diffusion/unet_audit.py` | `final_audit.json` |
| 历史扩散/图/序列 | `tools/` 下对应 `diffusion_*`、`graph_*`、`sequence_*` | 按实验索引定位 |

## 运行边界

训练前必须锁定数据版本、变量清单、站点角色、窗口、布局、种子、预算和评价日期。禁止把确认站或原 C/D 标签用于开发。禁止 PWV/IWV/TCWV/Tm 作为新模型输入或教师。权重、原始数据和大数组保留本机；轻量协议、指标、报告和哈希清单同步 GitHub。

---

> 当前科学进展请先读 [项目进展](../../../../../docs/CURRENT_RESEARCH.md) 和 [实验索引](../../../../../docs/EXPERIMENT_INDEX.md)。下文为2026-11-18的工程与迁移说明，历史验收项及服务器待办以对应记录为准；不代表11月23日最新模型排名。构建/发行暂存副本已外移，需使用源码重新打包。

# GNSS高分辨率场：同济超算工程

本工程实现“冻结M0基础场 → 连续空间残差 → 条件EDM → 真实站ES微调”，并提供同条件的确定性、高斯系数与GAN对照。采用原生PyTorch、单GPU训练任务和Slurm阶段依赖；GPU并发上限为2。25 km是发布格网间距，细尺度能力由独立站和站间差值评定。

当前交付是可迁移的服务器工程、封装数据和本机验收结果。正式精度由服务器训练后的锁定评价产生。旧实验继续暂停；本次未提交服务器收费作业。

## 1. 工程与数据

```text
src/gnss_field/
  data/             按角色、月份封装；时间连接；A输入条件
  geometry/         Lambert投影、77小块拼接、站高恢复和支持域
  models/           独立M0/M2、条件编码、空间基和生成头
  objectives/       MSE/NLL、EDM、真实观测ES和WGAN-GP
  training/         阶段训练、EMA、精确恢复与生产流程
  evaluation/       点值、细尺度、概率、情景、bootstrap和产品锁
  orchestration/    Slurm依赖、预算、Optuna和产物回执
configs/            科学配置与机器配置
slurm/              模板说明；实际脚本由campaign plan生成
docs/               批准方案、实现记录及审查记录
tests/              数值、角色隔离、训练恢复与作业编排测试
local_validation/   本机验收证据，不属于科学结果
```

数据已另存于研究目录的`02_data/processed/gnss_field_server_v1`，240个分角色月片覆盖2016—2020。迁移时保留完整目录结构及manifest，不将旧Windows训练目录作为服务器依赖。

| 角色/时段 | 作用 |
|---|---|
| A174 | 唯一观测输入，五变量共同有效；四帧A站IDW条件场 |
| B511，2016—2017 | M0训练和基础输入统计 |
| B511，2018 | 表示学习、预测头和真实站ES微调 |
| C110，2019 | 开发、双轨选模 |
| D110，2020 | 产品锁定后的回顾评价 |

沿用名义小时与EGM96。全域144×96、25 km格网，由77个24×24重叠小块拼接。归算关系为`ZWD=exp(-H/3000)×Interp(Z0)`，全部评分使用同一发布场查询算子。

真实数据审计得到：2016—2017合格基础小时17526；2018有支持B站小时8736；R=128、至少136站的系数拟合小时8712；2016—2018合格监督站时刻11469936。ES另按至少4站筛选，评价使用锁定F/S0/T清单。

## 2. Linux环境与迁移

平台驱动保持原状。以下安装发生于独立Conda环境，CUDA wheel版本应与实际驱动预检相符。

```bash
conda create -n gnss-field python=3.11 -y
conda activate gnss-field
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r configs/requirements-linux.txt
python -m pip install --no-deps dist/gnss_field-0.1.1-py3-none-any.whl
python -m gnss_field --help
```

编辑`configs/tongji_l40.yaml`中的持久运行目录、热数据/缓存目录和真实分区名；路径支持`~`及`${USER}`。环境名默认为`gnss-field`。正式脚本使用平台Conda初始化路径；若账户安装位置不同，应在生成脚本前调整调度模板并重新生成代码哈希。

默认单任务L40一张、7CPU、64GB内存。平台文档限制每GPU最多7CPU，因此把批准草案的8核修正为7核；科学有效批量不变。L40 GPU单项2元/卡时，实际账户配额、CPU/存储费用和可用资源以预检为准。

## 3. 先基准，后预算，再正式实验

全局参数放在子命令之前。规划、状态、报告默认不训练、不提交作业。

```bash
python -m gnss_field --config configs/tongji_l40.yaml audit --output validation/audit
python -m gnss_field --config configs/tongji_l40.yaml campaign plan \
  --scope bootstrap --directory runs/bootstrap
python -m gnss_field --config configs/tongji_l40.yaml doctor --site Tongji \
  --campaign-dir runs/bootstrap --output runs/bootstrap/preflight.json
python -m gnss_field campaign status --directory runs/bootstrap
```

bootstrap包含pack、audit、doctor、benchmark四个阶段，总GPU预留最多4卡时。每个阶段成功后，再显式提交下一阶段；控制器检查上游回执、哈希、账户和预留。省略`--execute`只显示提交计划。

```bash
python -m gnss_field campaign submit --directory runs/bootstrap --phase 0 \
  --preflight runs/bootstrap/preflight.json --benchmark-cap 4 --execute
# 依次完成phase 0、1、2后，以相同形式提交phase 3。
```

基准会在真实L40 `srun`步骤内测量完整场训练、逆算、各预测头预训练与观测更新、采样、缓存和恢复。实际USR1必须到达Python训练进程，在更新完成后保存，恢复结果与连续训练相同才可解锁预算。本机`smoke`始终不能授权Tongji正式预算。

基准输出`benchmark/benchmark.json`及`verified_preflight.json`。根据测量填写GPU任务/阶段卡时表，例如`{"mean_base":12,"mean_all":12,...}`；数值由实测确定，必须覆盖每个GPU任务。数组任务使用共同墙钟上限，长任务分段续跑；预算同时包含允许的尝试次数。默认蓝图中的24小时为保守占位上限，不是预计训练耗时。

```bash
python -m gnss_field --config configs/tongji_l40.yaml campaign plan \
  --scope research --directory runs/research \
  --benchmark runs/bootstrap/benchmark/benchmark.json --reservations reservations.json
python -m gnss_field campaign lock-budget --directory runs/research \
  --benchmark runs/bootstrap/benchmark/benchmark.json --total-card-hours TOTAL
python -m gnss_field campaign submit --directory runs/research --phase 0 \
  --preflight runs/bootstrap/benchmark/verified_preflight.json \
  --budget-lock runs/research/budget.lock.json --execute
```

科学配置、代码、数据和协议哈希必须与基准一致；变更后建立新实验。`resolved.json`固定完整科学配置与运行配置；`execution.json`固定执行目录与环境。

## 4. 实验流程与选模

生产任务链为M0/M0-all → 均值缓存 → 三折表示验证及门槛 → 最终空间基 → 共享编码器 → 五个study → 候选充分训练 → 完整2019重排及采样稳定性 → 方向门槛 → 消融/种子 → 产品锁 → 2020评价和导出。任一科学门槛未通过时记录独立状态，停止依赖扩展。

空间基27→256→256→R，主R=128。每小时每个成员仅有一个全域系数向量。高斯、EDM和GAN的系数预训练只作初始化，最终候选来自真实B站观测ES微调。EDM使用18步Heun、35次去噪调用，K=4训练；开发筛选K=16，完整重排K=64，K=128核对前缀与结论。

| 产品 | 选模依据 | 主要比较 |
|---|---|---|
| point_best | 站宏RMSE | 固定点预测对照B*及自身100 km版本 |
| joint_best | 预登记站组宏ES | 同表示高斯系数模型，辅以VS与CRPS |
| marginal_best | 站宏CRPS | M2边缘概率对照，不进入联合ES排名 |

每产品保留前三检查点，在完整2019重新排序。启动200步检查不选模；正式评价每1000步。最终报告包含RMSE/MAE/偏差、覆盖率、CRPS/NLL/区间、ES/VS、短距离差值、100 km降采样对照，以及地形/天气/稀疏和边界情景。2000次同步7日时间块bootstrap另给14日与密集站组敏感性。

## 5. 暂停、恢复和只读监控

```bash
# 请求训练在完成当前优化更新后保存；目录下全部任务检查此标志。
touch runs/research/PAUSE
# 恢复前移除标志；resume不重复提交仍活动的任务。
rm runs/research/PAUSE
python -m gnss_field campaign resume --directory runs/research --phase PHASE \
  --preflight runs/bootstrap/benchmark/verified_preflight.json \
  --budget-lock runs/research/budget.lock.json --execute
```

需要立即撤销活动作业时可使用`campaign pause --execute`，或平台`scancel JOBID`；恢复从最近完整检查点开始。预超时信号在`USR1@180`到达作业步骤；处理器只置位，不在信号内部写文件。不会无限自动requeue。

本机页面每5秒刷新，SSH每10秒只读拉取状态；断连保留上次真实记录。远端需能用`python3`导入本工程，可在SSH登录环境配置该独立环境。

```powershell
python -m gnss_field monitor --output monitor_cache --port 8767 `
  --ssh-host USER@HOST --remote-root /persistent/path/runs/research
```

浏览器打开`http://127.0.0.1:8767`。身份认证使用已有SSH配置，不存储密码。页面显示实际loss、RMSE、ES、CRPS、运行阶段、步数及更新时间，不插补精度。日志位于任务训练目录`training.jsonl`、`development.jsonl`，Slurm输出位于campaign的`logs/`。

## 6. 验收与结果解释

```bash
python -m pytest -q
python -m gnss_field --config configs/local.yaml smoke \
  --output local_validation/real_cuda --resume-updates 200
python -m gnss_field report --output runs/research
```

本机真实数据验收覆盖77块整场训练、FP64逆算、35NFE可微采样、站点矩阵与格网查询等价，以及实际ES训练200步和100+恢复100步的逐位一致。各模型关键路径的小样本测速验证了代码可执行，不用于预报正式RMSE或L40总耗时。

最终本地测试为193项全部通过；安装包已在源码目录之外独立安装，核对实际执行模块与发布源码一致。审查修复包括真实阶段批量、M2边缘概率权重锁定、固定评分集合bootstrap、名义覆盖率确认和7/14日敏感性。M2边缘概率对照单独交付站点分布，联合格网成员由高斯系数、EDM和GAN产品提供。

`tools/package_release.py`生成分离的代码/数据ZIP并核验每个文件SHA256和压缩包CRC；`tools/verify_wheel.py`检验安装后的实际代码。发布包附有小型验收证据，训练检查点与热缓存另行管理。

服务器首批仍需完成：账户/分区/驱动、真实步骤信号、L40吞吐量与存储配额，然后锁定正式预算。工程状态与各项测试证据见`docs/IMPLEMENTATION_LEDGER.md`和`local_validation/FINAL_VALIDATION.json`；最终科学结果生成后由报告模块汇总。
