# 稀疏站残差 EDM 与 Optuna

本包实现用户于2026-11-13确认的两阶段方案，原模型与原实验目录不作修改。
冻结 RAW_CURRENT seed17，使用39站同期五变量和四帧粗场；模型只在训练定义域D生成残差。
研究成功与代码测试通过分别报告，2020—2021始终标为回顾评价。

## 运行

在本研究的 `03_code` 目录运行，解释器固定为
`D:\anaconda3-2025.12-2\envs\pytorch\python.exe`。

```powershell
& 'D:\anaconda3-2025.12-2\envs\pytorch\python.exe' -X utf8 -m unittest discover -s residual_edm_v1/tests
& 'D:\anaconda3-2025.12-2\envs\pytorch\python.exe' -X utf8 -m residual_edm_v1 audit
& 'D:\anaconda3-2025.12-2\envs\pytorch\python.exe' -X utf8 -m residual_edm_v1 cache-mean
& 'D:\anaconda3-2025.12-2\envs\pytorch\python.exe' -X utf8 -m residual_edm_v1 smoke
& 'D:\anaconda3-2025.12-2\envs\pytorch\python.exe' -X utf8 -m residual_edm_v1 run
```

`run`按已登记预算完成第一阶段；通过三种子门槛才训练早期均值并进入第二阶段。
`search --arm det|gaussian|edm`、`promote`、`replicate`、`evaluate`支持单独运行。
`report`只汇总现有结果，不启动训练。`--stage stage2`选择第二阶段。
默认输出为 `04_models/residual_edm_v1/stage1` 或 `stage2`。

## 数据和接口

`data.py`提供audit、FrozenMean、cache_mean、ExperimentData。
batch包含coarse、stations、padding、maps、mu_condition、baseline_condition、domain、mu_mm、context_support；
仅训练的 `targets=True` 加入residual。推理使用 `targets=False`。
训练才使用旧usable_pair_hours；开发和回顾小时只检查历史边界，避免teacher可用性影响输入资格。
当前训练完整D小时65369，早期48752，晚期16617；定义域226格。
旧field_mm先乘20，残差再除以训练2×RMS；不去均值。

`model.py`提供ResidualModel(det/gaussian/edm)、encode_condition、denoise、sample。
17维站输入→64维四头；42通道U-Net，32/64/128；EDM sigma_data=.5；Heun18步、ρ7、σ范围.002—80。
域外残差恒零，合法背景条件可全域输入；不强制4×4块零均值。

`training.py`提供可恢复训练、Optuna顺序搜索；checkpoint保存模型、EMA、AdamW、RNG、样本排列和游标。
学习率3e-5—3e-4，weight decay1e-6—1e-2，EMA=.99/.999/.9999；EDM额外P_mean=-1.8—-.6，P_std=.8—1.5。
有效batch32，microbatch8，OOM只在前向/反向阶段降为4或2；AdamW更新阶段OOM直接失败，不在部分更新状态上重试。
第一阶段各12个trial（失败和剪枝计入），第二阶段各6个；每trial10000步。
TPE种子20261113，启动4trial；MedianPruner预热4000步，2000步间隔。
每种模型按2019月份×小时固定576时刻选型；EDM K8、固定噪声。
第一阶段最佳3个/第二阶段最佳2个完整trial延长50000步；完整2019 K32→64，必要时128，差值稳定阈值.005mm。

`workflow.py`负责选型、复验、锁定和汇总。精度门槛：相对均值改善≥.05mm、相对确定性残差≥.02mm；
三种子中最差相对确定性恶化≤.05mm。门槛是工程决策，时间块统计区间单列。
`early_mean.py`从头训练00—14均值seed17，原训练损失、4完整遍历；仅早期拟合统计；晚期15—18用于新残差训练。

## 评价和结果

主评分资格由D内插值模板、当前context支持、目标站角色和观测有效性固定；不依赖模型输出和teacher掩码。
先插值Z0，再乘站高恢复因子。记录主域和扩展D域分数、coverage、CRPS、区间、联合评分、运行时间、显存与NFE。
有限集合CRPS按经验集合计算，记录K和估计器；训练种子与采样种子分开。
保存站点样本、格网均值/标准差以及每月预先按时间选取的事件场样本；不保存全部格网集合。
配对7日块bootstrap共2000次，同一时间块用于所有站，各重复重新计算站宏RMSE。

状态查看：`status.json`、`*_trials.json`、各trial的`training.jsonl`、`best.json`、`selection.json`、`replication.json`。
SQLite保存trial身份；RUNNING恢复原trial，已有完成/剪枝/失败trial不重复创建。
模型/data/checkpoint契约不匹配时停止，不默默重用旧结果；均值缓存中断会保留partial文件供检查。
正式运行可能持续较长时间；剩余预算与实际GPU时间以文件为准，不预报未经测量的完成时间。
