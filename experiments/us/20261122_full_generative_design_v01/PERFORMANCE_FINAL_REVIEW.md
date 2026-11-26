# 性能修订最终窄审

## 限定结论

本轮发现的3项阻碍已修复并完成针对性复核；截至本次代码快照，未发现新的剩余阻碍。可以进入新运行目录下的真实 A800 性能重测及24卡时预算门。该结论不代表已经测得24小时内完成，也不改变数据、训练轮次、概率成员或18步采样合同。

范围：B16小时/member16、最多256小时成员对的评价与诊断分批；ObservationCritic相同有效pair的向量化；warm8→resume32持续计时；4个准备worker的月归属、缓存路径和启动计价；prepare→diagnostics→模型阶段的预算、暂停与重试；runtime_v03部署入口。未修改实现，未SSH或提交任务，未运行历史全库测试。

## 本轮阻碍及修复

### 1. 旧条件身份被新代码目录路径改变

原定位：`campaign.build_data` 的冻结背景源码资产路径，以及 `benchmark.run_benchmark` 的旧condition_identity检查。

最小复现：将相同字节的 background.py 放在 release_v02 和 runtime_v03 两个目录，给 VerifiedBackgroundProvider 相同operator描述；两者identity不同。仅保持三个冻结资产和data.py内容相同不能消除该路径差异，旧benchmark会拒绝重测。

已修复：`background_code_binding` 允许显式锚定v02源码路径，先要求锚定文件和当前执行源码SHA完全相同，再使用旧路径键。旧benchmark的actual_device和metadata fingerprint核对继续保留。针对性回归通过。

### 2. pilot训练受控暂停被提升为error

原定位：`diagnostics._execute_pilot` 的run_stage返回检查与 `Campaign.run_auxiliary` 异常分类。

最小复现：run_stage返回`status='paused'`及可恢复checkpoint，旧_execute_pilot抛RuntimeError；auxiliary因此保存error，而自动retry拒绝error。

已修复：paused分支转为TimeoutError，auxiliary保存paused；其他不完整结果仍作为error处理。新增回归先注入paused，再完成同一pilot并检查结果；通过。

### 3. 完成的pilot在阶段恢复时重新评分

原定位：`diagnostics._pilot` / `run_representation_diagnostics`。

最小复现：某fold已有完整outcome，阶段晚期暂停后重新进入诊断；旧入口仍调用_execute_pilot，重做已完成fold的模型加载与864小时评分，重复消耗剩余阶段额度。

已修复：发布outcome.commit，绑定outcome与population锁SHA、pilot科学/数据身份及checkpoint SHA；已完成且身份一致的fold直接复用原始结果。未提交或损坏的结果不冒充完成。新增回归禁止再次调用_execute_pilot，复用通过。

## 实际复核证据

- preparation、benchmark及6项staged/budget针对性检查：**23 passed，3.38秒**。
- 新anchor、pilot暂停/复用、B分批诊断及critic梯度检查：**4 passed，7.14秒**。
- 独立逐行pair参考实现：N48、3种mask（含全空），原始四个距离bin有76/201/529/315个pair，覆盖超过256的截断；向量化与参考实现的score最大差、输入梯度最大差均为**0**。
- 评价/诊断分批源码保留逐hour、member和stream的keyed noise，18步Heun合同未改；query/mean/mask沿小时顺序合并。未重复根任务已完成的36项评价检查。
- worker使用`data.config.cache_root`这一未加身份前缀的配置路径；父进程核对worker解析出的实际身份目录，避免双重namespace。worker的确定性与TF32设置已与正式入口一致。

## 最后部署入口核对

已只读检查 `runtime_v03/remote_bridge.py`、`authenticated_runner.py` 及现有code.tar.gz中的实际配置：

- 新release v03、新RUN v02；三个冻结资产、raw数据和背景源码锚定保持v02。
- GPU计费前CPU probe强制核对当前metadata fingerprint与原始benchmark一致，同时核对B条件和标签身份。
- 新recheck配置budget_seconds=3000，Slurm硬墙54分钟=3240秒、单A800、8 CPU、48 GB；加入已核实旧任务3697508的356秒后，最大累计为**3596秒**，小于1小时。
- 该旧任务以356秒写入新配置preflight_jobs；源benchmark路径为新RUN中的original_benchmark.json。
- staged_preparation=True；自动序列先prepare、再diagnostics，之后才进入模型阶段。完整prepare/diagnostics在24卡时内单独计价；短benchmark/recheck仍受累计1小时约束。
- 实际配置evaluation_hours_batch=16、evaluation_chunk=16，decoder上限256；部署脚本AST解析通过。未执行其顶层连接或提交动作。

真实持续速率、准备并行率和最终预算仍由新A800重测产物决定；旧301.414秒benchmark墙时与旧356秒Slurm计费时间分别保留，不相互替代。
