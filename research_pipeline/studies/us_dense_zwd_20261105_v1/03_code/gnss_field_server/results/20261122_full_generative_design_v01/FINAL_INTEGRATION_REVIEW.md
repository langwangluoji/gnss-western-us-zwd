# 完整生成管线整合审查

## 本轮结论

截至本轮复核，原发现的阻碍已完成相应修复；在本次限定范围内未发现仍需阻止整合的代码问题。正式服务器运行仍以真实 A800 benchmark、预算锁和数据/产物身份检查为启动条件。本结论不代表历史全库、真实 A800 benchmark 或完整科学训练已通过。

审查范围：`generative/benchmark.py`、`evaluation.py`、`campaign.py`、`diagnostics.py`、`training.py` 的科学流程、阶段依赖、预算和恢复接口。未重审已确认的 architecture/data；未修改实现文件，未启动正式训练或服务器任务。

## 重要发现及修复复核

| 原发现 | 最小复现或风险 | 已复核的修复 |
|---|---|---|
| B 标签版本遗漏 | 修改有限 B targets、保持条件和小时不变，旧实现仍从 step1 续到 step2 | trainer data_identity 绑定 B-label identity；checkpoint/cache 同步保存并核对 |
| 选择产物提交顺序 | 注入 best 文件写入异常，旧 last.pt 已 pending=False，但候选不存在，恢复跳过评价 | 先发布独立候选及 SHA，再提交 last.pt；失败回滚 pending、曲线和候选池 |
| 576 小时开发人口与预算不一致 | 旧通用入口可回退到全年 C，而 benchmark 按576计价 | 正式配置显式使用原日历576小时；campaign 检查登记人口 |
| latent128 定价缺口 | 诊断允许128，而原主 benchmark 按64计价 | benchmark 全流程强制128并登记 priced_latent_dim=128；campaign 检查所选潜维不超测量定价 |
| 最终评价继承旧 allocation 预算 | 只读替身实得旧 candidate budget7000、本次 runtime chunk600 | 最终评价统一当前任务 deadline，逐候选传剩余秒，Slurm evaluate 命令传本次 chunk |
| 块内停止丢失前缀 | 两次各计算3小时，旧实现均 committed_blocks=0 | 小时停止返回已完成前缀；内容寻址文件与原子块指针提交；恢复从前缀末尾继续 |
| C 标签版本遗漏 | 第168小时后中断；C标签100改200恢复，旧输出混合100/200两个版本 | evaluation lock 加 C_label_identity；campaign 核对当前 audit_C 的标签、条件及小时身份 |
| 诊断先运行后验预算 | 旧12个pilot各1800秒，仅全部跑完后检查1小时总cap | pilot启动前检查实测计划；共享ledger与绝对deadline扣费；已统一 remaining_seconds/deadline_unix 接口 |

完整模型分支继续从独立副本更新编码器/解码器，冻结表示阶段保留公共表示；posterior cache 只供固定表示的预训练阶段。最终候选池保留正式 point/joint 各前三名，并与 baseline、启动诊断评价分开。上述接口已逐段复核，未发现新增跨阶段依赖阻碍。

## 本次验证

- 独立运行：`tests/test_full_generative_benchmark.py`、`tests/test_full_generative_campaign.py`、`tests/test_full_generative_evaluation.py`，以及 `test_all_pilots_share_one_cumulative_budget`：**40 passed，5.12秒**。环境为指定 PyTorch Python，`PYTHONPATH=src;tools;tests`，CPU，OMP/MKL线程2。
- 对 benchmark 生产控制流执行无训练替身检查：覆盖 control、VAE、posterior cache、GAN/DF pretrain、三个 shared 和三个 complete 路径，以及最终记录生成。结果为 **10个优化阶段计价项完成，priced_dim=128**；所有临时产物均位于明确标记 MOCK_ONLY 的临时目录。本检查只验证控制流可达性，不提供真实性能数据。
- 已实测原问题的预算继承、C版本混合、停止前缀丢失；修复后相应回归均纳入上述针对性检查。
- 前一轮数据/训练针对性检查为35 passed、146.91秒，属于较早快照；本报告不把该结果扩大为最终全库检查。根任务另行负责最终整合检查和真实 CUDA 接受证据。

## 运行边界

本轮没有执行真实 A800 benchmark。真实 GPU 目标路径、冷读、实际采样反向传播、写盘和峰值显存仍以 benchmark 实际输出为依据；替身检查和两项纯函数测试均不能替代它。预算门保持未实测即拒绝正式启动的流程。本轮未发现可绕开上述实测条件的新入口。
