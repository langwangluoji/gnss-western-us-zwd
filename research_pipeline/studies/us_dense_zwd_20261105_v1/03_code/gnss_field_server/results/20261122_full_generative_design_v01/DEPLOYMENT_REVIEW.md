# 部署波窄审

## 限定结论

本轮只读复核后，未发现仍会阻止当前部署入口启动的 missing import、直接 KeyError、路径/schema 错配或新增单卡收费边界问题。发现的正式入口与 preflight 确定性运行设置不一致已修复并复读确认。可以继续根任务的冻结打包与认证流程。

本结论只覆盖部署代码和下列本机检查；没有执行 SSH、Slurm 提交、真实 A800 benchmark 或完整训练，也不把根任务的全套测试结果作为本代理独立执行结果。

## 核查范围

- `tools/run_full_generative_preflight.py`
- `tools/package_full_generative_release.py`
- `results/20261122_full_generative_design_v01/remote_bridge.py`、`local_bridge.py`
- `generative/benchmark.py` 新增真实 SIGUSR1 恢复验收段
- `generative/campaign.py` 新增最终 CPU 报告提交段
- 上述入口直接依赖的冻结背景、评价归档及报告字段接口

## 关键结论

1. **运行设置一致**：正式 campaign 入口在 import torch 前设置 CUBLAS workspace；与 preflight 一致启用 deterministic algorithms、cudnn deterministic，并关闭 cudnn benchmark 和 TF32。真实 SIGUSR1 检查在第四个完成的训练批次触发，验证暂停步数、连续/恢复 raw 与 EMA 权重及 CPU RNG。
2. **打包人口与路径**：按实际 packed_root 执行相同只读选择规则，得到 A48、B36、C12、D0个月；共享文件为 geometry、hours、manifest、stations、statistics。三个冻结资产分别重写为 release 下的 Linux 绝对路径及对应 expected_sha256。元资料可能保留 D 的角色或清单描述，本次未选择 D 观测值文件。
3. **完整性与部署**：上传按 bundle SHA 校验；解包限制普通文件/目录和授权根路径；部署核对逐文件 SHA，再发布固定 release。冻结数据以独立内容身份目录保存，release 通过固定路径链接引用。代码包包含实际背景和指标工具依赖；报告需要的 background_mean、heights 已在评价归档中提供。
4. **收费和顺序**：preflight 为 A800 分区、单 GPU、2 CPU、48 GB、60分钟，无自动 requeue；先CPU audit，再 afterok GPU preflight，再 afterany CPU handoff。只有 preflight_status=COMPLETE 才进入正式 campaign。预算检查位于全量准备和 pilot 前，pilot 共享已扣费预算及绝对截止时间；正式阶段按实际 Slurm accounting 扣总额及阶段额。
5. **最终报告**：最终评价结束后提交 intel CPU 报告任务，显式2 CPU、16 GB，无 GPU 请求；提交意图和返回 job_id 持久化，重复调用不会再次提交同一报告。报告执行失败会写 error 状态，不新增 GPU 任务。

## 本代理实际验证

- 对8个本轮入口/修改文件执行 AST 解析：通过。
- 只读统计实际数据选择：A48、B36、C12、D0，5个共享根文件。
- 两项针对性调度测试：**2 passed，0.19秒**，覆盖单卡 GPU+CPU dispatcher 以及最终评价 chunk 参数。
- 对最终 CPU 报告分支执行无提交替身：连续调用两次，仅形成1条 sbatch 命令；返回相同 report job_id；partition=intel；GPU请求=False。
- 复读正式入口确定性设置修复。

本轮未独立展开或重新校验最终 tar 文件；打包与桥接代码会在归档生成和服务器部署前执行各自的完整性校验。未运行桥接脚本的顶层代码，未建立任何远程连接。
