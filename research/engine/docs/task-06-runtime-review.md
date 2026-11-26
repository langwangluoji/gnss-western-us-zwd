# Runtime / CLI 集成审查

审查日期：2026-11-17。范围：CLI、runtime、只读监控、报告、发布摘要与配置，并追踪必要的 orchestration 接口。审查不启动训练、SSH 或 Slurm 作业；未改动实现文件。依据为 `docs/APPROVED_PLAN.md`，7 CPU 的部署修正按 `IMPLEMENTATION_LEDGER.md` 已记录裁定执行。

## 可复核问题

### P1：独立 benchmark 入口未核验当前内容与 campaign 哈希一致

位置：`src/gnss_field/cli.py:295-301`。

`settings(args)` 读取当前调用的配置和路径，但此分支直接把 `campaign.json` 中的哈希传给 `benchmark()`。runtime 仅检查 preflight 与传入哈希一致；成功后把这些哈希写进预算验证报告。与之相比，worker 在 `cli.py:143-150` 验证真实数据包、科学配置、发布源码和协议，并拒绝 synthetic pack。

触发条件：先为 rank128 / 数据 A / 发布 A 规划 campaign，再通过独立 benchmark 命令传入 rank64 / 数据 B，或修改源码后继续使用旧 campaign。真实 L40 信号测试即使成功，也不能证明其吞吐量对应被标记的 campaign。该报告随后能通过 research plan / lock-budget 的哈希比较，影响正式付费计算预算依据。

本地调用探针：在临时目录写入 rank128 的 campaign 哈希和匹配 preflight，给 CLI 传入 rank64、different-data 配置；仅 mock `gnss_field.runtime.benchmark` 捕获参数。输出 `actual_rank:64`、`actual_plan_matches:false`，CLI 返回 0。未运行任何训练或远程命令。

建议：独立 benchmark 在执行前读取 campaign 的 resolved 配置，并重新计算四项真实哈希；若允许显式 override，则同样检查并拒绝不一致。共享 worker 的 registered-pack 检查。增加配置、数据、源码不匹配时在训练前拒绝的针对性测试。

### P2：SSH 断连后的网页不显示同步失败

位置：`src/gnss_field/resources/monitor.html:15-16`；数据来源为 `monitor.py:31-33,54-60`。

SSH 失败时，服务保留旧快照并附加 `ssh_sync.sync_ok=false`。HTTP `/api/status` 仍正常返回 200；前端只在 fetch/JSON 失败时显示读取失败，完全不读取 `ssh_sync`。因此断连后旧的 RUNNING、步数和图线继续显示，用户只能从旧的更新时间自行猜测远端连接状态。

验证：现有 `test_ssh_sync_readonly_literal_path_and_stale_failure` 覆盖后端保留旧快照；前端源码检索不存在 `ssh_sync` 或 `sync_ok` 引用，且 `render()` 仅用运行状态更新时间。此问题是展示丢失，不涉及远程写操作。

建议：页面独立展示同步状态、最后成功同步时间与最新失败时间，保留旧指标和其真实采集时间。增加成功同步后断连的前端检查。

## 通过的边界与验收判断

- Slurm worker 与当前 `TrainingPipeline.run_task` 参数接口一致；replicate 的基准 seed 与目标 seed 分开传递。
- 标准 worker 重新核验配置、数据、代码、协议；非 GPU 阶段强制 CPU。计划及默认状态查询不调用 Slurm 提交。
- benchmark 预算验证标志位在真实 `srun` 步骤 USR1 到达、完整更新后暂停和恢复逐位一致后才设置。真实 Linux/Slurm 分支本次未执行，保留服务器验收要求。
- Slurm 脚本以 GPU reservation 限制墙钟时长；初始 benchmark 的单任务和总声明上限受四卡时限制。
- SSH 命令采用参数列表，远端路径作为 Python 字面量后进行 shell 引用；远端只调用状态采集，本地监听 127.0.0.1。未发现该路径写远端训练状态或提交作业。
- 发布摘要包含 src/configs/slurm，忽略 pycache/pyc/egg-info；运行产物及文档写入不会改变摘要。报告按实际 metrics.json 字段读取，不启动训练或进行 2020 选模。

规范结论：存在一项阻断预算依据可信绑定的 P1；应修正独立 benchmark 入口后再认定部署入口验收通过。质量结论：其余所审边界结构明确，另有一项监控同步状态展示 P2。父任务已提供的十项 CLI/runtime 测试和 CUDA profile smoke 通过结果作为既有证据；本次未重复执行整套测试，未把本地 mock 视为远程信号验证。
