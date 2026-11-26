# Task 06 修复复审

日期：2026-11-17。范围限定于 `task-06-fix-report.md` 所述 CLI/runtime 哈希绑定、SSH 断连展示和附带预算元数据、队列摘要；未扩大到科学训练流水线或全套测试。未修改实现文件。

## 结论

本轮所审修复通过。原审查 P1 与 P2 均关闭，未发现本轮范围内新增的可执行缺陷。服务器真实分配、信号与恢复验收仍保留为部署前检查；本结论不替代该验收。

## P1：benchmark 内容绑定已闭合

- `cli.py:297-311` 从 campaign 的 `resolved.json` 读取执行配置；显式配置或路径/device override 不一致时拒绝，再重新计算科学配置、数据、发布源码和协议四项哈希。只有匹配后才调用 benchmark。
- `runtime.py:25-35` 的 `verified_release_hashes` 要求 registered pack；`PackedStore` 校验 manifest 自身摘要与共享文件，`verify_all()` 校验所有数据分片字节。修改配置、源码或数据内容会改变摘要或直接报错。
- `runtime.py:373-381` 的公开 Python benchmark API 自行校验真实哈希，检查位于 doctor 与工程训练之前。worker 同样使用此函数，并把 project_root 传入 runtime benchmark。
- 真正的预算验证标志继续位于 L40 检查、实际 Slurm 步骤 USR1/恢复检查之后，本轮未弱化信号门槛。

新增测试覆盖四项哈希不一致、synthetic pack、显式 config/data-root 改动、真实配置/源码摘要变化、分片损坏和直接 Python API 拒绝。原先 rank128 campaign 配合 rank64 调用的路径现在会在训练前拒绝。

## P2：断连状态已独立展示

- `monitor.py` 在失败后保留 last_success，同时更新 last_attempt 和错误信息；首次失败且没有旧快照时也能返回同步状态。
- `monitor.html:15` 在选择训练 run 之前呈现 `payload.ssh_sync`，明确区分成功与失败，并显示最近成功、尝试时间和错误。旧训练指标及其时间保持原值。
- Node 执行测试实际运行页面的 render 函数，模拟成功后断连，验证失败文字和错误信息出现，同时训练阶段、更新时间、原始指标与损失图内容不变。该测试验证渲染逻辑，未冒充浏览器布局验收。

## 附带变更

预算元数据使用 basis effective16、M2 effective8，符合已批准方案；M0 使用匹配 runtime micro_hours/patch_chunk 的 effective8 实测 sweep，其他头按两小时内核线性外推 effective32。字段明确给出最大更新数、秒/更新、卡时与 2 CNY/卡时的 GPU 费用，并列明单阶段及验证、I/O、试验、重试等估算边界。真实 pipeline 的有效 batch 执行检查已交全项目审查，本复审不重复裁定。

队列摘要读取现有 campaign manifest，不查询或修改远程队列；SSH 参数及只读采集边界保持原状。

## 本次独立验证

执行：`D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_cli_monitor.py tests/test_runtime.py -q`

结果：**23 passed in 4.92s**，无跳过；包括 Node 前端执行测试。未执行全套测试、GPU 训练、SSH 连接或 Slurm/付费作业。
