# Within-forecast increment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Work proceeds inline with TDD and one final read-only review.

**Goal:** 在原H相同训练合同上验证内部lead增量监督。

**Architecture:** 复用原H、缓存和匹配AdamW续训；新增独立损失模块与封存训练/导出/评价器。旧科学源码及结果不改。

**Tech Stack:** 本机Conda pytorch、RTX4060、pytest、NumPy、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-11-26-within-forecast-increment.md`

## Global Constraints

lambda=0.5，双种子42/17各3000，连续训练块，raw主/EMA并列，保持原数据限制和确认隔离。每次代码变更在真实运行前冻结哈希；旧结果不覆盖。

## Review Focus

- 同起点lead差分与跨起点差分必须区分。
- NaN标签不会产生NaN梯度，联合mask和站权重手算一致。
- lambda0新训练器与旧训练器4更新全状态精确等价。
- raw/EMA必须绑定final3000和相应导出/缓存。
- 内部增量改善不能掩盖当前场或六小时代价。

## Tasks

- [x] 1. 写数值与mask测试，看到RED，新增独立loss，GREEN。
- [x] 2. 新增独立H训练/导出/评价脚本与campaign；真实4更新等价及新损失小试。
- [x] 3. 固定代码/协议与小试证据并同步；双种子正式3000。
- [x] 4. 汇总来源、raw/EMA、分层与共同月份区间；必要时补充日历匹配验证。
- [x] 5. 最终只读审阅、核验、更新入口并推送核对远端。

## Ledger

2026-11-26：从已核验H＋S结果推进E3；GPU无Python计算进程，额度剩80%。原位在codex研究分支实施，避免复制大型缓存和改变历史源路径。协议先于新增训练结果固定。

真实4更新：lambda0与原H的model/EMA/optimizer/scheduler/CPU和CUDA RNG/step全部精确相同；lambda0.5改变权重，每步10752个有限查询标签，warm三项为零。新损失及原H/HS回归共28项通过。[等价证据](../../../experiments/us/20261126_increment_smoke/equivalence.json)。

正式两种子各3000完成，约388秒/种子，raw/EMA主日历及冻结补充日历全部完成。内部增量微小改善，两日历六小时改善约0.003–0.005mm，补充区间跨零。维护35项通过，最终只读审阅主日历代码/权重/区间无Critical/Important/需处理Minor；补充结果沿同一核验器复算完成。协议spec及其预训练hash保持不变，实际状态写本计划及结果页。

最终增补审阅：补充raw内部增量及当前/F6共同月份区间由独立审阅精确复现；两日历研究说明与数据一致，无新增Critical/Important/需处理Minor。冻结源码/spec、补充触发receipt及相对链接复核通过，轻量证据随本轮结果同步。
