# Leave-out innovation audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, inline TDD and a final read-only review.

**Goal:** 用真实背景前向检验被压缩的创新信号，并量化正式制包成本。

**Architecture:** 新建独立诊断模块，复用冻结FusionData/P14Fusion；对条件站或几何组留出并重新前向，比较相同隐藏目标。

**Tech Stack:** 本机Conda pytorch、NumPy、pytest、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-11-26-leaveout-innovation-audit.md`

## Global Constraints

旧科学源码/权重/缓存不改；数据与结果role单列，旧数据限制保留；账户剩额75%停止新增任务。

## Review Focus

几何折确定性和物理组完整；背景依赖窗内全变量隐藏；cache self创新数值对齐；内存扰动恢复；LOSO拼接误差与训练OOF的区别。

## Tasks

- [x] 1. 几何折/留站集合测试RED→GREEN。
- [x] 2. 实现独立真实背景创新提取；首块82站＋8组全部计算。
- [x] 3. self缓存对齐和隐藏站扰动不变性核验，保存成本与尺度证据。
- [ ] 4. 据成本锁定扩大信息诊断的人口、拟合和验证边界并运行。
- [ ] 5. 审阅、同步轻量证据并核对远端。

## Ledger

承接增量监督双日历的微小收益，优先执行创新可预测性审计。原位使用已授权codex研究分支；首块工程测量规则先于新增结果冻结，不将其视为新模型验证。

首块完成：self/LOSO/LOGO创新RMS为4.546134/7.531074/12.580997mm；self对缓存最大差0.004704mm，隐藏站72h依赖窗内GNSS/T/P扰动对留出背景影响精确为零。LOSO82次调用约5.157秒、LOGO8次约0.468秒（5个背景起点），总约7.953秒。这是尺度与输入合同证据，不是新预测精度。
