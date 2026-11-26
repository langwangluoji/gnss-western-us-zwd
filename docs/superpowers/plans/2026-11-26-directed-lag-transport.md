# Directed station-lag transport Implementation Plan

状态：v1因异步缺测归一化Important问题在正式队列1532/1530步工程停止；原文件保留，[后续v2计划](2026-11-26-directed-lag-v2.md)从原来源重训。v1部分权重不参与正式精度排名，原架构假设及比较合同保持。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, inline TDD and one final read-only review.

**Goal:** 在匹配H_delta预算上验证显式有向站点×lag传递。

**Architecture:** 保留原H前向与源embedding，新增零读出的四头数值传递；factorized/joint同参数，只改变lag与空间的耦合。原源码和实验锁保持不变。

**Tech Stack:** 本机Conda pytorch、RTX4060、pytest、NumPy、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-11-26-directed-lag-transport.md`

## Global Constraints

每臂双种子各3000连续块，72h状态/146h上游合同；源TF3000、相同H_delta损失及优化器；raw主/EMA并列，不读取确认/C/D。

## Review Focus

source和H初始化匹配；物理变量单位与独立mask；查询子集和站点排列；lag位置和因果前缀；相对同预算H_delta及factorized的归因。

## Tasks

- [x] 1. 新分支测试RED→GREEN，模型warm及不变性/梯度核验。
- [x] 2. 独立训练/导出/评价器、4更新全状态重放和真实显存测量。
- [ ] 3. 架构图、锁定规则/源码、四组正式3000及raw/EMA评价。
- [ ] 4. 配对来源与指标、必要的冻结补充日期、分层及区间。
- [ ] 5. 最终只读审阅、核验、更新研究入口并同步核对远端。

## Ledger

LOSO面板未形成清晰额外线性信号；沿用self并改变传递结构，避免完整LOSO制包。用户已授权自主设计/实验；在既有codex研究分支原位新增独立模块，不改旧科学锁。

真实4更新通过：禁用新增分支时，与原H_delta共同model/EMA、按名称的AdamW状态、scheduler及CPU/CUDA RNG精确相同。两个启用新臂readout及key梯度非零；joint训练峰值分配约1059MiB，每步10752有限查询标签。新分支与原模型回归33项通过。
