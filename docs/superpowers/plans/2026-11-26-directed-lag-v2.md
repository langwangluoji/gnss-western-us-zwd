# Directed lag v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, inline TDD and final read-only review.

**Goal:** 修复异步缺测归一化，完成同预算有向时滞正式比较。

**Architecture:** 独立v2覆写变量池化；H载体、参数顺序及两新臂结构不变，从原来源重训。

**Tech Stack:** 本机Conda pytorch、RTX4060、pytest、NumPy、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-11-26-directed-lag-v2.md`及其引用的原架构合同。

## Global Constraints

原源码/锁/中间结果保留，v1不复用为v2权重；双种子每臂完整3000，连续时间合同及原data限制保持，raw主EMA并列，确认/C/D隔离。

## Review Focus

远GNSS与近温压缺测解耦；空支持有限梯度；继承调用实际走v2池化；参数创建/旧H等价；新来源锁和结果目录不混用。

## Tasks

- [x] 1. 复现v1异步缺测FAIL，停止并保存原中间结果。
- [x] 2. 独立v2变量归一化、warm/初始化/不变性及真实4更新重放。
- [ ] 3. 审阅修复、冻结v2源码/规则并运行四组3000。
- [ ] 4. 配对来源/指标/分层及冻结补充日期。
- [ ] 5. 完成最终审阅、核验及轻量同步。

## Ledger

v1 factorized两种子实际1532/1530步工程停止，未跑满3000、不参与正式排名。回归预期0.964538mm，v1实测0.004579mm，v2修正后通过。9项新旧模型测试已通过；v2真实4更新同源重放完成，joint峰值1504MiB，因果和数据角色合同不变。
