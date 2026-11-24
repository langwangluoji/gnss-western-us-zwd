# E01–E05 时空地形残差扩散 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在统一 p14 formal protocol 下逐步验证地形编码、显式边消息、因果时间条件、增量损失和基础场残差扩散。

**Architecture:** 复用现有 terrain_diffusion 数据与评价器，新增一个最小配置驱动入口；每个实验只打开一个机制，E05 冻结基础场并预测 OOF 残差。所有结果按实验目录保存轻量 metrics/report。

**Tech Stack:** Python, PyTorch, 现有 gnss_field_server terrain_diffusion 模块。

**Spec:** docs/superpowers/specs/2026-11-24-e01-e05-spatiotemporal-residual-diffusion-design.md

## Global Constraints

- 使用 410/137 站点合同、p14 protocol/hash、3 layouts 和 12×24 小时扩展评价。
- 禁止 PWV/IWV/TCWV；C/D 标签不读取；权重和大型数组不提交 Git。
- E01–E04 同时跑 random 控制和 hole100；E05 跑 hole100；seed42/17 固定。

## Review Focus

- protocol/hash 是否与 p14 一致：由合同审计测试覆盖。
- 时间条件是否只读取 t-1/t-2：由因果输入测试覆盖。
- OOF 残差是否避免同样本泄漏：由残差来源审计覆盖。
- 缺测掩码和增量配对：由损失单元测试覆盖。
- 正式指标是否区分 monitor 与 expanded：由报告审计覆盖。

### Task 1: 合同与配置入口

**Files:**
- Create: `tools/terrain_diffusion/e01_e05_experiment.py`
- Test: `tools/terrain_diffusion/test_e01_e05.py`

- [ ] 实现 `ExperimentConfig` 和 `run_experiment(config)`，固定 protocol、seed、采样臂及机制开关。
- [ ] 测试配置拒绝 PWV 类输入、确认 C/D 未加载，并输出 protocol/hash。
- [ ] 运行 smoke：`python -m tools.terrain_diffusion.e01_e05_experiment --experiment E01 --seconds 5`。

### Task 2: E01 地形编码

**Files:** Modify `e01_e05_experiment.py`; Create `results/20261124_E01_terrain/REPORT.md`。

- [ ] 加入训练集拟合的高程、坡度、海岸距离编码和缺失掩码。
- [ ] 运行 seed42/17 的 random/hole100 smoke 与短训，生成统一 metrics。
- [ ] 核验宏 RMSE、分层 RMSE 和小时增量指标。

### Task 3: E02–E04 空间与时间机制

**Files:** Modify `e01_e05_experiment.py`; Create E02/E03/E04 result reports。

- [ ] E02 固定 k=8 边消息与距离/方位/高差特征。
- [ ] E03 单层 GRU(hidden=64)，只读取 t-1/t-2。
- [ ] E04 加入 lambda_delta=0.2 的有效相邻时刻 Smooth-L1。
- [ ] 每个实验完成 smoke、短训、两 seed 评价并写入 report。

### Task 4: E05 基础场残差扩散

**Files:** Modify `e01_e05_experiment.py`; Create E05 result report。

- [ ] 冻结基础场，使用训练折内 OOF 预测生成残差条件。
- [ ] 保持扩散采样和 p14 formal 评价预算不变。
- [ ] 运行两 seed，比较宏 RMSE、CRPS、分层误差和增量指标。

### Task 5: 汇总、图表与同步

**Files:** Create `results/20261124_E01_E05_summary/REPORT.md` and comparison plot.

- [ ] 汇总每实验逐 seed/均值/布局指标与失败状态。
- [ ] 运行 `git diff --check`、测试和最小审计。
- [ ] 提交轻量代码、配置、报告和图表，推送 main，核对远端 HEAD。
