# 原始站点序列与连续查询融合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use TDD and one fresh final review.

**Goal:** 在既有约10mm参考之上检验保留原始站点时间与方向/高差信息的编码器。

**Architecture:** 每站因果编码72小时，查询-条件站局部交叉注意力联合几何与物理创新；零初始化残差头叠加冻结GRU+p14。整个新分支无需二维格网。

**Tech Stack:** 本机Anaconda pytorch、NumPy、原p14适配器与评价。

**Spec:** `docs/superpowers/specs/2026-11-25-direct-station-query-fusion.md`

## Global Constraints

- 禁止PWV/IWV/TCWV/Tm；仅允许ZWD、T、P及几何、地形、海岸、可用性/年龄/日历。
- 连续2016–2019训练，12小时批、3000步、seed42/17；只有条件站身份抽样。
- 原2020队列，137登记站、3布局、每月72小时；原模型权重和归一化完整保留。
- C/D及确认标签隔离；原始数组及权重不入Git。

## Review Focus

- 原始历史窗口越界、批次边界或未来/目标标签进入条件。
- 高程m与km误用、重复3km输送、无有效邻站时非有限输出。
- 查询/条件站重排改变固定点结果。
- EMA漂移冻结参考，或无预先登记的检查点选择。
- 新布局和新架构被错误解释为单一模块收益。

### Task1：模型和数据合同

Files: create engine `tools/terrain_diffusion/station_query.py` and `test_station_query.py`.

- [ ] 写失败测试：原始时间窗组织、缺测mask、三种时间模块因果性、初始参考相等、查询与条件重排、全空邻站。
- [ ] 运行RED；实现最小数据函数、站点编码器与局部注意力。
- [ ] 运行GREEN及本机真实启动探针；记录输入/输出有限和参考一致。

### Task2：完整训练和归档

- [ ] 实现不可静默覆盖的协议、源哈希、训练曲线、最终raw/EMA及评价入口。
- [ ] 在Q队列完成后启动Transformer32两种子3000步，保留75/25混合站点布局。
- [ ] 用统一float64配对分析，绘制实际架构与曲线，同步轻量证据到GitHub。
- [ ] 若未优于参考或Qema，保留负结果并按条件距离/季节/高程诊断下一轮。
