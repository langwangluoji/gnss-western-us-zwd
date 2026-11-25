# 概率条件信息直连实施计划

> **For agentic workers:** Use superpowers:executing-plans task-by-task. 用户已授权自主设计和运行。

**Goal:** 判断均值64维状态是否丢失概率校准需要的可用查询信息，并把验证后的条件视图用于时空/频域diffusion。

**Architecture:** 冻结联合Transformer及均值 → latent64/raw111/skip175 → Gaussian容量控制 → 同视图v及Gaussian预条件v。

**Tech Stack:** 现有Anaconda pytorch、NumPy及RTX4060，沿用当前工作分支与缓存。

**Spec:** [信息直连合同](../specs/2026-11-26-uncertainty-context-bypass.md)。

## Global Constraints

2016–2019连续3000块，只抽站身份；所有动态条件截至起点；禁止PWV/IWV/TCWV/Tm。原均值/模型代码和权重冻结，确认/C/D隔离。2020为已暴露开发、strict_oof=false。剩余额度75%停止新增研究。

## Review Focus

1. 新视图从已知缓存提取，不接触truth、未来mask或开发拟合统计。
2. Gaussian预条件器和score读取同一个视图、同一个源均值与同一归一化。
3. latent/64默认与原构造的参数初始化及数值完全一致，旧权重能加载。
4. 新视图和工厂实现哈希可核验，旧已完成实验不会被覆盖或重复计数。
5. 解析Gaussian与K32样本均值分开，秩区间使用有限K参考覆盖。

## Task 1：条件视图和模型工厂

Files: `tools/terrain_diffusion/native_probability_context.py`、`test_native_probability_context.py`。

- [ ] RED：缺视图/工厂时测试失败；验证concat逐值保持、query重排一致、构造默认与旧模型等同。
- [ ] GREEN：只实现latent/raw/skip及显式condition_dim/width工厂；不添加未验证天气变量。
- [ ] 核对Gaussian参数5070/8078/12174/12180及四类模型非默认条件维度的有限前向/采样。

## Task 2：训练、采样和预条件协议

Files: `native_weather_experiment.py`、`native_weather_evaluation.py`、`native_copula.py`及校准审计入口。

- [ ] RED：条件视图不一致时拒绝预条件；旧latent协议可读取。
- [ ] 接入`--condition-view`和`--width`，按块构造条件；登记来源/工厂哈希。
- [ ] 预条件器按其协议宽度构造，score按自身协议构造，条件维度必须一致。
- [ ] 全研究测试及真实缺测块小试通过后提交，再运行seed42 R111/S175/Lwide各3000块。

## Task 3：配对证据与晋级

Files: `maintenance/weather_native_20261126/`和独立`results/20261126_probability_context/`。

- [ ] 对比L64和参数匹配Lwide，复算解析/有限成员、距离/海岸/季节和联合评分。
- [ ] 有收益的视图复核seed17；再做同视图扩散与Gaussian/相关Gaussian，不将条件变化当作扩散独有收益。
- [ ] 归档表格、图件、哈希、失败原因并推送，核对远端一致和额度。
