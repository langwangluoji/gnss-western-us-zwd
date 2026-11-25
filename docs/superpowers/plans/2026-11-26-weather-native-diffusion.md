# 天气与原生站点时空扩散实施计划

> **For agentic workers:** Use superpowers:executing-plans; user has authorized autonomous implementation and experiments.

**Goal:** 在I1基础上检验长历史天气条件及无标签投影损失的时空频域扩散，直到形成可复现收益或额度达到停止线。

**Architecture:** 冻结I1→连续缓存→条件均值/尺度→空间或完整谱v扩散；R和F分别评分。

**Tech Stack:** Anaconda pytorch、NumPy、SciPy、Matplotlib，RTX4060 8GB。

**Spec:** [数据、架构及晋级合同](../specs/2026-11-26-weather-native-diffusion-design.md)。

## Global Constraints

连续2016–2019、3000块/更新；只抽样站身份。源权重冻结。禁止水汽柱变量，确认/C/D隔离。2020为开发评价。当前两阶段缓存非严格OOF，正式概率晋级需另建OOF来源。账户剩余75%停止新增研究。

## Review Focus

1. 未来1–6h真值不进入任何条件；窗口不越训练边界。
2. 条件与目标物理组隔离，完整历史固定条件身份。
3. 图谱及DCT全系数，双域共享同一状态/噪声，逆变换回mm。
4. 训练统计标准化；v逆变换不除极小alpha。
5. 每成员时间身份固定，fair CRPS与有限成员均值误差分开；查询人口改变单列诊断。

## Task 1：可测试算子与连续缓存

Files: `tools/terrain_diffusion/native_weather.py`、`native_weather_cache.py`、`test_native_weather.py`。

- [ ] RED：可逆DCT/GFT、缺测加权天气、历史因果和split边界测试。
- [ ] GREEN：原生查询天气/创新/趋势编码，完整正交变换及图核。
- [ ] 生成可恢复缓存，首先真实2块小试，再3000连续块、源seed42/17；核对I1保存预测和未来信息上界。

## Task 2：配对均值与概率模型

Files: `native_weather_models.py`、`native_weather_experiment.py`。

- [ ] RED/GREEN：零初始化均值、v旋转恢复、采样有限值和图重排一致性。
- [ ] 逐站模型误差＋原始ZWD/T/P＋mask/age＋身份/地形/海岸；GRU/LSTM/因果Transformer在同一72h历史、宽64和地形查询注意力下比较，各3000更新；24h作为历史范围控制。
- [ ] W0/W1/W2及Gaussian/空域v/图谱v/双频v，同缓存与最终3000步。冻结主候选后补充日历。
- [ ] 根据拟合、泛化、尺度和条件敏感性实测，记录优化理由再继续。

## Task 3：评估、图件与归档

Files: `maintenance/weather_native_20261126/`及新结果目录。

- [ ] 独立float64复算R/F、配对月份区间、fair CRPS/覆盖、站对差值、小时增量及六类分层。
- [ ] 架构PNG/SVG、学习曲线、逐月/区域/站点分析；记录负结果和剩余工作。
- [ ] 检查代码/协议/数据/权重哈希、轻量提交和推送一致；阶段间复核额度。
