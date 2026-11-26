# Centered NonGaussian Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. 用户已授权自主设计和运行；本计划承接已归档路线，不重复请求许可。

**Goal:** 分离基础场中心、概率尺度与非Gaussian形状的增益，给后续diffusion提供强对照。

**Architecture:** c265条件输出逐lead尺度和归一化Hermite系数；冻结G265耦合Gaussian初值；解析均值始终等于基础场。

**Tech Stack:** 现有Windows/pytorch环境、NumPy、Matplotlib。

**Spec:** [参数、数学与数据合同](../specs/2026-11-26-centered-nongaussian-control.md)。

## Global Constraints

- 独立新模块，不改在跑GP/Local/Affine源码、科学锁或旧结果。
- 三臂各3000连续块、K16训练同噪声流；K32推断137查询。
- 固定mu_base，尺度来自同S175；c265与新G265源一致。
- 2020开发、确认/C/D隔离、禁止PWV类变量；75%预算停止线与轻量Git同步。

## Review Focus

1. Gaussian矩正交与归一化正确，均值不依赖网络参数或成员数。
2. CRPS成员对项K(K-1)，NaN监督不污染预测或梯度。
3. 形状参数零初始化仍有梯度；unit variance误作标准差缩放不得发生。
4. ScaleOnly解析Gaussian、非Gaussian仅解析矩；K32覆盖与均值误差分开。
5. 查询条件/天气核不读取监督mask；完成seal与计分来源可追溯。

## Task1：数学与模型

**Files:** `tools/terrain_diffusion/native_centered_shape.py`、`test_native_centered_shape.py`。
**Interfaces:** `CenteredShape(mode).moments(c)`、`transform(c,eta)`、`fair_crps_torch(samples,truth)`、`masked_loss(model,target,c,gs,members=16)`。

- [ ] RED：解析Gaussian矩/归一化、零head样本继承、active子集一致、fair CRPS手算、缺测与梯度。
- [ ] GREEN：三臂近等参数实现；CPU和研究模块测试。
- [ ] 核验后提交源码与数学记录。

## Task2：独立训练与推断

**Files:** `native_centered_shape_experiment.py`、相应test、`maintenance/weather_native_20261126/centered_shape_campaign.py`。
**Interfaces:** 复用WeatherSources及训练guard；新codes绑定native_weather_evaluation.py；`--phase train/reference/evaluate`。

- [ ] RED：开发训练拒绝、非G265/未完成源拒绝、错误完成权重拒绝。
- [ ] GREEN：连续3000、源哈希、checkpoint/seal、真实4更新/137查询warm probe。
- [ ] 小试核验后提交，G265完成后双种子三臂各3000及完整K32采样。

## Task3：收益与归档

**Files:** `maintenance/weather_native_20261126/centered_shape_summary.py`及新results目录。

- [ ] 逐值核验解析均值=mu_base，标准差按归一化理论；ScaleOnly解析CRPS复算。
- [ ] 三项消融的月份配对、边际/站对/增量/energy/事件诊断，不把固定均值当新增精度收益。
- [ ] 更新架构图、结果入口，提交推送并核对远端。

## 当前执行状态

Task1数学/模型与Task2工程实现、真实缺测小试已完成；三臂各4实际更新、18088有限标签，初始化Gaussian样本最大差0。独立审阅补强工件recipe及joint来源守卫，RED→GREEN；未修改已锁定GP或Local/Affine。正式三臂在G265及预注册轨迹diffusion完成后顺序运行，避免同时增加GPU压力。
