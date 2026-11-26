# 最佳基础场与扩散融合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 使用executing-plans逐项执行；结束时进行一次独立只读复核。

**Goal:** 在冻结joint v2上重新拟合概率来源，实测同预算残差扩散与均值保持融合。

**Architecture:** 新joint嵌入/残差 → S175 → copula → G265 → Affine/Local。raw K32与独立双场差分K16同时输出；确定性点预测保持joint。

**Tech Stack:** 本机Anaconda pytorch环境，既有PyTorch模块、NumPy、Matplotlib；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-26-best-base-diffusion-fusion.md`。

## Global Constraints

保留旧科学锁与结果；2016–2019连续训练，开发两日历均完整报告；同种子joint/raw final3000；每臂3000块；只用新训练统计归一化；禁水汽柱变量；strict_oof=false；确认/C/D隔离；原数据/权重不进Git。

## Review Focus

- 错种子、错误均值版本或旧TF概率来源必须被拒绝。
- 开发嵌入必须与训练嵌入的joint身份相同；开发统计不能用于训练。
- 已存在的完成文件必须核验来源，不能仅凭文件名复用。
- 差分成员对不得重叠；NaN只按实际标签进入评分，不能跨日期拼增量。
- 新训练/评价来源与科学锁一致，空标签监督步数正确；概率中心与点输出分开。

## Task 1: 数据与来源

Files: 新建`maintenance/joint_fusion_20261126/campaign.py`、`test_contract.py`、`progress.md`、`lock.json`；结果目录`results/20261126_best_base_diffusion_fusion`。

- [x] RED：来源验证测试覆盖joint/last/同种子、开发角色、归一化与完成来源拒绝。
- [x] GREEN：实现最小注册/复用守卫；复用既有joint exporter生成双种子训练嵌入，开发嵌入复用已封存导出。
- [x] 核验新训练统计与数组、原joint开发点场一致；Gaussian/GP/两传输真实四更新通过，上游冻结依赖顺序见台账。
- [x] 冻结协议/来源/执行依赖，140bd62提交推送并核对远端后启动正式训练。

## Task 2: 配对训练与评价

Interfaces: 既有`native_weather_experiment`、`native_copula`、`native_weather_gp_experiment`、`native_weather_gp_diffusion_experiment`。

- [x] 每种子依次新拟合S175、copula、G265，再拟合Affine/Local，各神经臂3000块。
- [x] 原始两日历K32评价及joint评分，核验每个模型最终checkpoint和真实监督数。
- [x] RED→GREEN：独立成员差分保持均值/协方差与不重叠规则；输出K16匹配对照。

## Task 3: 归因与交付

Files: `maintenance/joint_fusion_20261126/summary.py`；研究报告与PNG/SVG图；更新当前入口。

- [x] 检查真实样本、哈希及点场，计算主/补配对区间、事件、分层和均值误差。
- [x] 绘制融合架构、训练曲线与性能图，实际查看。
- [x] 独立只读终审及交付核验：8模型、24输出、12派生成员逐值重算、2000共同月份区间通过。

最终交付动作：将本次已核验的报告、图与轻量结果提交推送，实时对照远端HEAD；提交号与同步结论以执行会话最终回报为准。结果入口为[正式融合报告](../../research_notes/2026-11-26-best-base-diffusion-fusion.md)。本轮不追加训练。
