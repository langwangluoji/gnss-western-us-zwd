# 实验运行手册

## 入口选择

### 当前基础场

从 `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server` 运行。v13 的主报告、指标、曲线和审计在 `results/20261124_unet_foundation_v13/`。

### 历史生成式实验

按 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) 进入 v07–v12、masked graph、sequence 和 observation transport 目录。历史实验结果只按各自协议解释，不覆盖当前 v13 基线。

## 新实验模板

1. 在 `docs/superpowers/specs/` 建立设计或协议；
2. 分配新的 `YYYYMMDD_<topic>_vNN` 目录；
3. 写入数据、输入、划分、预算和评价锁；
4. 先运行测试和 smoke；
5. 使用真实本机数据进行固定预算配对训练；
6. 保存状态、指标、曲线、预测审计和结果报告；
7. 检查 `git diff --cached --check`，确认没有 `.pt/.npz/.npy` 被提交；
8. 提交并推送，核对 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/main`。

## 当前推荐顺序

`E1401` 连续窗口与空间支撑审计 → `E1402` 点—图查询编码 → `E1403` 多尺度 U-Net 上下文 → `E1404` 点—图—U-Net 基础场 → `E1405` 折外残差诊断 → `E1406` 条件时空 diffusion。

在 E1401 和 E1405 未完成前，不直接扩大 diffusion 或用 XGBoost 作为主要精调器。

## 常用验证

```powershell
Set-Location research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server
python -m pytest -q
python -m gnss_field --help
```

每个结果报告都要回答：模型输入是什么、目标是什么、空间查询如何完成、误差如何计算、数据是否被重复使用、结论属于哪个证据等级。
