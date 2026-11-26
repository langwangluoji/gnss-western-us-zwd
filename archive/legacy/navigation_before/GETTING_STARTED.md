# 快速上手

## 先读什么

1. [项目架构](../../../docs/PROJECT_ARCHITECTURE.md)：理解目录和研究主线。
2. [当前研究进展](../../../docs/CURRENT_RESEARCH.md)：确认最新实验、数据角色和证据范围。
3. [实验索引](../../../docs/EXPERIMENT_INDEX.md)：按实验编号找到报告、协议和状态。
4. [方法验证规则](../../../docs/METHOD_VALIDATION.md)：判断一个结果能支持什么结论。
5. [本地产物说明](../../../docs/LOCAL_ARTIFACTS.md)：恢复原始数据、权重和缓存的位置。

## 最短路径

```powershell
Set-Location research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server
python -m pip install -e .
python -m pytest -q
```

当前 v13 结果入口：

`results/20261124_unet_foundation_v13/RESULTS.md`

当前代码入口：

- `tools/terrain_diffusion/unet_foundation.py`：113 维特征、网格 U-Net、训练与评价基础实现；
- `tools/terrain_diffusion/unet_hybrid.py`：点特征 MLP＋空间 U-Net 融合；
- `tools/terrain_diffusion/unet_large.py`：大宽度 U-Net 实验；
- `tools/terrain_diffusion/unet_diagnostics.py`：地形、月份、空间差值和训练末轮诊断；
- `tools/terrain_diffusion/unet_audit.py`：预测、真值、指标和权重哈希审计。

## 如何运行新实验

先复制一个已有实验协议，建立新的实验编号和独立结果目录；不要覆盖旧结果。训练前完成数据合同、输入变量、站点角色、随机种子、预算和评价日期锁定。训练后至少保存协议、状态、指标、曲线、审计清单和结果报告。大型权重、NPZ/NPY、原始数据和缓存只保留本机。

## 需要避免的误读

- 25 km 是发布网格间距，不是已经证明的空间精度；
- 2018 月首日开发评价不是全年盲测；
- 历史约 9 mm 结果使用不同输入、区域或划分，不能直接替代当前结果；
- 架构说明、网页建议和短训收益都不能单独构成方法有效性结论；
- 新研究禁止把 PWV、IWV、TCWV 或 Tm 作为输入或教师。
