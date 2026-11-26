# 美国西部 GNSS 湿延迟场重建

本项目研究稀疏 GNSS 网下的高分辨率 ZWD（天顶湿延迟）场重建，目标是在不使用 PWV/IWV/TCWV/Tm 条件或教师的前提下，利用 GNSS 站点气象、ZWD 历史、地形和空间支撑信息，输出任意查询点的连续 ZWD 场与时序样本。

当前区域为美国西部 **25–50°N、125–110°W 陆地域**，数据覆盖 2016—2020 年。25 km、144×96 是发布与区域上下文网格，不代表最终细节分辨率；细尺度能力通过未见站点、站间差值、空间空洞和时间增量评价。

代码与轻量研究证据同步至公开仓库 [langwangluoji/gnss-western-us-zwd](https://github.com/langwangluoji/gnss-western-us-zwd)。Windows长路径检出建议使用 `git -c core.longpaths=true clone`。

当前工程位于 `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server`。工作区中的20261105是历史名称，不代表当前仍使用2008—2011年的南加州小区域。

| 入口 | 内容 |
|---|---|
| [当前全部研究成果总览](docs/research_notes/2026-11-26-research-achievements-overview.md) | 数据体系、当前最佳模型与精度、概率/扩散/频域成果、全部路线和下一步接口 |
| [最佳基础场＋残差扩散融合实测](docs/research_notes/2026-11-26-best-base-diffusion-fusion.md) | joint v2载体、天气相关过程与Local扩散；8模型3000块、24组评价，完整架构与曲线 |
| [阶段研究总报告与最佳架构](docs/research_notes/2026-11-26-final-research-summary.md) | 全部已完成工作、当前重建与六小时预报精度、扩散/频域归因、PNG/SVG架构；末轮完成后停止新增实验 |
| [下一阶段计划与审阅](docs/NEXT_STAGE.md) | 统一阅读入口、因果边界、尺度公式、阶段出口和待补项 |
| [当前研究进展](docs/CURRENT_RESEARCH.md) | 区域、数据角色、最新结果与证据范围 |
| [实验索引](docs/EXPERIMENT_INDEX.md) | 当前区域实验、状态和报告 |
| [首轮时空预报实测](research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_temporal_forecast_screen/REPORT.md) | 26配置、GRU/LSTM/因果Transformer与12/48/72h历史对照 |
| [快速上手](docs/GETTING_STARTED.md) | 新用户或其他 agent 的最短复现路径 |
| [项目架构](docs/PROJECT_ARCHITECTURE.md) | 研究层、数据层、模型层、评价层和产物边界 |
| [方法验证规则](docs/METHOD_VALIDATION.md) | 结论等级、数据隔离和验收门槛 |
| [实验运行手册](docs/EXPERIMENT_RUNBOOK.md) | 入口脚本、测试、轻量实验和结果记录规范 |
| [优化路线](docs/OPTIMIZATION_ROADMAP.md) | 每个实验结束后的下一步、分支判断和验收门槛 |
| [工程说明](research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/README.md) | PyTorch实现、本机运行与超算流程 |
| [数据与权重说明](docs/LOCAL_ARTIFACTS.md) | 本地大文件、清单及Git边界 |
| [整理记录](maintenance/cleanup_20261123/README.md) | 归档、删除、验证与复盘 |
| [方法验证要求](research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/docs/METHOD_VALIDATION.md) | 真实环境验证及结论等级 |

## 目录

```text
docs/                       当前进展、实验索引、数据说明与整理设计
maintenance/                整理清单、脚本与验证记录
research_pipeline/
  data/pwv_data/            西部区域候选站原始压缩观测（本地）
  data/external/            EGM96高程基准（本地）
  scripts/、src/            数据解析所需上游支持代码
  studies/
    us_dense_zwd_20261105_plan/evidence/countries.geojson
    us_dense_zwd_20261105_v1/
      01_inventory/         原始站点和地形来源
      02_data/              最新区域数据与SRTM（本地）
      03_code/gnss_field_server/
        src/、tools/、tests/、configs/、docs/
        results/            当前区域各轮证据、负结果和权重
      03_code/              数据重建所需上游代码
      04_models/field_generation_v1/  扩区初始实验与划分修正记录
      06_reports/           覆盖审计与空间评价协议
```

旧区域研究及历史汇报移至项目外同盘目录 `../GAN对流层_历史归档_20261123/`，原相对路径保留。当前工程依赖的少量早期代码作为上游支持留在原路径，不能作为最新排名依据。

## 研究主线

当前最强确定性基础场候选是 **点特征 MLP＋小型 U-Net 空间修正**（v13，双种子平均宏 RMSE 10.828 mm）。大宽度 U-Net 48/96/192 在同口径开发评价中为 11.800 mm，说明当前瓶颈是空间信息表达、天气日期覆盖和空间支撑分布，而不是单纯增加宽度。

后续路线按以下顺序推进：连续天气窗口审计 → 点—图—查询编码 → 多尺度 U-Net 区域上下文 → 折外基础场残差 → 条件时空 diffusion。所有新方法先做真实本机配对试验，再讨论全量训练或论文结论。

## 运行

需要Python≥3.11、PyTorch 2.8.x及工程声明的依赖。从项目根目录：

```powershell
Set-Location research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server
python -m pip install -e .
python -m gnss_field --help
python -m pytest -q
```

新机器先按[快速上手](docs/GETTING_STARTED.md)和[数据与本地产物说明](docs/LOCAL_ARTIFACTS.md)恢复环境。Git 克隆包含代码、配置、协议、轻量指标和报告；原始观测、权重、预测数组和缓存另行保留。
