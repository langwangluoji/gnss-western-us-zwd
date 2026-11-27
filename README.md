# Sparse GNSS ZWD Field Reconstruction

稀疏 GNSS 观测下的天顶湿延迟空间重建、时间编码与联合概率场。美国西部是主要研究区域，欧洲是独立训练的区域验证；VMF3背景扩展单独评价。

## 从这里开始

| 入口 | 说明 |
|---|---|
| [当前成果与最佳结果](paper/README.md) | 现有最佳模型、数据范围和结论 |
| [模型各部分的贡献](paper/MODULE_CONTRIBUTIONS.md) | 模块作用、配对证据和待补对照 |
| [架构与流程图](paper/ARCHITECTURE.md) | 空间编码、时间融合、创新和概率生成 |
| [全部实验索引](experiments/README.md) | 每个实验的原理/结果入口、状态依据与来源 |
| [快速上手](docs/GETTING_STARTED.md) | 根目录安装、测试和兼容路径 |
| [本地数据与权重](artifacts/README.md) | 大文件位置和恢复边界 |
| [数据精简与恢复](maintenance/compact_20261127/README.md) | 核心保留、退役清单、云端模型与核验 |
| [目录迁移记录](maintenance/paper_restructure_20261126/README.md) | 上轮迁移、哈希与验证 |

## 目前的主要结果

| 区域与任务 | 模型 | 指标 | 评价范围 |
|---|---|---:|---|
| 美国当前空间重建 | p14＋GRU＋I1 | 站宏RMSE **10.319804 mm** | 两组2020抽样日历合并，已开发暴露 |
| 美国6小时预报 | 有向时滞Transformer joint v2 | 站宏RMSE **15.053927 mm** | 相同两组日历合并，已开发暴露 |
| 美国联合概率场 | joint v2＋天气GP＋Local | F6 CRPS **7.657557 / 8.750280 mm** | 主/补充日历、raw K32，独立于点RMSE |
| 欧洲当前重建 | p14＋GRU＋I1＋VMF3校正 | micro RMSE **11.406422 mm** | 2024完整年度回看，59有效确认站、63494共同键 |

美国主日历9.806169 mm是joint v2的当前场站宏RMSE，不是完整年度结果。欧洲相对原I1的15.927922 mm降低28.39%；该点场收益包含VMF3外部物理信息扩充。扩散的额外收益按Gaussian/Affine对照单独报告。不同区域、人口和汇总指标分别解释。

## 目录

```text
src/gnss_field/     可安装的核心工程包
research/engine/   研究工具、测试、历史配置与运行入口
research/*/        欧洲、融合、概率与频域研究代码及说明
experiments/       美国/欧洲轻量证据与逐实验说明
paper/             当前结论、模块贡献和架构入口
docs/              数据合同、手册与有日期的研究记录
artifacts/         本地数据、权重、缓存和冻结运行记录
archive/           历史兼容源与迁移前导航
maintenance/       本次整理脚本和核验回执
```

Git保存代码、协议、指标与图件；原始观测、核心权重及其数据依赖留本地。退役模型和拟合参数保存到GitHub Release，非核心大数组按[精简清单](maintenance/compact_20261127/README.md)退役。历史实验锁不改写。安装与复现见快速上手；历史研究版本固定在标签`archive/pre-paper-restructure-20261126`。
