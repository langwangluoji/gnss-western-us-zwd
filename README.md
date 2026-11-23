# 美国西部 GNSS 湿延迟场重建

当前研究范围为 **25–50°N、125–110°W陆地域**，数据覆盖2016—2020年。发布格网为25 km、144×96；有效空间分辨能力通过未见站点及站间差值评价。

代码与轻量研究证据同步至私有仓库 [langwangluoji/gnss-western-us-zwd](https://github.com/langwangluoji/gnss-western-us-zwd)。Windows长路径检出建议使用 `git -c core.longpaths=true clone`。

当前工程位于 `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server`。工作区中的20261105是历史名称，不代表当前仍使用2008—2011年的南加州小区域。

| 入口 | 内容 |
|---|---|
| [当前研究进展](docs/CURRENT_RESEARCH.md) | 区域、数据角色、最新结果与证据范围 |
| [实验索引](docs/EXPERIMENT_INDEX.md) | 当前区域实验、状态和报告 |
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

## 运行

需要Python≥3.11、PyTorch 2.8.x及工程声明的依赖。从项目根目录：

```powershell
Set-Location research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server
python -m pip install -e .
python -m gnss_field --help
python -m pytest -q
```

新机器先按[数据说明](docs/LOCAL_ARTIFACTS.md)恢复本地数据。Git克隆包含代码、配置、协议、轻量指标和报告；原始观测、权重、预测数组和缓存另行保留。
