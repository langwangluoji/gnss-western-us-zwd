# 本地数据、权重与恢复

Git仓库同步源码、配置、测试、报告、科学锁、轻量指标和部分图件。原始观测、地形栅格、训练权重、大型预测数组和数据缓存保留在本机，克隆仓库不等于取得这些大文件。

## 数据位置

- `research_pipeline/data/pwv_data/`：当前美国西部边界内候选站的原始 `.pwv.gz`，保留原始全时段资料，不截断观测。分析数据仅按2016—2020协议提取。其他区域原始文件已外移归档。
- `research_pipeline/data/external/egm96_15.gtx` 与 `us_nga_egm96_15.tif`：高程基准。
- `research_pipeline/studies/us_dense_zwd_20261105_plan/evidence/countries.geojson`：当前陆地域定义的上游依赖。
- `research_pipeline/studies/us_dense_zwd_20261105_v1/02_data/raw/srtm/`：官方SRTM下载。
- 该研究的 `02_data/processed/field_generation_v1/`：站点、几何、角色及原始提取缓存。
- 该研究的 `02_data/processed/gnss_field_server_v1/`：2016—2020、A/B/C/D角色月片。`manifest.json`记录每个数组的形状和SHA256；Git保留清单，本地保留数组。

## 权重与预测

工程 `results/` 各实验内保存训练权重、选择锁、预测数组和指标。三个真实工程验收目录 `build/full_generative_real_check_v01/v02/v03` 原位保留。权重索引见 `maintenance/cleanup_20261123/local_weights_manifest.json`；本次整理保护文件的SHA256见 `protected_artifacts_before.json`。

迁移时应复制上述数据目录和所需实验目录，保持根目录相对结构，核验manifest和权重哈希。历史锁可能记录原Windows绝对路径；复制后先核对身份并显式登记新运行，不能静默改写旧锁。`configs/local.yaml`的数据相对位置保持可用。上游解析代码依赖h5py、rasterio、shapely等额外包；日常当前工程依赖由pyproject.toml定义。

保留的 `residual_edm_v1` 包提供当前需要的模型/工具支持；其 `early_mean.py` 等旧训练入口仍需从归档恢复旧数据与额外模块后才能运行。`studies/europe_dense_zwd_20261104_v1` 只保留一致性测试所需两个源文件，不保留欧洲实验进展。

## 已删除的可重建缓存

只删除了完整特征缓存及Python/pytest临时缓存。原始数据、来源权重、实验锁、预测和指标均保留。两组大缓存的原元数据保存在本机 `maintenance/cleanup_20261123/deleted_cache_metadata`。

1. `results/20261121_grid_processor_pair_v01/scientific/cache`：约31.60 GiB。由 `tools/grid_pair_data.py` 中 `GridStudy.prepare()` 及 `tools/grid_pair_train.py` 使用的入口重建。重建会重新计算冻结输入特征；先查看该实验锁和数据身份，不能把重建当成重新训练授权。
2. `results/20261122_local_generative_architecture_v01/cache`：约3.24 GiB。由同实验 `pilot_data.py` 的 `prepare()` 重建，依赖保留的来源数据和其他缓存。

缓存游标与数组一起删除，避免把空数组当成已完成缓存。调用旧实验脚本前必须先恢复所需缓存；已有指标的阅读和当前最新U-Net/校正不依赖这两个已删除缓存。

## 历史归档

同盘项目外 `../GAN对流层_历史归档_20261123/` 保存旧区域、旧模型、历史汇报、非当前区域原始数据及打包副本。归档按原项目相对路径组织；恢复前查看操作清单，不覆盖当前研究。
