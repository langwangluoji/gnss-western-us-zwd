# 架构定向检索与原始来源

检索日期：2026-11-21（Asia/Shanghai）。类型：面向本项目的定向叙述性调研；不是穷尽式系统综述。检索入口为网络学术检索，原文核验入口包括 PMLR、GMD、NeurIPS、arXiv、JMLR、AGU 和作者代码仓库。预印本与正式期刊版按同一研究处理。

## 检索问题

稀疏、不规则 GNSS 条件与真实站监督下，哪种空间生成骨干最适合生成统一格网场，并支撑 Diffusion/GAN 对照？分别核查点到格网编码、空间处理器、联合采样、目标来源与计算成本。

## 已使用检索式

- convolutional conditional neural processes climate downscaling 2022
- Gridded Transformer Neural Processes 2025
- GINO Geometry Informed Neural Operator 2023
- Neural Diffusion Processes 2023
- "diffusion" "sparse observations" "neural processes" 2025 2026
- "Diffusion" "weather stations" downscaling neural process
- "Convolutional Conditional Neural Processes" 1910.13556 density
- "Ambient Diffusion" fixed mask identifiability observations
- "Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales"
- site.openaccess.thecvf.com SR3 Image Super-Resolution via Iterative Refinement Saharia 2021
- site.openaccess.thecvf.com SwinIR Image Restoration Using Swin Transformer 2021

## 核心原始资料

| 资料 | 原始来源 | 本项目用途 |
|---|---|---|
| Gordon et al., Convolutional Conditional Neural Processes, ICLR 2020 | https://arxiv.org/html/1910.13556v5 | 值/密度点到格网编码；基础模型的条件独立输出边界，Section 6 |
| Vaughan et al., Convolutional conditional neural processes for local climate downscaling, GMD 2022 | https://gmd.copernicus.org/articles/15/251/2022/ | 真实站监督、地形与未见站评价；其输入有 ERA-Interim，区别于 A-IDW |
| Ashman et al., Gridded Transformer Neural Processes for Spatio-Temporal Data, ICML 2025 | https://proceedings.mlr.press/v267/ashman25a.html | 不规则点/格网融合；小型 Swin 备选 |
| 同研究全文（预印本标题带 Large Unstructured） | https://arxiv.org/html/2410.06731v2 | Section 2.1 的边缘分布假设；Table 4 的参数/耗时；不是联合场生成模型 |
| 作者 Gridded-TNP 代码 | https://github.com/cambridge-mlg/gridded-tnp | 仓库确有 ConvCNP/SetConv 与 Swin-TNP 配置，未安装或运行 |
| Li et al., Geometry-Informed Neural Operator for Large-Scale 3D PDEs, NeurIPS 2023 | https://arxiv.org/html/2309.00583v1 | 图算子处理不规则几何、FNO 处理潜在格网；原任务为模拟 PDE 解算 |
| Dutordoir et al., Neural Diffusion Processes, ICML 2023 | https://proceedings.mlr.press/v202/dutordoir23a.html | 函数有限维边缘上的扩散；Section 5 明确一般边缘一致性无保证 |
| NDP 正文 PDF | https://proceedings.mlr.press/v202/dutordoir23a/dutordoir23a.pdf | 第 5 页与附录 E 的查询集合一致性边界 |
| Mardani et al., CorrDiff | https://arxiv.org/html/2309.15214v4 | U-Net 均值＋U-Net 残差去噪；ERA5→WRF 稠密监督，不能直接视为稀疏站证据 |
| Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models, NeurIPS 2022 | https://arxiv.org/abs/2206.00364 | EDM 的网络预条件、训练和采样分别核查 |
| Ravuri et al., Skillful Precipitation Nowcasting using Deep Generative Models of Radar, Nature 2021 | https://arxiv.org/abs/2104.00954 | DGMR 的空间连贯随机生成；原任务有稠密雷达序列，非稀疏 ZWD 重建 |
| Pacchiardi et al., Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization, JMLR 2024 | https://jmlr.org/papers/v25/23-0038.html | 支持生成网络评分规则训练；不提供固定 GNSS 测量下的全场可辨识保证 |
| 对应作者代码 | https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting | 已核实仓库归属，未安装或运行 |
| Manshausen et al., Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales, JAMES 2025 | https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004505 | HRRR 完整场先验训练＋稀疏站推理同化；不是仅靠稀疏站训练先验 |
| Daras et al., Ambient Diffusion, NeurIPS 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/file/012af729c5d14d279581fc8a5db975a1-Paper-Conference.pdf | 部分观测分布学习的测量条件；不能由论文标题推定固定站网可辨识 |
| Saharia et al., SR3, Image Super-Resolution via Iterative Refinement | https://research.google/pubs/image-super-resolution-via-iterative-refinement/ | 作者页面确认采用 U-Net 迭代去噪；照片观感与 GNSS 定量场真实性是不同目标 |
| Liang et al., SwinIR, ICCV Workshops 2021 | https://arxiv.org/html/2108.10257 | 残差 Swin 块与窗口交互；Section 3.1 的图像超分监督使用对应 HR 图像，非稀疏站标签 |

## 补充线索与检索限制

CVPR 2025 的 Satellite Observations Guided Diffusion Model for Accurate Meteorological States at Arbitrary Resolution 检索摘要显示使用 ERA5 预训练及 GridSat 条件；原站点全文请求返回 403。本轮不依赖未读全文作核心架构裁决。

Gridded-TNP 的 PMLR PDF 下载由检索工具识别为不支持的 application/octet-stream，已改读作者 arXiv HTML；NDP 则成功读取 PMLR PDF。SwinIR 的 CVF PDF 返回 403，已改读作者 arXiv HTML。未以搜索摘要替代上述可取得的正文。

## 本地核对来源

- `docs/superpowers/specs/2026-11-21-grid-residual-diffusion-gan-architecture-v02.md`：建议稿，未实现、未训练。
- `results/20261120_local_context_three_arm_v01/RESULTS.md`：三臂两折结果及评价边界。
- `src/gnss_field/models/subspace.py`：实际小型系数 EDM 与 Heun 调用路径。

Pro 网页意见作为另一次研究审阅记录，事实依据仍由原始资料及本地代码/结果支撑。
