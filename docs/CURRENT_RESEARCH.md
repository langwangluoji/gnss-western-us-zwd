# 当前研究进展：2026-11-23

美国西部25–50°N、125–110°W陆地域；25 km格网144×96，77个重叠600 km块。数据2016—2020，A174为唯一观测输入，B511训练、C110开发、D110回顾评价。基础工程使用2016—2017训练、2018表示/预测头；最新本机气象U-Net与创新校正使用2018抽样训练时刻，不等同于完整三年训练。

产品小时为名义时刻，UTC/GPST进一步声明以来源审计为准。采用EGM96，统一发布场查询与 `ZWD=exp(-H/3000)×Interp(Z0)`。

## 最新证据

路径相对工程 `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server`。

| 内容 | 路径 |
|---|---|
| 最新确定性校正 | `results/20261123_history_guided_refinement_v01/RESULTS.md` |
| 同样本144小时确认 | `results/20261123_history_guided_refinement_v01/confirmation_v02/results.json` |
| 新旧候选配对 | `results/20261123_history_guided_refinement_v01/champion_comparison.json` |
| 气象U-Net双种子 | `results/20261123_simple_weather_unet_v02/RESULTS.md` |
| GAN/Diffusion/VAE短训 | `results/20261123_enhanced_encoder_generative_v03/RESULTS.md` |
| 生成机制验证 | `results/20261123_generator_mechanism_validation_v01/VERDICT.md` |
| 同96小时历史插值对照 | `results/20261122_baseline_best_comparison_v01/comparison.json` |

## 最新同样本点精度

来源：`confirmation_v02/results.json` 的 `matched_historical`。C2019每月8—10日每6小时，共144小时；共同76站、10610站时。该开发集已历史暴露，D未读取。

| 方法 | 站宏RMSE/mm | 全记录RMSE/mm |
|---|---:|---:|
| 粗格网双线性 | 13.9015 | 16.9925 |
| A站直接IDW格网查询 | 10.4585 | 11.5618 |
| 原优选气象U-Net wide17 | 9.1106 | 9.5169 |
| wide42＋几何创新校正候选 | 9.0658 | 9.6579 |

原优选相对IDW的站宏RMSE降低约12.9%，新候选约13.3%。新候选相对原优选差−0.04482 mm，95%配对区间[−0.19577, 0.10557]；微RMSE较高，因此保留原优选产品。上述成绩是本机抽样训练与开发回看，不能据此声明全年、未见年份或真实细尺度恢复能力。

四个冻结起点各自在B内层选择读出后，C144回看均改善；这支持继续验证该流程，不表示同一固定配置已跨种子复现。最新报告的下一步是锁定新旧候选与评分人口，完成完整2019配对评价和同步7日块bootstrap。

## 整理复盘

混乱来自多代研究共用目录、外层README停留在旧区域、最新结果嵌在工程results内，以及不同实验重复使用B0/M2编号。今后以“完整实验目录＋锁文件＋评价清单”定位结果，报告同时写明区域、年份、站点数、站时数及宏/微指标。

当前区域的负结果与早期扩区证据保留。旧区域移出项目，可重建缓存删除不改变已有指标、科学源码或模型权重。
