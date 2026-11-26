# 20261122_full_generative_design_v01

当前公开来源：`experiments/us/20261122_full_generative_design_v01`；[来源目录](../../us/20261122_full_generative_design_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_full_generative_design_v01`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第1行：# 完整条件VAE、GAN与Diffusion：架构及全量训练协议

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第72行：完整模型优化阶段采用真实B站观测判别器：每站32维条件、残差及XYZ进入36→64→64编码；掩码均值/方差128维，加四个距离箱各64维站对特征，共384维，经384→128→1输出。每箱按固定物理顺序取最多256对。观测判别器62,465参数。对抗项与ES的生成器梯度尺度在首步及每100步记录；对抗项超过ES梯度10倍时停止，保留诊断，不自动改变锁定权重。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第100行：单张A800串行作业，总额度24卡时。先做真实冷月数据准备、全部目标、35次采样反传、写盘和K16/K128评价短测；128维配置用于保守定价。硬件短测累计不超过1卡时；缓存准备与B训练内部诊断分别占用已登记的科学阶段预算，全部计入24卡时总额。包含全量训练、缓存、开发评价、全年候选复排的保守预算超限则停止，不自动缩减科学训练或扩大额度。
- [experiments/us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md](../../us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md)，第7行：- `runtime.staged_preparation=true` orders `prepare → diagnostics → control → vae_representation → posterior_cache → prior/shared/complete stages → final_evaluation`. Its default remains false for existing runs.
- [experiments/us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md](../../us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md)，第8行：- `run_full_generative_preflight.py` supports the staged benchmark-only path: benchmark, estimate the complete campaign, create the immutable budget lock, and publish `COMPLETE`. Its CPU handoff starts the first pending stage.
- [experiments/us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md](../../us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md)，第15行：- Sustained update costs are multiplied by the complete registered workload. `stage_startup_seconds` and `posterior_cache_startup_seconds` enter their own stage exactly once, before the 1.5 safety margin.
- [experiments/us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md](../../us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md)，第16行：- Preparation and scientific pilots have their own stage caps within the complete 24-card-hour limit. The one-hour short-preflight cap covers measured benchmark and recheck allocations.
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第24行：先建立合成归档测试，确认缺失 report 模块时出现 3 项预期失败；bootstrap 增补测试在缺失输出键时再次确认预期失败。随后运行：
- [experiments/us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md)，第6行：- All seven stages execute their observation or posterior-draw objectives; shared ES preserves encoder/decoder while complete refinement updates separate model copies.
- [experiments/us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md)，第8行：- Registered KL warmup records beta, unweighted normalized KL, hourly KL sum and negative ELBO; complete VAE retains beta=1.
- [experiments/us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md)，第9行：- Complete raw and EMA development metrics survive in evaluation_history, including nested metrics and NumPy/tensor values converted to portable CPU values.
- [experiments/us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md)，第16行：- Cooperative development evaluation observes the allocation budget and training signals through a temporary callback. EvaluationPaused returns a paused checkpoint with pending=True; raw weights, RNG, curves and rankings recover exactly while completed updates remain committed.

## 历史证据范围与信息条件

- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第5行：本轮Pro两轮讨论全文为`PRO_REVIEW_FULL.txt`与`PRO_FINAL_REVIEW.txt`，最终建议已取回。模型采用新独立模块，不复用旧生成器权重。冻结背景资产的历史路径和SHA256保留在机器清单，正文统一使用功能名称。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第11行：训练原始26,304小时，四帧历史及真实站支持审计后26,262小时。原始B有效站时刻13,132,476；各小时实际可监督站数由当前输入支持和固定查询几何共同确定。至少4个合格B站才训练。训练固定参考站数442，用于似然与KL共同归一。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第17行：训练监督来自真实站点；不存在独立密集高分辨率场真值。VAE后验样本是GAN和Diffusion的潜空间预训练目标，模型精度由独立C站评价。空间零空间的细节用场图和功率谱诊断，面积平均精度不从独立站成绩直接推断。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第38行：每小时、每成员共享一个全域潜变量z，主配置64维；128维通过B训练内部诊断决定。它控制整张场的共同变化，不是逐站独立噪声。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第62行：独立有效差异尺度σ_eff候选3.18/6.36/12.72 mm，通过B训练内部物理站组折和分离天气面板的先验CRPS选择，NLL作同分次序。该尺度表示未解析差异，不解释为实测GNSS仪器噪声。观测概率产品在地表查询后加一次独立差异项；发布场成员不包含这项站点独立噪声。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第84行：第一层比较共享且冻结的条件编码器、解码器和误差核，检验VAE/GAN/Diffusion先验差别；三者均获得真实站ES微调。第二层为三个独立完整模型，允许各自更新条件编码器、解码器及先验。第二层结论对应完整体系，不单独归因于先验。解码器更新后不再用旧后验缓存继续DSM。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第106行：全年使用K128 IID概率成员，其相同噪声前缀K64用于同一检查点的采样敏感性核查；点均值使用独立反向配对噪声池，概率ES-U不使用配对成员。报告RMSE、MAE、偏差、CRPS、观测NLL、ES、VS、80/90/95%覆盖与宽度；分组看季节、高程、站距、密度、边界及预登记天气过程。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第108行：结构评价报告0—25/25—50/50—100/100—200 km站间差值，原场相对4×4平均至100 km再回采样的收益，以及逐站打乱成员编号后的空间依赖变化。功率谱、场图及残差图用于解释；独立站与同步7日块bootstrap（2,000次）、14日敏感性提供精度证据。
- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)，第110行：交付每模型point_best/joint_best权重、训练和开发曲线、全年逐站集合预测、点值与概率地图、校准与宽度图、站间变化及空间依赖图、预登记过程场图和频谱、实际卡时和来源哈希。队列结束不自动读取D2020或扩展下一轮实验。
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第1行：# 训练与独立站评价图表导出
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第7行：- 训练目标曲线：各训练阶段分别绘图；同时导出所有数值训练历史 CSV。
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第9行：- 独立 C 站地图：每个模型 point 产品的逐站 RMSE，以及融合背景 RMSE 减模型 RMSE 的增益。地图使用投影坐标和真实站位置。
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第10行：- 80/90/95% 观测预测区间：覆盖率、区间宽度及覆盖率—宽度图。计算先在各站内按有效时刻平均，再对有观测的站等权平均，与现有集合评价口径一致。
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第20行：图中数值来自独立 C 站观测。25 km 输出间距与站点精度分别解释，图表不提供独立密集格网真值结论。高度数组缺失时明确跳过高度条件面板。此模块计算点值 RMSE 增益区间；ES 置信区间及天气过程/频谱由其他评价模块负责。
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)，第32行：最近完整结果为 **7 passed，17.92 s**。测试独立验算遮罩、逐站增益、站宏覆盖率与池化覆盖率的区别、距离箱端点及站对计数、缺高度跳过、未选候选忽略、同档只读取一次、point/joint 不同赢家、未评分位置非有限预测拒绝、固定种子重复性、站宏/微 RMSE 区别和整组重采样区间。

## 原始协议、结果与报告

- [experiments/us/20261122_full_generative_design_v01/ARCHITECTURE.md](../../us/20261122_full_generative_design_v01/ARCHITECTURE.md)（documentation；read）
- [experiments/us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md](../../us/20261122_full_generative_design_v01/PERFORMANCE_ORCHESTRATION_REPORT.md)（report；read）
- [experiments/us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md](../../us/20261122_full_generative_design_v01/RESULTS_EXPORT_REPORT.md)（report；read）
- [experiments/us/20261122_full_generative_design_v01/TASK1_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TASK1_TDD_REPORT.md)（report；read）
- [experiments/us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md](../../us/20261122_full_generative_design_v01/TRAINING_TDD_REPORT.md)（report；read）
- [experiments/us/20261122_full_generative_design_v01/population/C2019.lock.json](../../us/20261122_full_generative_design_v01/population/C2019.lock.json)（protocol；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
