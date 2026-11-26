# 当前研究进展：2026-11-26

最新G175/G265双种子各3000块完成：[天气相关核完整配对报告](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_weather_projective_gp/paired/REPORT.md)。近邻F6差值解析CRPS分别改善20.29%/20.76%，G265相对G175额外改善0.59%；基础场点预测与S175解析均值/标准差逐值不变。G265→Local/Affine以及[固定基础场均值的形状/尺度三臂](superpowers/specs/2026-11-26-centered-nongaussian-control.md)均完成双种子各3000更新，完整配对汇总尚待完成。阶段旧状态见下文，当前以本段和实际实验锁为准。

新增[研究优先级与尚待解决的问题](research_notes/2026-11-26-research-priorities-after-weather-gp.md)：优先比较自条件/留站创新、检验有向时滞传递、建立严格折外残差，再评估强Gaussian之上的扩散收益。[逐站诊断](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_weather_projective_gp/g175_locality/REPORT.md)显示PLTX/P475/PTEX六小时误差约22–25mm；137注册查询中当前/F6有标签站均为135个。

研究优先级已据实测重排：[下一步突破方向](research_notes/2026-11-26-next-breakthrough-directions.md)。优先固定均值、校准天气/lead相关性，再检验有向时滞创新传递和训练空间分布；长历史、连续滚动场与频域作为独立后续候选。

最新问题核验：[扩散收益与均值误差分离](research_notes/2026-11-26-diffusion-benefits-and-mean-error.md)。Full改善未来近邻差值CRPS13%–15%，当前/远距离及全场评分有代价；扣除有限成员方差后，Full当前均值MSE相对基础场的差值区间仍为正。点预测继续保留9.818mm基础场，天气相关核及Local/Affine衔接继续按固定方案验证。

新增执行：[逐站因果编码、天气条件核与轨迹扩散](research_notes/2026-11-26-weather-projective-trajectory.md)。同历史12配置和C/H/S续训全部双种子完成；Full近邻站对F6差值CRPS改善13%–15%，同图Gaussian控制约11%，支持优先校准空间依赖。天气GP两臂双种子正式3000块训练进行中；其Local/Affine衔接已实现并通过真实缺测4更新，尚未正式训练。项目默认568测试、研究模块165测试通过。[全流程PNG/SVG](figures/weather_native_20261126/weather_projective_trajectory.png)与精确公式已归档。


新增执行：[逐站原生天气、因果时间编码与残差扩散](research_notes/2026-11-26-native-weather-diffusion.md)。两种子各12组、每组3000连续块与raw/EMA评价已完成。相同72h状态历史下，完整联合GRU/LSTM/Transformer的双种子F6宏RMSE分别为14.177009/14.251512/14.127401mm；Transformer相对原Transformer的F6差值95%共同月份区间[−0.288695,−0.110130]mm，当前R差异区间跨零。详见[双种子配对报告与曲线](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_native_mean_analysis/paired/REPORT.md)。逐站模型误差、GNSS/T/P、mask/age、身份、有向地形/海岸查询和新增均值头零初始化已列为[必做目标](research_notes/2026-11-26-goal-update.md)。C/H/S各新增3000步的均值续训及强Gaussian参考轨迹扩散正在执行；[全流程架构图](figures/weather_native_20261126/gp_trajectory_process.png)区分这两条独立实验线。

**最新主评价：历史观测创新校正达到9.817064 mm。** 原GRU+p14为9.929151 mm；I1_history两种子9.867690/9.766438 mm，平均改善0.112086 mm，共同月份95%区间[−0.193456,−0.042281] mm；小时增量平均5.577307 mm，原参考5.583871 mm。I0仅当前创新会增加小时增量误差；I2年度交互相对I1的额外收益区间跨零，因此冻结9维I1作为补充验证的主候选。详见[设计、架构与训练记录](superpowers/plans/2026-11-25-observed-innovation-refinement.md)及[统一配对报告](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_warm_joint_precision/combined_precision/REPORT.md)。I系列是3000个连续训练块的解析拟合，不是新增Transformer或diffusion训练；原14条神经训练轨迹各3000更新也已完成。

主候选和六个系数哈希已在读取新增日期前锁定。冻结模型在2020每月15–17日由10.869282改善至10.740343 mm，两种子均改善，增量6.355685→6.334460 mm；合并72个抽样日后为10.441761→10.319804 mm，仍非完整年度。完整成果、六类误差、逐站地图与下一步见[结果和编码方案](research_notes/2026-11-25-precision-final.md)。主日历海岸150–300 km档增加0.104 mm，补充日历同档改善0.110 mm，后续需关注天气过程与地形的联合影响；新diffusion尚未在此基础场上重训。

训练分布瓶颈已实测：[距离重平衡D对照](superpowers/specs/2026-11-25-distance-balanced-precision.md)。重放双种子各3000个连续块，随机训练目标59.4%距当前条件站<50 km，25%hole100仍58.7%；137个开发坐标×3布局的纯几何比例为16.3%。D双种子已完成，EMA平均9.916258 mm，相对同架构S平均改善0.008096 mm；相对原参考的seed17方向略增，未替换原参考。固定训练窗口容量探针14.007→11.235 mm说明新分支能够拟合残差，该探针不作开发精度排名。

精度衔接实测：[既有约10 mm基础场上的继续优化](research_notes/2026-11-25-precision-continuation.md)。T2/T5、Q、S、A、D全部双种子3000步完成。新增神经分支未形成对原参考的双种子一致提升；当前推进上述I1历史观测创新候选。修正幅度和误差分层图已归档；原确认集及C/D标签保持隔离。

精度优先级更新：[在既有约10mm基础场上的3000步正式衔接](superpowers/plans/2026-11-25-warm-joint-precision.md)。最佳GRU+p14参考、原归一化与8×8历史完整接回新16×16Transformer；seed42复现9.99836mm，保存预测最大差7.63e-6mm。冻结参考T2/T5与低学习率联合T5双种子3000步均已完成，未形成稳定晋级收益；后续I1结果见上。

代码与实测：[时空融合与谱残差实验全流程及12张架构图](research_notes/2026-11-25-joint-spectral-implementation.md)。修正版T0–T5各60步、五折T5各60步、表示100步、九组概率及Gaussian各100步、四组v及Gaussian各100步已完成；[报告](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_joint_spectral_pilot_v2/REPORT.md)。规则256锚点自重构6.85/8.61mm，几何适应256诊断降至0.61/0.68mm，物理组留出仍20.21mm。A0同100步epsilon/v为344.63/33.13mm，支持先解决表示及高噪声端数值问题。该进展更新下列“尚未实现”设计状态；随后基础场T2/T5、Q/S/A/D已完成双种子3000更新，概率分支仍为工程短训。

频域探索：[Transformer与时空频域残差扩散可行性](research_notes/2026-11-25-spectral-diffusion-feasibility.md)。已归档时域/频域、空域/图谱及双频域架构、风险与分阶段对照；本机用已有p14开发残差完成1384个72小时站点窗的谱诊断和不规则Fourier/GFT代数核验。该结果不代表新频域网络训练收益。

方法依据：[已有时序扩散与本项目的对照](research_notes/2026-11-25-temporal-diffusion-comparison.md)。核对TimeGrad、CSDI、TMDM、CorrDiff、GenCast等8项工作，明确当前固定AR创新方案与条件动态的差别；动态转移和条件尺度属于新增待验证消融。

当前设计：[掩码Transformer时空融合与标准化残差场](superpowers/specs/2026-11-25-masked-transformer-fusion-residual-design.md)。保留p14与海岸/地形分支，以72小时、16×16因果时间状态接入两级残差空间解码器；基础场冻结后，构建折外残差与标准化创新扩散。架构流程图已导出PNG/SVG；随后T2/T5双种子3000更新已完成，见页首统一结果。

下一步：[p14时空融合优化与残差扩散实验计划](superpowers/plans/2026-11-25-p14-fusion-next-stage.md)。以现有同条件实测为起点，先完成双种子，再分开检验训练策略、残差解码融合及长历史；固定基础场后比较高斯与diffusion。该增补是待执行计划，当前结果不代表新增架构或扩散已经训练。

p14完整载体融合已完成12个3000步训练配置与2个冻结对照：[连续训练计划](superpowers/plans/2026-11-25-p14-continuous-fusion.md)。训练2016–2019连续小时、2020每月72小时×3布局开发评价，双种子站宏RMSE：p14续训9.947306、空间特征9.936249、Transformer9.950066 mm；来源见[完成快照](../maintenance/transformer_architecture_20261125/evidence_snapshot.json)。新Transformer架构着重检验时间特征与空间场的融合机制，尚无新增结构成绩。

2026-11-25真实本机预报首轮完成：[结果与图件](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_temporal_forecast_screen/REPORT.md)及[分析与后续决策](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_temporal_forecast_screen/ANALYSIS.md)。26配置、双种子、12/48/72h；2016–2018内层拟合、2019选模、2016–2019重拟合、2020抽样开发。六小时宏RMSE：空间持久性15.163、当前帧14.203、Transformer72为13.914mm；小时增量尚未改善。本轮小型空间载体及C1海岸输入，尚非p14复现、完整年度或diffusion结果。

下一阶段统一入口：[计划与审阅修订](NEXT_STAGE.md)。新时间分支处于设计阶段；历史重建指标与未来预报分开评价。

长历史数据核验：[时空模型与数据组织审阅](superpowers/specs/2026-11-25-temporal-history-data-organization.md)。现有731段训练包已复算连续小时、重复、缺测和滞后相关；建议新建唯一小时轴后比较12/24/48/72/168小时。该核验不代表新时序模型训练结果。

最新正式结果：[p14 固定支撑编码器](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_fixed_support_encoder_p14_formal/REPORT.md)。410 个训练站、137 个验证站、12 个月×24 小时×3 布局、两种子完成 600 秒训练；扩展宏 RMSE 为 **10.806 mm**，CRPS 为 8.286 mm，小时增量宏 RMSE 为 5.809 mm。相对 v13 点特征＋小 U-Net 的 10.828 mm 改善 0.022 mm，属于同一精度水平。`architecture_comparison.png` 和 `learning_curves.png` 已提供架构与训练曲线对比。p14 是确定性基础场，历史 diffusion 数值未在 p14 残差上重训。

最新空间采样筛查：[p15 空间支撑采样](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_optimization_campaign_p15/REPORT.md)。在 120 秒固定时间监控中，random 6.286 mm、hole100 6.526 mm、hole150 7.155 mm、distance_stratified 7.782 mm；这些是内部筛查分数，不能与 p14 扩展宏 RMSE 混排。当前保留 random 控制并加入 hole100 空间外推臂，其他采样需完整扩展评价。

最新完成：[v13 同输入 U-Net 基础场比较](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_unet_foundation_v13/RESULTS.md)。五组十次拟合全部完成，包含大宽度 U-Net48。双种子平均宏 RMSE：MLP11.175、U-Net16为11.745、U-Net24为11.400、点特征＋U-Net16为10.828、大U-Net48为11.800 mm。融合模型是优先候选；开发日历重采样区间跨零，下一步优先空间留出监控和天气日期覆盖。29项测试及22组预测核验通过，确认及C/D标签未读取。

最新完成： [v12夜间残差场实验分析](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261124_residual_field_v12/ANALYSIS.md)。E1201–E1207七组、两个种子、32次拟合全部完成，主队列4小时12分34秒。K16扩展开发评价中，相关高斯残差宏RMSE为11.880 mm，残差扩散12.055 mm，扩散后XGBoost为12.386 mm；26项测试通过。基础场与残差分工有效，后期过拟合和空间联合分布是下一步重点。E1208冻结采样复核见同一报告；下列v11及历史结果使用各自评价口径。


最新实践：[v11观测编码与直接扩散](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_observation_transport_v11/RESULTS.md)。10组真实训练、24,000次更新完成。2018每个月月1日24小时、137验证站、3布局的扩展评分中，两种子平均宏RMSE：base14.307、raw12.456、transport12.231、高斯12.285、扩散12.339mm；固定高程IDW12.066mm。raw按季度选模规则进入B，未用扩展结果回改。扩散对50km以内站对差值CRPS比独立高斯改善约5.0%，保边际空间shuffle使其变差0.135/0.131mm，支持局地联合生成；50–150km关联仍需校准。C门槛未通过，自生成历史时序训练未启动。下一步优先空间空洞训练和固定支撑查询：训练/验证最近观测中位距离39.69/77.93km，查询64→137也会改变输出。18测试、20组预测复算和4组门槛复算通过；确认138及C/D标签未读取，未达8mm。


最新冻结权重诊断：[v10瓶颈分析](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_terrain_diffusion_v10_diagnosis/ANALYSIS.md)。同人口固定高程IDW12.290mm优于当前B1平均16.299mm，优先改进观测/高差编码。训练真实目标历史带来明显优势，自生成历史在layout0的两种子诊断中没有稳定点值收益；查询32→137有输出敏感性，K4→K32约减少1.3mm集合均值评分误差。优化有有效学习，尚不能判定收敛或统一过拟合；本次未重训、未读取确认/C/D标签。

最新实现：[v10真实本机筛查](../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_terrain_diffusion_v10/RESULTS.md)。已按允许变量独立QC激活410训练/137验证合同，确认138及C/D标签保持隔离。B1/B2/D0/D1两种子各1000步，平均宏RMSE分别16.299/18.820/19.338/18.860mm；这是新人口下的缩减筛查，不能与旧v04约9mm直接比较。已提供真实训练曲线、概率/地形诊断、24h生成及完整4h×K4×DDIM20反向验证；正式P2/P3未完成，当前未达8mm。

设计入口：[地形感知时空图扩散v2](superpowers/specs/2026-11-23-terrain-temporal-diffusion-v2.md)。旧A+B685站按100km块与物理组形成训练410、验证137、空间确认138；原C/D各110保留登记角色。训练池内动态条件/目标轮换，直接生成ZWD。旧A-only协议用于历史复现，不再作为新轨道的唯一条件规则。v10原型与正式设计的差异在代码README逐项列出。

新研究设计（尚未训练）：[掩码时空图扩散设计](superpowers/specs/2026-11-23-masked-spatiotemporal-diffusion-design.md)。按用户新方向直接生成ZWD，采用MAE式遮站编码、地形图和因果时序条件，暂不使用v04均值背景。设计包含允许变量QC、A唯一条件输入、连续查询一致性门槛、空间时间验证及分阶段本机训练预算。浏览器协作因内置浏览器初始化失败未提交，具体记录与待交接材料见配套browser-handoff文档；文献核验由客户端完成。该设计不改变下列实验结果或当前保留基线。

最新v09见工程 `results/20261123_diffusion_observation_v09/VERDICT.md`：完成24/64维多尺度表示与观测空间CRPS/时间CRPS配对，两种子、4800预训练＋1600微调更新。月27日K128，24维时间分支平均CRPS从6.1741降至5.5980，但宏RMSE从9.1645升至9.3266mm；64维时间分支为9.3573mm。时间分支四组的成员shuffle均使增量CRPS变差，支持局部时间关联机制，seed42效应较小。64维训练投影减少约42%残差平方误差，未转化为外层精度收益。继续保留v04约9.003mm，尚未达8mm；优先检验固定均值与中心化扩散残差。后验证指标复算误差0、全格网回查2.36e-5mm。

Diffusion数据量后续见工程 `results/20261123_diffusion_data_v08/VERDICT.md`：保持A174和原残差坐标系，72段×3000步仍选原300/600步，与72段×1500步一致。扩至144个2016/2017训练片段、3000步，月27日宏RMSE9.2778/9.7595mm，原DiT9.1606/9.1594mm；月22日平均综合分略改善，但无跨日历一致收益。两种子原轨迹复现、保存指标复算误差均0。当前继续保留v04，不直接扩大该24维残差扩散配置。

真实扩散候选筛查见工程 `results/20261123_diffusion_v07/VERDICT.md`：条件Gaussian、MLP Diffusion、普通/稀疏FFN Diffusion Transformer，两种子各1500更新。普通DiT在两日历K64/K128平均综合分中领先新候选；固定月27日宏RMSE为9.1606/9.1594mm，未超过v04的9.0099/8.9959。24维受限残差投影只减少约18%训练残差平方误差，新候选90%区间覆盖约35%—44%，当前保留v04。已验证100步v扩散、20步DDIM、因果生成、完整背景＋残差格网回查；稀疏4专家top2没有稳定优势且未实现算力节省。

DiT的月27日K64时间成员shuffle使增量CRPS由2.8509/2.9448降至2.8445/2.9035，当前未证明其成员时间对应有益，不能把自回归结构当作合理动态已验证。

遮站自监督与端到端图生成小试见工程 `results/20261123_masked_graph_v06/RESULTS.md`。固定月27日71站/20108站时，遮站预训练宏RMSE为10.2518/10.2637mm，直接监督图模型10.2844/10.4706mm；有局部开发收益，但原v04约9mm更好，当前保留v04。K128时间相关潜变量相对独立潜变量的1h增量CRPS改善8.9%/5.6%；完成8×24×144×96格网样例及隐藏值隔离、因果性、回查和指标复算。总6400均值更新+2400生成头更新，非VAE/Diffusion全量训练，未达8mm。

图融合追加小试见工程 `results/20261123_graph_fusion_v05/RESULTS.md`：两种子、self/graph 每臂600步，冻结历史创新与生成分支，统一格网算子。固定月27日71站/20108站时，graph宏RMSE为9.0166/8.9959 mm，原v04为9.0099/8.9959；seed17内层选择零步。此次图模块未形成增益，保留v04；K128、时间因果、节点重排和查询分块核验完成。它检验追加模块，不代表端到端图生成模型上限。

## 新目标与输入约束（连续场探索）

用户最新目标是具有地形细节、天气过程连续性的生成式ZWD场序列，先本机小试再全量验证。ERA5只允许与站点气象变量对应的信息，排除水汽柱数据。审计发现下表历史气象U-Net含ERA5 PWV，因此旧成绩属于原输入条件，不代表新约束下的成绩。

新小试在工程`results/20261123_sequence_spike_v01/`与`results/20261123_sequence_spike_v02_grid/`，仅使用A站Z0/T/P，从头训练；文献依据和历史制包筛选限制见v01/RESEARCH.md。v01为直接坐标原型，v02统一格网训练/评价算子，状态以各目录results.json与验证记录为准，均非正式新冠军。

v02已完成两种子本机确认：在2018内部外层12个24小时片段、71有效站/19907站时上，时间相关残差相对逐时独立残差的一小时增量CRPS改善8.18%/6.97%；逐时保持全部样本而打乱时间成员对应后评分退化。独立噪声K128复核同方向。GRU均值RMSE在两种子上均高于逐帧MLP，不能声称时间编码器优越。已生成8成员×24小时×144×96序列，检查格网回查误差约1.36e-5 mm。结论为时间相关残差机制有小试支持，尚非最终GAN/Diffusion/VAE方法。

进一步探索见`results/20261123_sequence_mechanisms_v03/RESULTS.md`：均值500→2000步未改善月22日检查集；随机分支350→1400步有小幅概率收益；固定750m高差编码不稳定；多时间尺度收益对独立采样敏感，暂不替换单尺度。

后续适应性小试见`results/20261123_sequence_innovation_v04/RESULTS.md`：冻结v03均值/随机场，使用A站当前及1h/3h观测创新，B-fit拟合ridge、B-inner选正则化。在2018每月27日12段24小时、71站/20108站时上，两种子站宏RMSE从9.5095/9.5665降至9.0099/8.9959 mm，CRPS从5.4119/5.4638降至5.1215/5.1482 mm。只用当前创新会损害时间增量评分，加入历史后恢复并改善。超过偏差-only对照，独立K128同方向；逐时随机区间宽度不变。该候选值得后续验证，不称最终盲测、最终生成模型或细尺度物理恢复证据，历史PWV队列限制仍适用。

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

### P15 正式续训更新（2026-11-24）
P15 原始 6.29/6.53 mm 等数字是固定时间 monitor RMSE，仅用于采样臂筛查。统一扩展评价续训结果为：random 11.042 mm，hole100 10.804 mm（两 seed 均值）；p14 基线 10.806 mm。hole100 与基线持平，random 未提升。后续优先使用 hole100 进行残差目标、显式边消息和时间外推验证，正式指标沿用宏 RMSE、微 RMSE、MAE、Bias、CRPS、P90/P95 及小时增量 RMSE。

### 下一阶段方案（2026-11-25）
详细的 2016–2019 训练 / 2020 验证、边消息空间编码、基础场与残差 diffusion 配对方案见 [下一阶段优化方案](superpowers/specs/2026-11-25-next-optimization-plan.md)。网页 Pro 协作会话已打开，但本轮正文回收未完成；方案仅使用本机已核验的 p14/V15/扩展 U-Net 结果。

### 执行计划入口（2026-11-25）
下一阶段每一步的文件、接口、测试、停止条件和提交节奏见 [时空联合 Diffusion 实施计划](superpowers/plans/2026-11-25-spatiotemporal-diffusion-implementation-plan.md)。
Karpathy 指南已落实为：先锁合同、先写失败测试、最小改动、每步独立验证、每步单独提交。

### 完整实验协议（2026-11-25）
新阶段的实验原理、数据合同、模型分支、训练步骤、评价图件、停止规则和归档格式见 [完整实验协议](superpowers/specs/2026-11-25-spatiotemporal-experiment-protocol.md)。
