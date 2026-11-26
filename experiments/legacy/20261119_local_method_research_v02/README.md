# 20261119_local_method_research_v02

当前公开来源：`experiments/us/20261119_local_method_research_v02`；[来源目录](../../us/20261119_local_method_research_v02)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261119_local_method_research_v02`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261119_local_method_research_v02/selected_method.lock.json](../../us/20261119_local_method_research_v02/selected_method.lock.json)，covariance_parameters.objective：3.3941169523845645

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261119_local_method_research_v02/selected_method.lock.json](../../us/20261119_local_method_research_v02/selected_method.lock.json)，status：local_development_candidate
- [experiments/us/20261119_local_method_research_v02/selected_method_v02.lock.json](../../us/20261119_local_method_research_v02/selected_method_v02.lock.json)，status：post_diagnostic_local_development_candidate

## 报告中的阶段声明

- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第113行：全年初次汇总在5、6月缺组时失败。所有月预测保留，修复后按固定组身份重建49908组时，缺测不填补。恢复锁文件单列FP32月预测与FP64评分。v01非嵌套候选记录和失败运行保留在实验清单中，不混入确认结果。

## 历史证据范围与信息条件

- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)，M0.metrics.fine.scope：common complete coarsening stencils intersected with S0
- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)，blend.metrics.fine.scope：common complete coarsening stencils intersected with S0
- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)，innovation_two_scale.metrics.fine.scope：common complete coarsening stencils intersected with S0
- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)，local_basic.metrics.fine.scope：common complete coarsening stencils intersected with S0
- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)，local_rich.metrics.fine.scope：common complete coarsening stencils intersected with S0
- [experiments/us/20261119_local_method_research_v02/full2019_blend/results.json](../../us/20261119_local_method_research_v02/full2019_blend/results.json)，evaluation_scope：C2019 full-year development
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第3行：本轮已与内置网页Pro完成三轮研究审阅，并在本机完成条件模型、联合高斯、重尾与天气条件尺度实验。可继续采用的研究基础是**冻结M0与A站IDW的混合均值，加上天气条件化的局部空间相关高斯残差场**。它取得了全年开发集的点值与联合评分收益。25 km仍表示输出间距；独立站证据尚未支持新增真实细尺度信息的主张。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第7行：A174为输入，B511的2018观测用于新增拟合，C2019为已查看的开发评价，D2020保持封存。M0沿用第5000步冻结EMA，曾使用B2016—17。B组留出检验新增方法的迁移，不代表整个M0链条从未见过这些物理站。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第29行：混合减M0为-0.348773 mm，2000次同步7日块配对95%区间[-0.398947,-0.285689]；混合减IDW为-0.282674 mm，区间[-0.413629,-0.166853]。全年包含53个有观测的时间块。C2019参与过开发，区间未包含全部方法选择的不确定性，不能作为最终盲测确认。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第44行：站观测预测另加独立地表未解析误差项tau。混合均值下重新用B拟合后，sigma1=8.560606 mm、sigma2=0.010000 mm（Z0），tau=6.182703 mm（地表）。粗尺度触下界，当前复合似然没有检出额外贡献；200 km组内监督并不足以否定远距离相关性。tau合并了观测和未解析变化，不能解释成纯仪器噪声，也不加入发布场图像。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第53行：ES/VS均采用128成员，组内按既定算子评分。点值产品使用解析均值。独立高斯直接从diag(Sigma)抽样；成员置乱保留每站每小时的有限集合边缘，两者承担不同对照作用。高斯CRPS和区间使用解析边缘，三种依赖对照在理论上相同。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第55行：混合均值下，相关减独立高斯的ES为-0.146238、VS为-0.251282。7日时间×组2000次重采样区间分别[-0.226025,-0.073547]、[-0.350605,-0.150790]，14日敏感性方向一致。收益在6个密集组的范围内成立，不直接代表陆地区域面积平均误差或跨组联合分布。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第79行：该模型升为优先概率候选，常数尺度保留为匹配基线。C2019分层诊断参与了候选结构的动机，随后在同一开发年复评；参数没有用C拟合，结果仍属于开发证据。完整输入与组重采样核验见`../20261119_local_conditional_probability_v01/review_crossed.json`。
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)，第99行：拟以R=64作为小试起点，EDM和GAN共享条件、表示和观测端ES，余项保持高斯。z为标准高斯时恢复当前高斯先验；本轮B站端协方差数值重建误差小于2.2e-13 mm²。这个检查验证线性代数接口，尚未训练该生成头，也未证明其精度收益。R64的空间迁移、条件独立余项假设、条件信息与均值漂移仍需后续小试检查。

## 原始协议、结果与报告

- [experiments/us/20261119_local_method_research_v02/B_only_parameters.lock.json](../../us/20261119_local_method_research_v02/B_only_parameters.lock.json)（protocol；read）
- [experiments/us/20261119_local_method_research_v02/RESULTS.md](../../us/20261119_local_method_research_v02/RESULTS.md)（report；read）
- [experiments/us/20261119_local_method_research_v02/local_method.lock.json](../../us/20261119_local_method_research_v02/local_method.lock.json)（protocol；read）
- [experiments/us/20261119_local_method_research_v02/results.json](../../us/20261119_local_method_research_v02/results.json)（metadata；read）
- [experiments/us/20261119_local_method_research_v02/selected_method.lock.json](../../us/20261119_local_method_research_v02/selected_method.lock.json)（protocol；read）
- [experiments/us/20261119_local_method_research_v02/selected_method_v02.lock.json](../../us/20261119_local_method_research_v02/selected_method_v02.lock.json)（protocol；read）
- [experiments/us/20261119_local_method_research_v02/verification.json](../../us/20261119_local_method_research_v02/verification.json)（metadata；read）
- [experiments/us/20261119_local_method_research_v02/full2019_blend/results.json](../../us/20261119_local_method_research_v02/full2019_blend/results.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
