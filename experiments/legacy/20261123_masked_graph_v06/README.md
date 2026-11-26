# 20261123_masked_graph_v06

当前公开来源：`experiments/us/20261123_masked_graph_v06`；[来源目录](../../us/20261123_masked_graph_v06)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_masked_graph_v06`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)，第1行：# 动态遮站图生成场小试

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)，第15行：训练与选择全部完成后固定外层评估，报告均值宏/微RMSE、K64及独立噪声K128 CRPS、1h增量CRPS、区间覆盖与宽度。生成24h完整格网样例。验证隐藏值扰动不变、未来扰动不变、图节点重排、查询分块及格网回查一致。源码/依赖/数据/模型保留哈希，代码与轻量结果同步GitHub。

## 历史证据范围与信息条件

- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)，第5行：数据沿用v03哈希锁定的A174/B-fit/inner/outer及2016—2018片段；新值输入仅Z0/T/P。历史队列PWV筛选限制继续适用。C/D不读取，月22/27日已暴露外层仅作开发评价。保持71有效站、20108站时的月27日主指标，不改变人口。
- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)，第13行：冻结各臂选中均值与图表征，各训练400步时序相关潜变量生成头；masked另训练同初始化/预算的逐时独立潜变量对照。共享空间8维Gaussian潜变量与中心化二次基，AR(1)演化，神经系数条件于图读出，解析条件均值为0。它是概率评分训练的生成网络原型，不宣称已实现VAE或Diffusion。每100步按inner CRPS+0.5增量CRPS选择，训练K8，选择K32。
- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)，第15行：训练与选择全部完成后固定外层评估，报告均值宏/微RMSE、K64及独立噪声K128 CRPS、1h增量CRPS、区间覆盖与宽度。生成24h完整格网样例。验证隐藏值扰动不变、未来扰动不变、图节点重排、查询分块及格网回查一致。源码/依赖/数据/模型保留哈希，代码与轻量结果同步GitHub。
- [experiments/us/20261123_masked_graph_v06/RESULTS.md](../../us/20261123_masked_graph_v06/RESULTS.md)，第26行：seed42在月22日的遮站收益区间跨零，不能称跨日历稳定显著。时间成员重排诊断采用K64，表中独立/相关对照采用独立噪声K128。
- [experiments/us/20261123_masked_graph_v06/RESULTS.md](../../us/20261123_masked_graph_v06/RESULTS.md)，第34行：月22/27日属于已暴露开发日历，C/D未读取。输入值仅Z0/T/P，历史队列仍含PWV筛选，正式协议须重建允许变量质控。
- [experiments/us/20261123_masked_graph_v06/RESULTS.md](../../us/20261123_masked_graph_v06/RESULTS.md)，第38行：保留遮站隔离训练及共享时序潜变量两个机制。下一候选应保留原模型的多尺度聚合和当前/历史创新锚定，再检验图编码是否增加信息。此次新图骨干同时改变了多尺度背景和读出，不足以把相对v04的差距归因于自监督或图网络本身。该后续组合尚未实测。
- [experiments/us/20261123_masked_graph_v06/REVIEW.md](../../us/20261123_masked_graph_v06/REVIEW.md)，第1行：# 独立代码与结果审查
- [experiments/us/20261123_masked_graph_v06/REVIEW.md](../../us/20261123_masked_graph_v06/REVIEW.md)，第11行：结论：遮站相对direct在day27改善0.03261/0.20694mm，但相对v04高1.24190/1.26781mm；seed42月22日收益区间跨零。K128相关潜变量增量CRPS相对独立改善约8.9%/5.6%。保留原v04，不声明8mm、全新空间泛化或物理细节恢复。

## 原始协议、结果与报告

- [experiments/us/20261123_masked_graph_v06/PROTOCOL.md](../../us/20261123_masked_graph_v06/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_masked_graph_v06/RESULTS.md](../../us/20261123_masked_graph_v06/RESULTS.md)（report；read）
- [experiments/us/20261123_masked_graph_v06/REVIEW.md](../../us/20261123_masked_graph_v06/REVIEW.md)（report；read）
- [experiments/us/20261123_masked_graph_v06/experiment.lock.json](../../us/20261123_masked_graph_v06/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_masked_graph_v06/results.json](../../us/20261123_masked_graph_v06/results.json)（metadata；read）
- [experiments/us/20261123_masked_graph_v06/selection.lock.json](../../us/20261123_masked_graph_v06/selection.lock.json)（protocol；read）
- [experiments/us/20261123_masked_graph_v06/verification.json](../../us/20261123_masked_graph_v06/verification.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
