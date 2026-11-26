# 20261123_sequence_innovation_v04

当前公开来源：`experiments/us/20261123_sequence_innovation_v04`；[来源目录](../../us/20261123_sequence_innovation_v04)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_sequence_innovation_v04`。

用途类别：`uncertain`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第1行：# A观测创新对连续生成场的校正小试

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第3行：v03显示均值加训无稳定收益、随机分支加训收益较小且区间仍偏窄。本轮在看到v03结果后追加，属于适应性探索，不称独立确认研究。
- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第7行：比较：未校正；B-fit全局偏差校准；A创新的50/150/300km核×无/750m高差权重六分量；同样六分量加1h、3h历史（18分量）。背景场和随机隐状态都冻结。校正在格网形成后以同一双线性和高度算子查询。每组ridge强度[0.0001,0.001,0.01,0.1,1]只按inner站宏RMSE选择；FP64训练站等权平方误差，截距不惩罚。
- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第9行：训练、inner沿用v03；回看月22日片段，新增月27日片段作为本轮预留天气检查，各12段24小时，6小时热身。所有系数及lambda锁定后才评分；月27日在此前连续小试未用，不保证整个项目历史从未暴露。始终不读C/D。
- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第11行：评分比较点值RMSE、增量RMSE，以及相同随机样本加同一确定性校正后的CRPS/增量CRPS；随机集合不重采样以制造不同人口。K64与独立噪声K128。明确创新机制须优于全局偏差校准，不能把简单偏差去除包装为复杂空间机制优势。
- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)，第13行：输入仍仅Z0/T/P；历史队列PWV筛选限制不变。机制是已有项目校正思路在新信息条件、序列随机场下的验证，不声称首次提出ridge或观测创新。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第1行：# 观测创新与历史条件：连续生成场小试
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第5行：主要结果为2018每月27日24小时片段、71有效B外层站、20108站时。月22日回看结果另保留。二者均不是整个项目的最终盲测；C/D标签未读取。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第20行：none为未校正；bias为公共高度Z0常数偏移，经高度算子转换后不是地面ZWD统一常数；current使用当前A创新的六个几何核；history进一步加入1h和3h历史，总18分量。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第22行：- 种子42：历史创新校正相对未校正的站宏RMSE改善5.25%，CRPS改善5.36%。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第30行：- 种子17：历史创新校正相对未校正的站宏RMSE改善5.96%，CRPS改善5.78%。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第38行：当前创新改善点值，但两个种子的小时增量评分均变差；加入历史创新后，该项评分恢复并优于未校正。月22日回看的点值与概率方向一致；独立噪声K128复核的概率指标也同向，确定性均值本身不随集合大小变化。相同随机样本只作确定性平移，逐时区间宽度保持不变，收益不是靠放宽区间取得。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第40行：这支持“输入观测约束＋因果历史校正”作为下一阶段候选。核函数和ridge本身是既有方法，本项目旧输入条件下也探索过创新校正；新的证据针对禁用水汽柱、统一格网和连续随机场的组合，不据此声称已形成论文创新或已恢复全部细尺度物理过程。
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)，第42行：输入Z0/T/P仍来自历史PWV完整性筛选队列，正式研究须从原始文件重建允许变量质控。海岸距离、自监督编码、连续大空间块留出与长序列状态衔接尚未验证。
- [experiments/us/20261123_sequence_innovation_v04/REVIEW.md](../../us/20261123_sequence_innovation_v04/REVIEW.md)，第3行：独立审查确认：创新在公共高度Z0空间计算；先格网插值后恢复站点高度；0/1/3h切片正确；训练站等权FP64拟合，缩放仅来自拟合样本；B-inner独立选正则化；两种子系数锁定后才评分月22/月27片段。bias对照是公共高度常数，不是统一地面偏移。
- [experiments/us/20261123_sequence_innovation_v04/REVIEW.md](../../us/20261123_sequence_innovation_v04/REVIEW.md)，第5行：复审确认：点值bootstrap采用确定性预测，不使用有限集合均值；概率bootstrap重算记录加权指标，人口一致；确定性平移恒等式误差约2.84e-14，区间宽度未改变；两种子、两段日历的独立K128概率方向一致。未发现严重或重要未解决问题。

## 原始协议、结果与报告

- [experiments/us/20261123_sequence_innovation_v04/PROTOCOL.md](../../us/20261123_sequence_innovation_v04/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_sequence_innovation_v04/RESULTS.md](../../us/20261123_sequence_innovation_v04/RESULTS.md)（report；read）
- [experiments/us/20261123_sequence_innovation_v04/REVIEW.md](../../us/20261123_sequence_innovation_v04/REVIEW.md)（report；read）
- [experiments/us/20261123_sequence_innovation_v04/experiment.lock.json](../../us/20261123_sequence_innovation_v04/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_sequence_innovation_v04/results.json](../../us/20261123_sequence_innovation_v04/results.json)（metadata；read）
- [experiments/us/20261123_sequence_innovation_v04/selection.lock.json](../../us/20261123_sequence_innovation_v04/selection.lock.json)（protocol；read）
- [experiments/us/20261123_sequence_innovation_v04/verification.json](../../us/20261123_sequence_innovation_v04/verification.json)（metadata；read）

## 待核查项

- 无明确状态证据；不由目录名称或指标文件推断完成。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
