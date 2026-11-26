# 20261123_sequence_mechanisms_v03

当前公开来源：`experiments/us/20261123_sequence_mechanisms_v03`；[来源目录](../../us/20261123_sequence_mechanisms_v03)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_sequence_mechanisms_v03`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)，第1行：# 连续场进一步机制探索：预登记小试

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)，第63行：成功训练与评价耗时274.1秒，设备NVIDIA GeForce RTX 4060 Laptop GPU。内置浏览器自动化仍初始化失败，网页模型复核未完成。

## 历史证据范围与信息条件

- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)，第8行：3. 固定纯距离逐帧2000步的均值及编码：单时间尺度AR生成350步/1400步，与逐潜维独立时间尺度AR生成1400步比较。隐状态仍8维，相同头初始化、噪声和评分预算；仅rho参数由1变8。两臂均使用相同CRPS+0.5时间增量CRPS+0.1固定站序差值CRPS。
- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)，第10行：继续固定Z0/T/P输入与A/B角色，不读C/D标签。旧队列PWV筛选限制仍存在，不称最终干净数据协议。所有输出使用统一格网四角非线性生成、双线性查询和高度还原。
- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)，第12行：训练与inner/outer继承v02，每段30小时含6小时热身。新增确认集为2018每月22日24小时（另加热身），沿用B-outer站组；不用于选择、阈值或调参。该日期新于本连续场小试，不保证整个项目从未暴露，不能称最终盲测。全部模型完成后统一回看确认集。
- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)，第16行：评价记录RMSE、CRPS、1h增量CRPS、90%覆盖与宽度，K64及独立噪声K128；12个月天气块配对bootstrap只作探索性不确定性描述，不伪装多轮调参后的确认性显著性。地形分组阈值500/1500m预先固定，报告分组有效站与站时。
- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)，第3行：本轮比较训练预算、地形条件聚合和多时间尺度随机隐状态。所有对照为相同允许输入Z0/T/P、统一格网算子、两种子本机训练。仍沿用历史PWV筛选队列，C/D标签未读取。
- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)，第5行：确认集为2018每月22日的24小时片段，另加6小时热身，沿用B外层站组；不用于本轮训练或选择，但不保证项目历史从未暴露。以下指标均为该确认集。
- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)，第49行：独立噪声K128复核：延长随机分支训练的边缘/增量CRPS在两个种子均同向改善。多时间尺度相对单尺度，种子42边缘CRPS由5.7018变为5.7121 mm，改善方向反转；种子17仍改善。故多时间尺度只保留为采样敏感的小幅候选，暂不替换单尺度。
- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)，第57行：地形分层有效站、站时及RMSE保留在verification.json；高海拔分组若样本少，不能作整个山地代表。独立噪声K128与全部学习曲线见results.json。外层和确认集方向不一致时保留结果，不靠改评价集挑胜者。
- [experiments/us/20261123_sequence_mechanisms_v03/REVIEW.md](../../us/20261123_sequence_mechanisms_v03/REVIEW.md)，第3行：按requesting-code-review技能执行独立只读审查。均值两臂初始化、批次、模型参数、优化器及选择规则匹配；高差只改前15个聚合特征，背景和评分支持不变。多rho按最后8维广播，零向量初始化不额外消耗随机数。确认标签未进入特征、归一化、拟合或选择，全部训练完成后统一评分。

## 原始协议、结果与报告

- [experiments/us/20261123_sequence_mechanisms_v03/PROTOCOL.md](../../us/20261123_sequence_mechanisms_v03/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_sequence_mechanisms_v03/RESULTS.md](../../us/20261123_sequence_mechanisms_v03/RESULTS.md)（report；read）
- [experiments/us/20261123_sequence_mechanisms_v03/REVIEW.md](../../us/20261123_sequence_mechanisms_v03/REVIEW.md)（report；read）
- [experiments/us/20261123_sequence_mechanisms_v03/experiment.lock.json](../../us/20261123_sequence_mechanisms_v03/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_sequence_mechanisms_v03/results.json](../../us/20261123_sequence_mechanisms_v03/results.json)（metadata；read）
- [experiments/us/20261123_sequence_mechanisms_v03/verification.json](../../us/20261123_sequence_mechanisms_v03/verification.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
