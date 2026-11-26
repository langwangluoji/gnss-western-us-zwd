# 20261123_diffusion_observation_v09

当前公开来源：`experiments/us/20261123_diffusion_observation_v09`；[来源目录](../../us/20261123_diffusion_observation_v09)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_diffusion_observation_v09`。

用途类别：`uncertain`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第1行：# 多尺度表示与观测空间扩散训练

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第3行：用户批准继续优化。保持A174、Z0/T/P、原72段2016/2017训练窗口、B-fit/inner/outer与固定月22/27日人口不变，冻结v04背景；当前为适应性开发实验，C/D不读取，历史PWV队列限制继续适用。
- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第7行：每个空间/种子从头进行1200步v预测预训练，batch128，AdamW0.001；每300步按inner K16的CRPS+0.5增量CRPS选点。两空间使用相同公共参数初始化，额外模式位置参数独立初始化；维度及算力不同，不称严格同参数。
- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第9行：采样为100步余弦扩散的6步DDIM，训练/选择/主评价保持一致；v07的20步结果仅作为历史参考。6步用于允许真实采样链的端到端梯度，不将步数改变混作新机制收益。
- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第11行：每个选定起点复制为两臂，各200步，AdamW0.0002。每步一个随机4小时窗口、K4生成集合，完整反传全部反向扩散步和窗口内自回归历史。窗口初始前小时系数由B-fit训练估计提供（仅训练），首日小时0使用零历史；评价24小时自由滚动，全部历史来自模型。损失为CRPS/10＋可选0.5增量CRPS/10＋0.05 v预测MSE。v项使用相同128训练小时抽样作保持项。两臂共享随机窗口、噪声、起点和更新步数。
- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)，第13行：所有B-fit站参与窗口的观测空间损失，掩码保持；不使用插值格网作为真值。inner每100步选择，包含0步起点。所有配置完成后才读外层评价，K64及独立K128；同时报告不微调起点，点RMSE为采样均值。
- [experiments/us/20261123_diffusion_observation_v09/RESULTS.md](../../us/20261123_diffusion_observation_v09/RESULTS.md)，第5行：## 固定月27日：71站、20108站时，独立K128
- [experiments/us/20261123_diffusion_observation_v09/RESULTS.md](../../us/20261123_diffusion_observation_v09/RESULTS.md)，第17行：点RMSE使用采样均值，原v04为解析均值。K64复算及独立K128均保留。地理最近邻100km以内成对CRPS、海拔分组和负值比例见comparison.json；它们属于开发诊断。
- [experiments/us/20261123_diffusion_observation_v09/RESULTS.md](../../us/20261123_diffusion_observation_v09/RESULTS.md)，第29行：当前是固定已暴露日历上的适应性开发，未读取C/D标签。A输入仅Z0/T/P，历史队列PWV筛选限制仍适用。
- [experiments/us/20261123_diffusion_observation_v09/REVIEW.md](../../us/20261123_diffusion_observation_v09/REVIEW.md)，第1行：# 独立只读复核
- [experiments/us/20261123_diffusion_observation_v09/REVIEW.md](../../us/20261123_diffusion_observation_v09/REVIEW.md)，第3行：2026-11-23，sequence_review对训练实现及完成后的指标、报告、shuffle和验证记录进行了独立只读复核，未发现阻断提交的问题。
- [experiments/us/20261123_diffusion_observation_v09/VERDICT.md](../../us/20261123_diffusion_observation_v09/VERDICT.md)，第7行：2018每月27日共12段24小时，71站、20108站时。下表为两种子均值、独立K128，CRPS越低越好。v04点值为解析均值，扩散点值为采样均值。
- [experiments/us/20261123_diffusion_observation_v09/VERDICT.md](../../us/20261123_diffusion_observation_v09/VERDICT.md)，第21行：观测损失通过完整6步DDIM及4小时历史反传，可以修正仅拟合估计潜系数产生的偏窄分布。24维单独边际CRPS两种子均选零步，而时间分支均选200步；64维单独边际分支仅seed17选200步。因此损失作用与表示、种子有关。
- [experiments/us/20261123_diffusion_observation_v09/VERDICT.md](../../us/20261123_diffusion_observation_v09/VERDICT.md)，第29行：下一轮优先分离均值与随机残差：保持v04均值，把扩散样本中心化，检查能否保留本轮校准/时间收益并减少均值漂移。中心化可能改变覆盖与相关结构，需配对评价，不能预先认定有效。随后再比较训练窗口的自生成历史启动与当前估计系数启动，检验4小时训练到24小时自由滚动的差异。上述为待检验假设，本轮未执行。
- [experiments/us/20261123_diffusion_observation_v09/VERDICT.md](../../us/20261123_diffusion_observation_v09/VERDICT.md)，第37行：使用美国西部当前区域、A174的Z0/T/P与B-fit监督，B-inner选点；未读取C/D标签。月22/27日均为开发暴露日历。历史数据队列曾使用PWV完整性/QC筛选，正式协议仍需按允许变量重建队列。结果针对同小时因果重建，不是未来预报或真实细尺度天气恢复证明。原始数据、权重和生成数组保留本地，仓库同步轻量证据。

## 原始协议、结果与报告

- [experiments/us/20261123_diffusion_observation_v09/PROTOCOL.md](../../us/20261123_diffusion_observation_v09/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_diffusion_observation_v09/RESULTS.md](../../us/20261123_diffusion_observation_v09/RESULTS.md)（report；read）
- [experiments/us/20261123_diffusion_observation_v09/REVIEW.md](../../us/20261123_diffusion_observation_v09/REVIEW.md)（report；read）
- [experiments/us/20261123_diffusion_observation_v09/VERDICT.md](../../us/20261123_diffusion_observation_v09/VERDICT.md)（report；read）
- [experiments/us/20261123_diffusion_observation_v09/comparison.json](../../us/20261123_diffusion_observation_v09/comparison.json)（metadata；read）
- [experiments/us/20261123_diffusion_observation_v09/experiment.lock.json](../../us/20261123_diffusion_observation_v09/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_diffusion_observation_v09/results.json](../../us/20261123_diffusion_observation_v09/results.json)（metadata；read）
- [experiments/us/20261123_diffusion_observation_v09/selection.lock.json](../../us/20261123_diffusion_observation_v09/selection.lock.json)（protocol；read）
- [experiments/us/20261123_diffusion_observation_v09/verification.json](../../us/20261123_diffusion_observation_v09/verification.json)（metadata；read）

## 待核查项

- 无明确状态证据；不由目录名称或指标文件推断完成。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
