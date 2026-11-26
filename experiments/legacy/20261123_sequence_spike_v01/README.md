# 20261123_sequence_spike_v01

当前公开来源：`experiments/us/20261123_sequence_spike_v01`；[来源目录](../../us/20261123_sequence_spike_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_sequence_spike_v01`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)，第1行：# 连续条件随机场机制小试（预登记）

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)，第19行：网页复核：browser-collab初始化失败，错误为“failed to write kernel assets: 系统找不到指定的路径”。未发送材料；客户端原始来源核验继续。

## 历史证据范围与信息条件

- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)，第7行：输入仅取现有数据包的A Z0、温度、气压。Tm/PWV不参与新模型、基线、归一化及新有效性计算；从头训练，不继承旧权重。历史数据包上游曾以五变量完整性/PWV范围筛选样本和站网，因此本轮属于固定历史队列上的禁用输入小试，不能称为完全独立于PWV的原始数据协议。正式实验前须重建逐变量质控。
- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)，第9行：先验冻结：2016/2017每月8、18、25日各连续24小时训练；2018每月1日内部选择；2018每月16日外层诊断；每段额外6小时热身，评分只用最后24小时。不读取C/D标签。B357/77/77沿用既有整50km组划分；该划分不是大区域连续留出，外层已历史开发暴露，不称最终盲测。
- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)，第13行：生成机制：冻结GRU均值和特征；神经网络输出8个线性与8个中心化二次潜变量基系数。共享空间潜变量形成相关场，独立时间噪声与可学习AR(1)隐状态作配对。两臂均以公平CRPS及一小时增量CRPS、站点对差值CRPS训练。二次项提供非高斯残差，但不是GAN、VAE或Diffusion；作为三类模型需要超越的轻量机制基线。解析残差期望为零，有限集合不强制去均值。
- [experiments/us/20261123_sequence_spike_v01/RESEARCH.md](../../us/20261123_sequence_spike_v01/RESEARCH.md)，第21行：输入应显式携带观测值、有效掩码、相对位置、高差与时间间隔；距离核/集合编码避免把稀疏观测先压成一张光滑图后丢失站网信息。动态气象分支与静态地形分支分开，再在多个空间尺度融合。时间编码器需要检验未来扰动不影响过去输出，历史缺测需要掩码而不是零值伪观测。
- [experiments/us/20261123_sequence_spike_v01/RESEARCH.md](../../us/20261123_sequence_spike_v01/RESEARCH.md)，第39行：旧气象U-Net采用Z0/T/P/Tm/PWV，其中ERA5 PWV是水汽柱信息。新增用户约束下，新实验只保留Z0/T/P，Tm也暂不使用。旧权重不迁移，归一化重算。
- [experiments/us/20261123_sequence_spike_v01/RESEARCH.md](../../us/20261123_sequence_spike_v01/RESEARCH.md)，第41行：原始制包程序`field_generation_v1/prepare.py`还以joint_valid、五变量有限性和ERA5PWV范围筛选整行，并基于覆盖率选择站网。新实验未使用PWV数值，但沿用该历史队列。正式论文前应从现有原始文件重建仅依赖允许变量和GNSS质量标志的质控；保留A/B/C/D角色，报告可用样本变化，不悄悄重新选一个更有利站网。

## 原始协议、结果与报告

- [experiments/us/20261123_sequence_spike_v01/PROTOCOL.md](../../us/20261123_sequence_spike_v01/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_sequence_spike_v01/RESEARCH.md](../../us/20261123_sequence_spike_v01/RESEARCH.md)（documentation；read）
- [experiments/us/20261123_sequence_spike_v01/results.json](../../us/20261123_sequence_spike_v01/results.json)（metadata；read）
- [experiments/us/20261123_sequence_spike_v01/verification.json](../../us/20261123_sequence_spike_v01/verification.json)（metadata；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
