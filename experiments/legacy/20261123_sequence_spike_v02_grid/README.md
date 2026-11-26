# 20261123_sequence_spike_v02_grid

当前公开来源：`experiments/us/20261123_sequence_spike_v02_grid`；[来源目录](../../us/20261123_sequence_spike_v02_grid)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_sequence_spike_v02_grid`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_sequence_spike_v02_grid/PROTOCOL.md](../../us/20261123_sequence_spike_v02_grid/PROTOCOL.md)，第1行：# 连续场机制小试：统一格网算子确认

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第41行：浏览器协作：指定内置浏览器后仍在工具初始化阶段失败；open_in_codex只返回queued，未观察到成功打开或模型回答。原始文献检索已完成，网页独立复核未完成。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第43行：成功运行耗时93.1秒（不含失败尝试和后验核查）；源码、配置、权重哈希与验证明细见同目录JSON。

## 历史证据范围与信息条件

- [experiments/us/20261123_sequence_spike_v02_grid/PROTOCOL.md](../../us/20261123_sequence_spike_v02_grid/PROTOCOL.md)，第7行：这是一轮基于算子审计触发的工程确认，不增加外层选型轮次；保留v01全部结果。沿用两种子500步均值与350步随机生成训练。PWV上游筛选的历史队列限制仍保留，C/D标签不读取。
- [experiments/us/20261123_sequence_spike_v02_grid/PROTOCOL.md](../../us/20261123_sequence_spike_v02_grid/PROTOCOL.md)，第9行：新增诊断：逐时独立打乱成员编号，严格保持每个时刻边缘集合与空间联合分布，破坏跨时间耦合；比较一小时增量CRPS，检验是否真正利用了时间结构。此诊断仅在训练完成后执行，不参与调参。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第3行：2026-11-23。RTX4060真实数据训练；两种子各500步逐帧/GRU均值、350步独立/相关随机场。此轮为机制筛查，尚未验证最终GAN、Diffusion或VAE。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第5行：输入仅A站Z0、温度、气压；从头训练。数据包历史PWV筛选限制仍保留，详见 ../20261123_sequence_spike_v01/RESEARCH.md。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第20行：- 种子42：增量CRPS相对改善8.18%；等天气块差值95%区间[-0.3555, -0.1624] mm。独立噪声K128另见results.json。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第22行：固定站点编号相邻配对的空间差值CRPS：独立6.6796，相关6.4954 mm；不是地理近邻或地形分层证据。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第23行：- 种子17：增量CRPS相对改善6.97%；等天气块差值95%区间[-0.3115, -0.1260] mm。独立噪声K128另见results.json。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第25行：固定站点编号相邻配对的空间差值CRPS：独立6.8224，相关6.6706 mm；不是地理近邻或地形分层证据。
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)，第41行：浏览器协作：指定内置浏览器后仍在工具初始化阶段失败；open_in_codex只返回queued，未观察到成功打开或模型回答。原始文献检索已完成，网页独立复核未完成。
- [experiments/us/20261123_sequence_spike_v02_grid/REVIEW.md](../../us/20261123_sequence_spike_v02_grid/REVIEW.md)，第1行：# 独立只读审查
- [experiments/us/20261123_sequence_spike_v02_grid/REVIEW.md](../../us/20261123_sequence_spike_v02_grid/REVIEW.md)，第9行：复审建议进一步独立回查全格网序列与适配器的站点值，已加入full_grid_station_roundtrip_max_mm检查；结果以verification.json为准。

## 原始协议、结果与报告

- [experiments/us/20261123_sequence_spike_v02_grid/PROTOCOL.md](../../us/20261123_sequence_spike_v02_grid/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_sequence_spike_v02_grid/RESULTS.md](../../us/20261123_sequence_spike_v02_grid/RESULTS.md)（report；read）
- [experiments/us/20261123_sequence_spike_v02_grid/REVIEW.md](../../us/20261123_sequence_spike_v02_grid/REVIEW.md)（report；read）
- [experiments/us/20261123_sequence_spike_v02_grid/results.json](../../us/20261123_sequence_spike_v02_grid/results.json)（metadata；read）
- [experiments/us/20261123_sequence_spike_v02_grid/verification.json](../../us/20261123_sequence_spike_v02_grid/verification.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
