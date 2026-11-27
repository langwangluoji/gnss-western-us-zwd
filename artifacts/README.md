# 本地产物位置

此目录保留核心数据与模型，Git只保存本说明。现行保留与退役规则见[数据精简说明](../maintenance/compact_20261127/README.md)。

- `legacy_layout/research_pipeline/`：美国原始数据、I1/joint及关键对照、最新概率融合与其完整依赖；保留旧运行的轻量记录，非核心大数组按清单退役。内部保留原相对层级；活动代码已移到根src及research/engine。
- `campaigns/europe_transfer_20261126/`：欧洲迁移数据缓存及已有模型。
- `campaigns/europe_vmf3_20261126/`：VMF3特征、权重、概率模型与全年采样数组。

研究目录中的cache/runs/probability/evaluation以及旧路径为本地兼容junction。公开轻量结果分别在experiments/us、experiments/europe与research研究报告中。

外部欧洲Model4来源仍保留在旁边项目原目录，本次没有移动或重训原CNN。复制到另一台机器时按迁移清单恢复文件并核对SHA256，然后执行`python -m research.layout`。

2026-11-26只删除了两个可由data.npz恢复的重复数组；2026-11-27按新授权精简退役实验，具体文件与SHA256见[逐文件清单](../maintenance/compact_20261127/inventory.csv)。核心评价样本保留；旧权重、拟合参数与小数组从[Release](https://github.com/langwangluoji/gnss-western-us-zwd/releases/tag/archive%2Fpre-data-compaction-20261127)恢复。退役的大型评估、embedding和训练缓存需按历史流程重新生成，Git标签不含这些数组。
