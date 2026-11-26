# 本地产物位置

此目录的大文件保留本机，Git只保存本说明。

- `legacy_layout/research_pipeline/`：原美国数据、完整运行记录、权重、预测与历史支持源。内部保留原相对层级；活动代码已移到根src及research/engine。
- `campaigns/europe_transfer_20261126/`：欧洲迁移数据缓存及已有模型。
- `campaigns/europe_vmf3_20261126/`：VMF3特征、权重、概率模型与全年采样数组。

研究目录中的cache/runs/probability/evaluation以及旧路径为本地兼容junction。公开轻量结果分别在experiments/us、experiments/europe与research研究报告中。

外部欧洲Model4来源仍保留在旁边项目原目录，本次没有移动或重训原CNN。复制到另一台机器时按迁移清单恢复文件并核对SHA256，然后执行`python -m research.layout`。

删除项只有已证明在完整data.npz中逐字节保留的两个重复输入数组；提取其同名NPY成员即可恢复。原始数据、唯一权重和正式评价样本完整保留。
