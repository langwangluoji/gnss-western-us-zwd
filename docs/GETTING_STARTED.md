# 快速上手

在仓库根目录运行。需要Python≥3.11、PyTorch2.8.x；本机环境为Anaconda的`pytorch`。

```powershell
python -m pip install -e ".[dev]"
python -m research.layout
python -m gnss_field --help
python -m pytest -q
```

`research.layout`建立可重建的本地兼容目录：旧研究路径映射到新源代码和本地产物；Windows使用目录junction，其他平台使用相对符号链接。它不会训练、改实验锁或复制大数据。公开克隆只含代码与轻量证据；核心模型需要本地artifacts，退役权重可从Release恢复，退役大数组需按历史流程重新生成，见[精简与恢复说明](../maintenance/compact_20261127/README.md)。

主包在`src/gnss_field`，研究工具和568项工程测试在`research/engine`；欧洲研究及其30项测试在`research/europe_*`。根pytest显式收集这些测试与整理回归测试，避免扫描历史运行中保存的多份源码快照。

当前完整本机套件已验证611项通过，见[本轮测试回执](../maintenance/compact_20261127/postdelete_tests.json)。公开克隆缺少观测数据时，标记为`real_data`的CUDA观测检查会明确跳过。历史模型一致性检查使用随仓库保留的兼容源，正常克隆也应执行。

先读[结果指南](../paper/README.md)、[模块贡献](../paper/MODULE_CONTRIBUTIONS.md)、[架构](../paper/ARCHITECTURE.md)和[实验目录](../experiments/README.md)。本地数据恢复见[产物说明](../artifacts/README.md)。

历史版本复现先检出`archive/pre-paper-restructure-20261126`到独立目录，再恢复登记的数据与权重；不要将新路径写回旧实验锁。
