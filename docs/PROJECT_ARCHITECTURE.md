# 工程与研究结构

核心工程包`src/gnss_field`提供模型、数据、训练、评价与CLI。`research/engine/tools`保留研究算法及实验入口；各有日期的research子目录保留欧洲、联合生成和其他研究实现。

`experiments`是轻量证据层，按区域与实验编号组织，并提供逐实验说明。`paper`是当前科学结论层，说明架构、模块贡献、评价范围与待补证据。`docs/research_notes`保留有日期的历史研究判断。

`artifacts`是本地存储层，包含原始数据、唯一权重、预测数组和冻结运行记录；`archive`保存历史兼容源码及旧导航。历史绝对路径通过兼容映射读取，不覆盖旧科学锁。

活动代码、公开轻量结果和本地运行记录的对应关系见[迁移映射](../maintenance/paper_restructure_20261126/path_mapping.json)。[模型流程](../paper/ARCHITECTURE.md)给出实际网络结构。
