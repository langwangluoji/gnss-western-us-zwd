# 本地数据与恢复

详见[artifacts说明](../artifacts/README.md)和[2026-11-27精简清单](../maintenance/compact_20261127/README.md)。原始观测、核心模型及其完整缓存保留本地；退役模型、拟合参数和小数组归档到GitHub Release，非核心大数组按清单退役。科学锁、指标、图表和负结果保留原文，公开Git不含GB权重或原始数据。

迁移前完整路径及哈希记录在本地`maintenance/paper_restructure_20261126/inventory_before.local.csv`；精简前清单、逐实验存储状态、云端对象索引与执行回执在`maintenance/compact_20261127`随Git发布。旧科学锁保存原始身份，新路径由兼容映射提供。旧完整回放若使用已退役大数组，需按原合同重新生成。

此前2026-11-23归档仍在项目外同盘`../GAN对流层_历史归档_20261123/`，本次不修改它。
