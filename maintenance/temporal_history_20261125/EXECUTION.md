# 执行记录

计划：docs/superpowers/plans/2026-11-25-spatiotemporal-diffusion-implementation-plan.md。
执行权威修订：docs/NEXT_STAGE.md 及长历史数据组织审阅。

- Ruling: 使用 codex/temporal-data-experiments 分支在现有数据工作区实施，避免复制大型本地权重及改变原始路径；历史数据包不修改。持续推送授权已存在。
- Ruling: 初轮先构建连续时间数据并完成确定性时间对照；diffusion 在基础场和折外残差接口验证后接入，不以旧p14的10.8mm冒充未来预报成绩。
- Ruling: 使用产品名义小时，原始头为 UNSPECIFIED 时保持该声明，不自行猜测 UTC/GPST。
- P0：数据模块缺失时5个测试失败；实现后通过。连续数据已重建为43848×410×3，另有8784×137×3开发数据；海岸源sha256与原始允许变量QC均已归档。
- P1/P4：时间编码器缺失时9个测试失败；实现后通过。追加精确持久性跳连测试先失败后通过。评分宏/微区分及缺失目标测试先失败后通过。新代码及原固定支撑相关测试共20项通过。
- 工程基线：全工程567通过、1失败（tests/test_unet_foundation.py::test_grid_readout_and_gradient，之前测试遗留strict deterministic导致CUDA grid_sample backward拒绝）。该测试单独运行通过；不将此状态称为全套通过。
- 制包第一次在海岸下载网络断开时失败；显式下载同一来源后重跑成功，增加原始QC阶段收据。未修改历史数据包。
- Ruling: 当前先实现8×8固定支撑、小型空间U-Net式跳连、C1查询点海岸输入及精确点IDW持久性跳连；C2边海岸、完整地形FiLM与p14权重迁移另做消融，避免把原型预报与p14空间重建混排。
- 运行中：2016–2018拟合、2019选择更新数、从头重拟合2016–2019、2020抽样开发评价；5种模型与12/48/72h两种子。2步GRU/Transformer真实训练与预测smoke已完成。
