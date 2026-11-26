# 20261123_terrain_diffusion_v10

当前公开来源：`experiments/us/20261123_terrain_diffusion_v10`；[来源目录](../../us/20261123_terrain_diffusion_v10)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_terrain_diffusion_v10`。

用途类别：`engineering`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第1行：# v10 地形时序图扩散：实现与真实本机筛查

## 类别依据

- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第40行：## 工程验证

## 状态依据

- [experiments/us/20261123_terrain_diffusion_v10/query_demo/metrics.json](../../us/20261123_terrain_diffusion_v10/query_demo/metrics.json)，status：joint_query_generation

## 报告中的阶段声明

- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第3行：状态：**P0/P1工程验证与两种子缩减筛查完成；正式P2、P3及全量训练待推进。**
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第5行：1. P1：非有限预测参与人口过滤。已改为真实标签定义固定评分人口，人口内任何非有限预测直接报错；回归测试先失败再通过。八组真实预测全部有限，修复后指标复算差0。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第6行：2. P2：时序微调遇到整小时无标签时空损失。已保持该小时生成与历史推进，跳过缺失监督项和空节点辅助v图；缺测小时完整反向测试先失败再通过。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第7行：3. P2：encoder/initial只有路径没有源权重哈希。新增checkpoint_receipt，记录SHA256、选中步和来源配置；回归测试先失败再通过。修复前已完成的筛查以postrun_audit旁证补记当前本机源权重，保留原config不改写为事先记录。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第8行：4. P2：报告固定加载默认数据。改为每个run按config数据路径加载并核验数据哈希，测试先失败再通过。独立评价同样检查训练权重的数据合同。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第12行：审查确认没有发现隐藏目标当前标签进入条件编码器；条件/目标物理组分离，推理使用生成历史。所有原型配置差别、D1额外编码器训练成本、K4概率评价局限、短日历范围已写入README和RESULTS。正式P2/P3、全季节、多密度、查询扩展及高分辨率稀疏算子不是本轮已完成证据。

## 历史证据范围与信息条件

- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第5行：已构建直接地表ZWD图扩散、条件因果Transformer、生成历史GRU、MAE式遮站编码器迁移，以及完整可微的4小时生成损失。网络384,682参数。代码、训练命令与后续预算见训练说明。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第9行：允许变量QC直接从547个原始测站文件重建，只使用ZWD/T/P。410训练站提供2016/17的72段数据；137验证站提供2018月1日标签；138确认站及原C/D标签未读取。训练缓存有效率97.97%，验证标签有效率97.86%。原始队列继承历史PWV完整率筛选，记录层独立QC不改变这段站点选择历史。来源时制字段本次为UNSPECIFIED，统一按产品名义小时对齐。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第32行：对种子42的B1与D1，额外使用独立噪声种子19007、K8、两个完整24小时和三布局，保持137目标站固定。结果为：B1宏RMSE16.606mm、CRPS13.346mm；D1宏RMSE19.801mm、CRPS11.992mm。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第34行：D1的一小时增量CRPS为3.806mm；逐小时独立打乱成员编号后为4.281mm，增加0.475mm。这个单种子诊断支持继续检验成员时间关联。它不证明细尺度天气过程已被正确恢复，后续需要两种子、更多月份及高斯对照的同口径时间检验。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第38行：右图固定为元数据顺序首站ALBH，不按误差挑选。全程使用成员自身历史，没有验证站真实历史暖启动。图中既有概率演变，也显示该站趋势幅度尚需改善。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第42行：- 12项测试通过：允许列独立QC、角色访问隔离、v参数反演、因果性、目标置换、采样梯度、已知人口指标、时序梯度、非有限预测拒绝、全小时缺测、权重来源、报告数据合同及隐藏标签隔离等（多个性质在同一测试内）。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第44行：- 完整4h × K4 × DDIM20、16目标和82条件站：前后向7.68秒，峰值显存1920.4MiB，梯度有限。没有截断主轨迹历史梯度。硬件RTX4060 Laptop GPU。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第46行：- 独立实现审查的四项问题已修复，见审查记录。原始数据、权重与预测数组保留本机；公开仓库同步代码、配置、指标、曲线及来源合同。
- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)，第54行：本轮原型选择：32查询、随机条件布局、固定学习率、EMA0.95、FP32；条件图最近12点、均值/最大值全局摘要、无支撑时最近点回退；地形为现有25km最近节点。这些简化使本机筛查可复现，但还没有检验正式方案的空间混合抽样和更丰富的边消息机制。当前注意力内部为稠密矩阵加邻接掩码，大规模高分辨率查询需要真正的稀疏算子；不能用独立小块采样替代联合场。海岸距离、密度敏感性、查询集扩展与72小时滚动列入后续验收。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第1行：# 独立实现审查与修复
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第6行：2. P2：时序微调遇到整小时无标签时空损失。已保持该小时生成与历史推进，跳过缺失监督项和空节点辅助v图；缺测小时完整反向测试先失败再通过。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第8行：4. P2：报告固定加载默认数据。改为每个run按config数据路径加载并核验数据哈希，测试先失败再通过。独立评价同样检查训练权重的数据合同。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第10行：此外修复Windows默认编码读取UTF-8配置造成中文路径损坏的问题，真实报告生成已重跑成功。梯度检查点捕获可变历史字典曾触发重算不一致；采样入口固定字典绑定，时序梯度测试通过。
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)，第12行：审查确认没有发现隐藏目标当前标签进入条件编码器；条件/目标物理组分离，推理使用生成历史。所有原型配置差别、D1额外编码器训练成本、K4概率评价局限、短日历范围已写入README和RESULTS。正式P2/P3、全季节、多密度、查询扩展及高分辨率稀疏算子不是本轮已完成证据。

## 原始协议、结果与报告

- [experiments/us/20261123_terrain_diffusion_v10/RESULTS.md](../../us/20261123_terrain_diffusion_v10/RESULTS.md)（report；read）
- [experiments/us/20261123_terrain_diffusion_v10/REVIEW.md](../../us/20261123_terrain_diffusion_v10/REVIEW.md)（report；read）
- [experiments/us/20261123_terrain_diffusion_v10/b1_seed42/config.json](../../us/20261123_terrain_diffusion_v10/b1_seed42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/b1_seed42/metrics.json](../../us/20261123_terrain_diffusion_v10/b1_seed42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/overfit/config.json](../../us/20261123_terrain_diffusion_v10/overfit/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/overfit/metrics.json](../../us/20261123_terrain_diffusion_v10/overfit/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/query_demo/config.json](../../us/20261123_terrain_diffusion_v10/query_demo/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/query_demo/metrics.json](../../us/20261123_terrain_diffusion_v10/query_demo/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/summary.json](../../us/20261123_terrain_diffusion_v10/screen/summary.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b1_s17/config.json](../../us/20261123_terrain_diffusion_v10/screen/b1_s17/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b1_s17/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/b1_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b1_s42/config.json](../../us/20261123_terrain_diffusion_v10/screen/b1_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b1_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/b1_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b2_s17/config.json](../../us/20261123_terrain_diffusion_v10/screen/b2_s17/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b2_s17/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/b2_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b2_s42/config.json](../../us/20261123_terrain_diffusion_v10/screen/b2_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/b2_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/b2_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d0_s17/config.json](../../us/20261123_terrain_diffusion_v10/screen/d0_s17/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d0_s17/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/d0_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d0_s42/config.json](../../us/20261123_terrain_diffusion_v10/screen/d0_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d0_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/d0_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d1_s17/config.json](../../us/20261123_terrain_diffusion_v10/screen/d1_s17/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d1_s17/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/d1_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d1_s42/config.json](../../us/20261123_terrain_diffusion_v10/screen/d1_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/screen/d1_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/screen/d1_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/validation24_b1_s42/config.json](../../us/20261123_terrain_diffusion_v10/validation24_b1_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/validation24_b1_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/validation24_b1_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/validation24_d1_s42/config.json](../../us/20261123_terrain_diffusion_v10/validation24_d1_s42/config.json)（metadata；read）
- [experiments/us/20261123_terrain_diffusion_v10/validation24_d1_s42/metrics.json](../../us/20261123_terrain_diffusion_v10/validation24_d1_s42/metrics.json)（metadata；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
