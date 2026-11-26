# 20261123_enhanced_encoder_generative_v02

当前公开来源：`experiments/us/20261123_enhanced_encoder_generative_v02`；[来源目录](../../us/20261123_enhanced_encoder_generative_v02)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_enhanced_encoder_generative_v02`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第1行：# 加宽条件编码器及三个独立生成模型

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261123_enhanced_encoder_generative_v02/status.json](../../us/20261123_enhanced_encoder_generative_v02/status.json)，stage：PAUSED

## 报告中的阶段声明

- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第44行：本轮沿用864训练小时和之前已暴露的开发人口；没有新增盲测，D2020继续封存。局部训练阶段通过后，服务器全2016—2018小时及更大集合成员数需要另外锁定数据与预算。本机顺序执行，60分钟总墙钟门槛，失败、暂停及预算耗尽分别记状态。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第3行：Task1 complete: 契约测试先失败后实现；当前16项通过，包括训练资格不读取留出标签。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第5行：Task2 complete: 独立三模型实现，不依赖历史VAE解码；编码器32/64/128、解码32/16。两种子加宽均值训练完成，375473参数。加宽并未在所有已暴露开发人口优于小模型，保留对照，不宣称容量收益已成立。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第7行：Task3 complete: 真实GPU短测，三模型梯度有限、短程优化器保存恢复参数差0；文件锁、累计预算、待评价恢复、产物哈希及独立点/联合检查点已实现。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第11行：Ruling: v01 VAE完成600步ELBO后在ES阶段遇到一个无合格拟合B站的小时。仅按训练资格筛选，864→863小时，三模型同一清单；评价人口不变。v02重新初始化生成模型，不继承失败尝试的VAE权重；228.56秒既往计算计入60分钟总门槛。旧批次保留追溯。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第13行：Task4 complete: 标签隔离和格网查询沿用已审计数据；独立审阅补齐了实际缓存SHA检查。活动v02进程27696已真实运行VAE超过800步，ES微调评价完成、点与联合检查点均已保存，错误日志为空。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第17行：Task6 engineering review complete: 独立审阅未发现剩余P1阻断；需要保留观测分布/解码场分布及指标宏微平均的解释。最终精度结论等待本轮训练完成。

## 历史证据范围与信息条件

- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第1行：# 加宽条件编码器及三个独立生成模型
- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第3行：2026-11-23，本机完整网络训练批次。当前输入仍为用户确认的现有A站GNSS与ERA5站位采样气象量，不另称为独立完整天气格网。
- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第7行：42通道：四帧五变量粗格20通道、四帧支持标记4、四帧GNSS数值/密度散布8、地形与位置6、时间周期4。发布144×96公共高度格网；预测站值统一为双线性插值后乘exp(-H/3000)。不使用历史学习底座或空间基。
- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第38行：当前没有稠密干净目标，因此本轮是“完整采样链＋真实B站ES/CRPS”的观测评分训练，不执行或声称已执行标准全场DSM。网络、求解器、初始噪声均保留真实采样过程，梯度穿过全部11次调用。固定站监督仍不能唯一约束无站区域；空间结构效果由独立站组与发布格检验。
- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第42行：两个独立选模产品：点值最优、站组ES最优。分别报告各自同一检查点的RMSE、MAE、偏差、CRPS、ES、VS、区间覆盖及宽度；加入逐站打乱成员的空间依赖对照。VAE最终使用先验自由采样，不用后验重构代替部署结果。
- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)，第44行：本轮沿用864训练小时和之前已暴露的开发人口；没有新增盲测，D2020继续封存。局部训练阶段通过后，服务器全2016—2018小时及更大集合成员数需要另外锁定数据与预算。本机顺序执行，60分钟总墙钟门槛，失败、暂停及预算耗尽分别记状态。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第5行：Task2 complete: 独立三模型实现，不依赖历史VAE解码；编码器32/64/128、解码32/16。两种子加宽均值训练完成，375473参数。加宽并未在所有已暴露开发人口优于小模型，保留对照，不宣称容量收益已成立。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第7行：Task3 complete: 真实GPU短测，三模型梯度有限、短程优化器保存恢复参数差0；文件锁、累计预算、待评价恢复、产物哈希及独立点/联合检查点已实现。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第13行：Task4 complete: 标签隔离和格网查询沿用已审计数据；独立审阅补齐了实际缓存SHA检查。活动v02进程27696已真实运行VAE超过800步，ES微调评价完成、点与联合检查点均已保存，错误日志为空。
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)，第17行：Task6 engineering review complete: 独立审阅未发现剩余P1阻断；需要保留观测分布/解码场分布及指标宏微平均的解释。最终精度结论等待本轮训练完成。

## 原始协议、结果与报告

- [experiments/us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v02/ARCHITECTURE.md)（documentation；read）
- [experiments/us/20261123_enhanced_encoder_generative_v02/generators.lock.json](../../us/20261123_enhanced_encoder_generative_v02/generators.lock.json)（protocol；read）
- [experiments/us/20261123_enhanced_encoder_generative_v02/progress.md](../../us/20261123_enhanced_encoder_generative_v02/progress.md)（documentation；read）
- [experiments/us/20261123_enhanced_encoder_generative_v02/status.json](../../us/20261123_enhanced_encoder_generative_v02/status.json)（status；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
