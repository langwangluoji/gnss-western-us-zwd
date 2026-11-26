# 20261123_enhanced_encoder_generative_v03

当前公开来源：`experiments/us/20261123_enhanced_encoder_generative_v03`；[来源目录](../../us/20261123_enhanced_encoder_generative_v03)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_enhanced_encoder_generative_v03`。

用途类别：`diagnostic`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/diagnosis.lock.json](../../us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/diagnosis.lock.json)，purpose：Post hoc fixed-panel optimization and fit diagnosis; no checkpoint selection; C/D not scored

## 类别依据

- [experiments/us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/RESULTS.md)，第1行：# 三类模型短训：优化与拟合诊断

## 状态依据

- [experiments/us/20261123_enhanced_encoder_generative_v03/status.json](../../us/20261123_enhanced_encoder_generative_v03/status.json)，stage：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第44行：本轮从864个候选训练小时中使用863个合格小时，沿用之前已暴露的开发人口；没有新增盲测，D2020继续封存。局部训练阶段通过后，服务器全2016—2018小时及更大集合成员数需要另外锁定数据与预算。本机顺序执行，60分钟总墙钟门槛，失败、暂停及预算耗尽分别记状态。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第5行：- 加宽条件编码器32/64/128，保持解码器32/16；两个种子各1000步已完成。两个种子的外部表现尚未证明加宽编码器具有稳定优势，不能把容量增加本身当作精度收益。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第6行：- VAE：600步观测ELBO＋1200步观测评分，1800步已完成；四组人口、两个独立产品、K32评价已完成。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第7行：- GAN：1600步已完成；四组人口、两个独立产品、K32评价已完成。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第8行：- Diffusion：完整格网6步Heun、11次去噪调用；1000步及四组人口、两个产品K32评价全部完成。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第17行：2026-11-23追加固定样本拟合诊断已完成：24个训练时刻与24个内部留出天气时刻，分别查询拟合站和内部留出站；初始化、点值产品、联合产品、最终权重均自由采样K8。重复计算的内部RMSE、CRPS、ES、站组ES与归档结果差均在0.00001 mm内。C48三模型的时间、站序、标签、支持掩码一致，共3675个有效站时刻。训练和诊断累计计入2378.485秒，约39.6分钟；当前队列已结束。研究判断见fit_diagnosis/INTERPRETATION.md；完整结果表和图见RESULTS.md、figures/及fit_diagnosis/。

## 历史证据范围与信息条件

- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第1行：# 加宽条件编码器及三个独立生成模型
- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第3行：2026-11-23，本机完整网络训练批次。当前输入仍为用户确认的现有A站GNSS与ERA5站位采样气象量，不另称为独立完整天气格网。
- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第7行：42通道：四帧五变量粗格20通道、四帧支持标记4、四帧GNSS数值/密度散布8、地形与位置6、时间周期4。发布144×96公共高度格网；预测站值统一为双线性插值后乘exp(-H/3000)。不使用历史学习底座或空间基。
- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第38行：当前没有稠密干净目标，因此本轮是“完整采样链＋真实B站ES/CRPS”的观测评分训练，不执行或声称已执行标准全场DSM。网络、求解器、初始噪声均保留真实采样过程，梯度穿过全部11次调用。固定站监督仍不能唯一约束无站区域；空间结构效果由独立站组与发布格检验。
- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第42行：两个独立选模产品：点值最优、站组ES最优。分别报告各自同一检查点的RMSE、MAE、偏差、CRPS、ES、VS、区间覆盖及宽度；加入逐站打乱成员的空间依赖对照。VAE最终使用先验自由采样，不用后验重构代替部署结果。
- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)，第44行：本轮从864个候选训练小时中使用863个合格小时，沿用之前已暴露的开发人口；没有新增盲测，D2020继续封存。局部训练阶段通过后，服务器全2016—2018小时及更大集合成员数需要另外锁定数据与预算。本机顺序执行，60分钟总墙钟门槛，失败、暂停及预算耗尽分别记状态。
- [experiments/us/20261123_enhanced_encoder_generative_v03/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/RESULTS.md)，第1行：# 独立GAN、Diffusion与VAE本机训练结果
- [experiments/us/20261123_enhanced_encoder_generative_v03/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/RESULTS.md)，第3行：当前批次使用863个合格训练小时，三模型分别完整更新；不是服务器全量小时实验。D2020未读取。
- [experiments/us/20261123_enhanced_encoder_generative_v03/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/RESULTS.md)，第18行：C2019仅48个开发时刻，非全年、非盲测。K32集合尚未达到服务器K64/128稳定性复核规模。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第1行：# 加强编码器与独立三模型训练进度
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第6行：- VAE：600步观测ELBO＋1200步观测评分，1800步已完成；四组人口、两个独立产品、K32评价已完成。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第7行：- GAN：1600步已完成；四组人口、两个独立产品、K32评价已完成。
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)，第11行：数据：864个原始候选训练小时中863个满足至少4个有效拟合B站；空间48小时评价从原始864小时时间表固定选取，不随训练资格变化。内部24小时K8选模；C2019为48小时已暴露开发集，D2020保持封存。当前为完整网络的本机训练，不是全年小时的服务器全数据训练。

## 原始协议、结果与报告

- [experiments/us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md](../../us/20261123_enhanced_encoder_generative_v03/ARCHITECTURE.md)（documentation；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/RESULTS.md)（report；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/generators.lock.json](../../us/20261123_enhanced_encoder_generative_v03/generators.lock.json)（protocol；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/progress.md](../../us/20261123_enhanced_encoder_generative_v03/progress.md)（documentation；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/status.json](../../us/20261123_enhanced_encoder_generative_v03/status.json)（status；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/RESULTS.md](../../us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/RESULTS.md)（report；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/diagnosis.lock.json](../../us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/diagnosis.lock.json)（protocol；read）
- [experiments/us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/results.json](../../us/20261123_enhanced_encoder_generative_v03/fit_diagnosis/results.json)（metadata；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
