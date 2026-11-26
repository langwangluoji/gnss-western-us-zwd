# 20261123_diffusion_v07

当前公开来源：`experiments/us/20261123_diffusion_v07`；[来源目录](../../us/20261123_diffusion_v07)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_diffusion_v07`。

用途类别：`diagnostic`。状态索引：`no_evidence`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第1行：# 观测约束残差潜空间扩散：完整原型架构

## 类别依据

- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)，第50行：## 时间成员对应诊断

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

未定位到明确证据。

## 历史证据范围与信息条件

- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第9行：A[A174当前及历史Z0/T/P] --> B[多尺度MLP背景及历史创新校正 v04]
- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第25行：冻结v03逐帧多尺度均值和v04当前/1h/3h创新ridge，不使用旧含PWV权重。共同高度格网均值记为mu_t，表面观测算子H_s包括四角双线性插值及exp(-height/3000)。输出Z0_t=mu_t+Phi z_t，站点输出H_s Z0_t。
- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第51行：训练使用B-fit系数的前小时值，采样使用自己已生成的前小时值，构成p(z_1|c_1)∏p(z_t|z_(t-1),c_t)。它在物理时间上因果，属于同小时条件重建，不是未来天气预测。自由滚动与教师强制的差异由24h生成评分检验。首小时使用零历史并显式标记。
- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第53行：采样从标准Gaussian开始，使用预先固定20个DDIM步，eta=0；标准化z_hat截断至[-6,6]用于数值稳定，并报告触及比例。每个成员独立噪声，空间查询共享同一系数；时间连续性由前一小时条件学习，而不是事后平滑。
- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第67行：两种子42/17，四种模型各1500更新、batch128；每300步以B-inner自由滚动K16的CRPS+0.5增量CRPS选点。无B-inner潜目标拟合。八次训练完成后评月22/27日固定人口，K64与独立K128。均值报告采样均值并说明Monte Carlo误差，不能冒充解析均值。检查扩散恒等式、零噪声端点、因果条件、无B输入、站点/格网一致、来源哈希。
- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)，第69行：8mm是固定人口的均值目标；扩散的独立价值还必须体现为CRPS、时间增量评分和覆盖率。不能用更丰富纹理代替准确性或校准证据。当前历史数据队列仍受PWV筛选影响，正式论文协议须重建允许变量质控。
- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)，第7行：2018年每月27日，71有效站、20108站时；下表采用独立噪声K128。点RMSE使用生成集合均值；v04为解析均值，有限成员误差需结合K64检查。
- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)，第33行：ridge系数是B-fit残差的受限子空间估计，不是稠密真值或潜变量后验样本。解码器所不能表示的残差没有独立新增噪声分支，覆盖率可暴露这一限制。此候选筛查不能直接证明未观测细节恢复。
- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)，第35行：不读取C/D；输入仅A的Z0/T/P，历史制包PWV筛选限制继续适用。月22/27日均属已暴露开发日历，不作最终盲测。
- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)，第39行：## 独立集合复核
- [experiments/us/20261123_diffusion_v07/REVIEW.md](../../us/20261123_diffusion_v07/REVIEW.md)，第1行：# 独立审查
- [experiments/us/20261123_diffusion_v07/REVIEW.md](../../us/20261123_diffusion_v07/REVIEW.md)，第7行：检查24模式token与锚点/历史条件维度，top2路由归一化及负载平衡项。各模型匹配更新数、batch与抽样流，不把不同架构的同种子误称严格同权重起点；稀疏原型实际计算全部专家，不宣称加速。
- [experiments/us/20261123_diffusion_v07/VERDICT.md](../../us/20261123_diffusion_v07/VERDICT.md)，第5行：月27日独立K128，普通DiT宏RMSE为9.1606/9.1594mm，原v04为9.0099/8.9959mm。稀疏DiT为9.3445/9.1225mm，没有跨种子一致优势。普通DiT约10.4万参数，稀疏DiT约20.4万；本轮不优先增加专家规模。
- [experiments/us/20261123_diffusion_v07/VERDICT.md](../../us/20261123_diffusion_v07/VERDICT.md)，第9行：用户随后允许增加输入数据量，已据此启动独立v08样本/步数配对；A174保持不变。原架构锁与此轮结果保持原样。后续训练不能以更密格网或更丰富纹理替代概率校准与固定人口精度。

## 原始协议、结果与报告

- [experiments/us/20261123_diffusion_v07/ARCHITECTURE.md](../../us/20261123_diffusion_v07/ARCHITECTURE.md)（documentation；read）
- [experiments/us/20261123_diffusion_v07/RESULTS.md](../../us/20261123_diffusion_v07/RESULTS.md)（report；read）
- [experiments/us/20261123_diffusion_v07/REVIEW.md](../../us/20261123_diffusion_v07/REVIEW.md)（report；read）
- [experiments/us/20261123_diffusion_v07/VERDICT.md](../../us/20261123_diffusion_v07/VERDICT.md)（report；read）
- [experiments/us/20261123_diffusion_v07/comparison.json](../../us/20261123_diffusion_v07/comparison.json)（metadata；read）
- [experiments/us/20261123_diffusion_v07/experiment.lock.json](../../us/20261123_diffusion_v07/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_diffusion_v07/results.json](../../us/20261123_diffusion_v07/results.json)（metadata；read）
- [experiments/us/20261123_diffusion_v07/selection.lock.json](../../us/20261123_diffusion_v07/selection.lock.json)（protocol；read）
- [experiments/us/20261123_diffusion_v07/verification.json](../../us/20261123_diffusion_v07/verification.json)（metadata；read）

## 待核查项

- 无明确状态证据；不由目录名称或指标文件推断完成。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
