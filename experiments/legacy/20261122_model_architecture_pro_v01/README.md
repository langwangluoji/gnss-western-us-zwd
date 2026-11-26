# 20261122_model_architecture_pro_v01

当前公开来源：`experiments/us/20261122_model_architecture_pro_v01`；[来源目录](../../us/20261122_model_architecture_pro_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_model_architecture_pro_v01`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第1行：# GNSS场下一阶段结构与实验方案

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第3行：版本：20261122_model_architecture_pro_v01。状态：方案与数学诊断完成；生产模型尚未实现，本轮未提交服务器作业。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第127行：此候选的单点矩与空间协方差已完成Gauss-Hermite积分复核。它同时改变边缘和协方差，胜出后还须与同协方差Gaussian比较，才可讨论超越二阶结构的收益。固定系数的格点偏态绝对值<1、超额峰度<3；初始b范围[-1,1]的偏态上限更小。当前校准Gaussian方差下，独立项对站端偏态的稀释因子中位约0.156。这个因子是旧协方差下的诊断，不能把格点偏态上限直接乘它当作新模型站端偏态的普遍上限；必须用新场四角联合矩或共同成员重新计算。它是待检验的强简单对照，尚不确定适合作为主生成器。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第169行：生成模型推进条件为：同均值下，空间内层ES/VS较锁定Gaussian改善且CRPS不增加，改善在预登记天气块中可复现；偏态收益还须优于同协方差Gaussian。无剩余非Gaussian信号时，交付Gaussian结果并停止本轮生成结构扩容。没有预算内收益与证明Diffusion/GAN无效是不同结论。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第203行：- 已完成试验与本机复核：`../20261122_generation_postrun_research_v01/RESULTS.md`、`gaussian_scale_control.json`、`member_stability_summary.json`。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第205行：- Pro协作会话：<https://chatgpt.com/c/6af6fa29-6a4c-83e8-a0a7-15d62767383b>；两轮已完成回答及采用/修正决定见`PRO_REVIEW.md`。网页模型建议作为审阅意见，不替代实测或推导证据。

## 历史证据范围与信息条件

- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第9行：点值目标仍是可信的独立站精度改善，约6 mm是待验证目标。概率目标是空间联合分布与区间可信度。两类收益分别报告。25 km称输出间距；有效空间分辨率须由独立站间变化证据支持。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第20行：U --> G[空间场／独立项双尺度Gaussian审计]
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第43行：仅用允许的训练B站拟合空间幅度，α约0.60，已得到更好的ES/CRPS。它保持均值不变，不能解释为点值改善。当前τ=6.1827 mm来自残差模型拟合，包含未解析变化与观测误差，不能解释成独立测量的仪器噪声，也不能当作6 mm精度的物理下限。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第45行：只缩空间场、缩整个观测残差两种Gaussian的CRPS接近，VS有所不同，提示进一步检查场与独立项的分配。它尚未证明某个分配正确，亦未证明CRPS/ES原则上无法辨识参数。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第47行：这些结果属于带历史底座及先验B暴露的增量开发。原外层已用于本轮方法发现，继续用于探索性复核。C2019为开发资料，D2020保持封存。这里的7.4—7.5 mm不能与C2019全年RMSE直接排序。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第49行：## 3. 点值支路：均值独立优化
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第76行：P2有硬幅度界 `|Δμ|≤(|J1|+|J2|)/2`。底座本身吸收A观测，A自影响会缩小创新；若B残差可预测但J很小，P2可能受限。因此P3是必要的表达能力对照。去XY、零初始化及有界修正均不能保证训练后细节正确，需由独立站验证。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第89行：P为原冻结零均值空间Gaussian场，η是站端独立标准正态；τ只添加一次。发布的潜在场标准差与站端预测区间分别保存，不能把站端独立项伪装为连续空间纹理。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第100行：站组仅由几何预登记；缺测时取真实有效主子矩阵，不补造标签。重叠组的NLL是复合目标，不能把组数当独立样本数。统计/线性代数使用FP64。拟合范围先登记为两个尺度均在[0.1,1.8]；若触边，作为边界诊断，不根据外层放宽。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第104行：本批固定相关长度，避免把场方差、短程相关、独立项同时放开。只有配对增量与距离分箱诊断持续显示相关结构失配，才另行登记相关长度实验。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第118行：设P,Q为相互独立、同协方差的完整Gaussian场，格点标准差s；在s>0处定义X=(P+Q)/(√2s)、Y=(P−Q)/(√2s)：
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第125行：s=0处显式置R=0。尺度头固定b=0，尺度—偏态头可学b。最后发布 `μ+gR`，站端加 `gα_nτη` 一次。系数仅依赖条件，或后续与载体独立的系数噪声；不得读取P/Q，否则精确零均值保证失效。末层零初始化时逐成员复现已校准Gaussian。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第127行：此候选的单点矩与空间协方差已完成Gauss-Hermite积分复核。它同时改变边缘和协方差，胜出后还须与同协方差Gaussian比较，才可讨论超越二阶结构的收益。固定系数的格点偏态绝对值<1、超额峰度<3；初始b范围[-1,1]的偏态上限更小。当前校准Gaussian方差下，独立项对站端偏态的稀释因子中位约0.156。这个因子是旧协方差下的诊断，不能把格点偏态上限直接乘它当作新模型站端偏态的普遍上限；必须用新场四角联合矩或共同成员重新计算。它是待检验的强简单对照，尚不确定适合作为主生成器。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第139行：三臂共享载体定义与冻结底座，各自独立权重；同一成员的系数噪声ω、P、Q、η分别独立，成员之间也独立。不能把一份随机系数给所有成员共享。输入虽同为71通道，旧版的“69＋先验场＋白噪声”与新版“69＋双通道系数状态”语义不同，必须升级接口、禁止无校验加载旧权重。
- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)，第141行：Diffusion以同调度解析零网络采样结果作参照，检查FP32零初始化误差。本方案没有独立HR真值/系数真值，使用EDM预条件的采样器并不等于已做真实HR残差去噪训练。对外须称观测评分监督的扩散式采样参数化。不能将IDW参考图直接作为真值来补齐DSM监督。

## 原始协议、结果与报告

- [experiments/us/20261122_model_architecture_pro_v01/MODEL_PLAN.md](../../us/20261122_model_architecture_pro_v01/MODEL_PLAN.md)（documentation；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
