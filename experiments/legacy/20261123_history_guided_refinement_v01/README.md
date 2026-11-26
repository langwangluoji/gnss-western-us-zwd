# 20261123_history_guided_refinement_v01

当前公开来源：`experiments/us/20261123_history_guided_refinement_v01`；[来源目录](../../us/20261123_history_guided_refinement_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_history_guided_refinement_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_history_guided_refinement_v01/objective_probe/plan.lock.json](../../us/20261123_history_guided_refinement_v01/objective_probe/plan.lock.json)，purpose：distinguish geometric information from scalar calibration and station-MSE vs station-RMSE objective

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261123_history_guided_refinement_v01/status.json](../../us/20261123_history_guided_refinement_v01/status.json)，stage：COMPLETE
- [experiments/us/20261123_history_guided_refinement_v01/confirmation_v02/status.json](../../us/20261123_history_guided_refinement_v01/confirmation_v02/status.json)，stage：COMPLETE
- [experiments/us/20261123_history_guided_refinement_v01/objective_probe/status.json](../../us/20261123_history_guided_refinement_v01/objective_probe/status.json)，stage：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261123_history_guided_refinement_v01/PLAN.md](../../us/20261123_history_guided_refinement_v01/PLAN.md)，第13行：工程检查：FP64正规方程残差、标签隔离、零站/置换不变性、格网与查询等价<=0.001mm、源权重哈希不变。保留全部负结果、逐站预测和运行时间。本机首轮不提交服务器作业。若无收益，根据诊断登记下一轮，禁止事后改变已完成试验门槛。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第34行：48个全格网案例的发布查询最大误差小于0.000053 mm，低于0.001 mm阈值。首次TF32路径最大差约0.0018 mm，失败记录保留；最终扩展指标与发布验证采用禁用TF32的严格FP32。原冻结权重和已拟合系数未改动。初始拟合特征来自原TF32路径。详细锁文件、脚本、预测和失败记录均保留。本轮没有提交服务器作业。

## 历史证据范围与信息条件

- [experiments/us/20261123_history_guided_refinement_v01/PLAN.md](../../us/20261123_history_guided_refinement_v01/PLAN.md)，第1行：# 历史证据驱动的本机模型改进试验
- [experiments/us/20261123_history_guided_refinement_v01/PLAN.md](../../us/20261123_history_guided_refinement_v01/PLAN.md)，第11行：先冻结选择，再报告外层144小时×B77及C2019已暴露48小时；均保留原人口。跨种子外层同向、平均至少0.5%改善及配对天气块区间作为小试收益证据；不保证必然通过。C仅确认方向，不用于选择。所有旧评价均已开发暴露，不能称盲测；晋级后另登记新确认时刻/流程。D2020封存。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第3行：本轮找到可继续验证的精度改进流程。四个冻结起点各自在 B 内层选择低维读出，随后在相同 C144 开发回看上均改善；这并不等于同一个固定配置已跨种子复现。现有最佳产品暂保留。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第6行：输入为四帧 A 站 IDW 五变量、有效掩码、GNSS splat、地形坐标和周期信息，共42通道。气象量为 A 站位 ERA5 采样，不是完整独立气象格网。冻结 U-Net 在 A 站的公共高度预测与真实 A 观测作差；按50/100/200 km及不限制/750 m高差形成6个核，构造核均值或24维均值、方向、高差矩。以 B357 训练站、863合格训练小时拟合 FP64 站均衡 MSE 岭回归，内层留站留天气选择配置。输出完整144×96格网，经统一双线性查询和高程恢复评分。小网络16/32/64、124529参数；宽编码器32/64/128、375473参数。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第8行：## 扩展开发回看
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第18行：区间来自2000次同步月份天气块重采样，共12块，不包含历史选型不确定性。C2019已有历史开发暴露；本轮日期与原48小时不同，不称盲测。D2020未读取。此前 B 外层三组区间低于零，小网络seed42区间跨零。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第31行：常数和振幅再校准未解释主要收益。仅水平三核弱于所选几何方案，但24维方向矩的独立贡献仍需单因素检验，不能把整块收益均归给方向矩。直接平滑站宏RMSE的凸读出改善内层、外层有升有降，保留原锁定岭回归候选。仅重估深层特征读出收益小且不一致，当前证据不足以将瓶颈归于末层优化不足。
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)，第37行：先锁定当前新旧两候选、共同站群、有效站时和主指标，在本机完成完整2019开发年配对评价，报告站宏/微RMSE、季节稳定性及同步7日块bootstrap。当前收益来自确定性创新读出，不归于GAN、Diffusion或VAE；概率模型的改进须在固定均值后独立检验CRPS/ES和校准。25 km仍仅为输出间距，本轮不主张已恢复真实细尺度结构。全年证据支持之后再安排服务器扩大训练。

## 原始协议、结果与报告

- [experiments/us/20261123_history_guided_refinement_v01/PLAN.md](../../us/20261123_history_guided_refinement_v01/PLAN.md)（documentation；read）
- [experiments/us/20261123_history_guided_refinement_v01/RESULTS.md](../../us/20261123_history_guided_refinement_v01/RESULTS.md)（report；read）
- [experiments/us/20261123_history_guided_refinement_v01/experiment.lock.json](../../us/20261123_history_guided_refinement_v01/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/results.json](../../us/20261123_history_guided_refinement_v01/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/status.json](../../us/20261123_history_guided_refinement_v01/status.json)（status；read）
- [experiments/us/20261123_history_guided_refinement_v01/confirmation/confirmation.lock.json](../../us/20261123_history_guided_refinement_v01/confirmation/confirmation.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/confirmation_v02/confirmation.lock.json](../../us/20261123_history_guided_refinement_v01/confirmation_v02/confirmation.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/confirmation_v02/results.json](../../us/20261123_history_guided_refinement_v01/confirmation_v02/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/confirmation_v02/status.json](../../us/20261123_history_guided_refinement_v01/confirmation_v02/status.json)（status；read）
- [experiments/us/20261123_history_guided_refinement_v01/objective_probe/plan.lock.json](../../us/20261123_history_guided_refinement_v01/objective_probe/plan.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/objective_probe/results.json](../../us/20261123_history_guided_refinement_v01/objective_probe/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/objective_probe/status.json](../../us/20261123_history_guided_refinement_v01/objective_probe/status.json)（status；read）
- [experiments/us/20261123_history_guided_refinement_v01/small_17/results.json](../../us/20261123_history_guided_refinement_v01/small_17/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/small_17/selection.lock.json](../../us/20261123_history_guided_refinement_v01/small_17/selection.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/small_42/results.json](../../us/20261123_history_guided_refinement_v01/small_42/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/small_42/selection.lock.json](../../us/20261123_history_guided_refinement_v01/small_42/selection.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/wide_17/results.json](../../us/20261123_history_guided_refinement_v01/wide_17/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/wide_17/selection.lock.json](../../us/20261123_history_guided_refinement_v01/wide_17/selection.lock.json)（protocol；read）
- [experiments/us/20261123_history_guided_refinement_v01/wide_42/results.json](../../us/20261123_history_guided_refinement_v01/wide_42/results.json)（metadata；read）
- [experiments/us/20261123_history_guided_refinement_v01/wide_42/selection.lock.json](../../us/20261123_history_guided_refinement_v01/wide_42/selection.lock.json)（protocol；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
