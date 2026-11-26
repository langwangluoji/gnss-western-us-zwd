# 20261122_local_operator_exploration_v01

当前公开来源：`experiments/us/20261122_local_operator_exploration_v01`；[来源目录](../../us/20261122_local_operator_exploration_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_local_operator_exploration_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_local_operator_exploration_v01/PLAN.md](../../us/20261122_local_operator_exploration_v01/PLAN.md)，第1行：# 本机空间算子探索

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261122_local_operator_exploration_v01/status.json](../../us/20261122_local_operator_exploration_v01/status.json)，stage：COMPLETE
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_0/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_0/status.json)，stage：gate_refine
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_0/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_0/status.json)，status：COMPLETE
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_1/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_1/status.json)，stage：gate_refine
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_1/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_1/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261122_local_operator_exploration_v01/PLAN.md](../../us/20261122_local_operator_exploration_v01/PLAN.md)，第15行：沿用每折内层至少0.5%改善、外层同向、短距离变化与线性对照核查。筛查全失败时不扩大对应模型，重新向Pro讨论。若某类有稳定信号，再登记匹配的浅层神经算子训练或新天气确认；概率收益和点值收益分别评价。

## 历史证据范围与信息条件

- [experiments/us/20261122_local_operator_exploration_v01/physical_output_plan.lock.json](../../us/20261122_local_operator_exploration_v01/physical_output_plan.lock.json)，scope：Post-run engineering parameterization, all methods equally; no retuning, no new independent confirmation
- [experiments/us/20261122_local_operator_exploration_v01/C2019_confirmation/results.json](../../us/20261122_local_operator_exploration_v01/C2019_confirmation/results.json)，scope：fixed C2019 development dates; not full-year or final retrospective
- [experiments/us/20261122_local_operator_exploration_v01/C2019_replication/results.json](../../us/20261122_local_operator_exploration_v01/C2019_replication/results.json)，scope：fixed C2019 development dates; not full-year or final retrospective
- [experiments/us/20261122_local_operator_exploration_v01/PLAN.md](../../us/20261122_local_operator_exploration_v01/PLAN.md)，第5行：保持上一批576训练小时、48内层小时、原288外层锚点、物理站组、地表标签与公共格网查询。B2018含历史暴露，所有结果称增量开发。C2019、D2020不读取。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第3行：已找到值得继续验证的方向：保留A站创新量的方向、高差和尺度信息，再映射到统一格网。两组固定C2019开发时刻均复现点值改善。当前证据更支持改善站点到格网的表达，而不是继续扩大原非线性预测头。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第5行：本批是用户授权的本机探索spike：三个有限算子家族、两折天气门控训练、两组各48个C开发时刻和数值复核。未修改生产源码及环境，未操作服务器或D2020。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第9行：B2018保持576训练小时、48空间×天气内层时刻、原288外层锚点。历史底座/先验和既往外层已有暴露，属于增量开发。C两组分别固定每月7日0/6/12/18时和14日3/9/15/21时，各48小时、76个可评分站；第二组在查看第一组后登记，模型和参数全部不变。C是已暴露开发年，D继续封存。先提交格网预测，再读取C标签。两折模型分别报告，没有按时刻或指标择优拼接。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第29行：核旁路：在公共格点定义水平100 km、高程750 m的平滑协方差核，用真实A查询矩阵构建Kobs=QA Kgrid QA^T。对当小时A创新解线性系统，整体幅度仅用拟合B估计；两折均由内层选中噪声比0.1。底座本已使用A，因此此处是回归校正，不能解释成独立观测的贝叶斯更新，也不能证明山脊阻断水汽。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第41行：全场范围检查发现：原底座在支持域外已有负Z0；24维折0在复验中另有1个支持格点为负。因此原始分数全部保留，另登记共同正值输出参数化Z0=0.1 logaddexp(0,raw_Z0/0.1)，对全部方法在格点、查询之前应用。输出检查和正值场分别见physical_output_checks.json与各组positive_fields.npz；包含全陆地和原支持标记，不删评分样本。它是事后工程约束检查，不算新增独立精度确认，也不能保证支持域外精度。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第51行：随后保留现有ResU-Net骨干，把已验证的站点几何聚合/核修正作为共享条件候选。优先比较6维简版与24维已验证版，不因名称更复杂而扩网。方向读出与核旁路属于已测试的两种均值方案；二者相加、共同送入网络仍需独立试验。选定并冻结共同均值后，Gaussian、Diffusion、GAN使用相同条件和评分人口。概率分支应保护均值，ES/VS收益不替代点值目标。详细后续安排见NEXT_PLAN.md。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第53行：若全年优势消失则保留更简单方法；若门控不稳定优于静态核则不扩门控。没有完整独立高分辨率目标场时，不把IDW图当作扩散真值。尚未运行新Diffusion/GAN训练、全年度新模型评价或服务器作业。
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)，第59行：- C2019_confirmation/、C2019_replication/：协议、完整格网、逐站预测、指标与区间。

## 原始协议、结果与报告

- [experiments/us/20261122_local_operator_exploration_v01/PLAN.md](../../us/20261122_local_operator_exploration_v01/PLAN.md)（documentation；read）
- [experiments/us/20261122_local_operator_exploration_v01/RESULTS.md](../../us/20261122_local_operator_exploration_v01/RESULTS.md)（report；read）
- [experiments/us/20261122_local_operator_exploration_v01/ablation_plan.lock.json](../../us/20261122_local_operator_exploration_v01/ablation_plan.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/experiment.lock.json](../../us/20261122_local_operator_exploration_v01/experiment.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/gate_selection.lock.json](../../us/20261122_local_operator_exploration_v01/gate_selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/gate_training.lock.json](../../us/20261122_local_operator_exploration_v01/gate_training.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/physical_output_plan.lock.json](../../us/20261122_local_operator_exploration_v01/physical_output_plan.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/results.json](../../us/20261122_local_operator_exploration_v01/results.json)（metadata；read）
- [experiments/us/20261122_local_operator_exploration_v01/selection.lock.json](../../us/20261122_local_operator_exploration_v01/selection.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/status.json](../../us/20261122_local_operator_exploration_v01/status.json)（status；read）
- [experiments/us/20261122_local_operator_exploration_v01/C2019_confirmation/protocol.lock.json](../../us/20261122_local_operator_exploration_v01/C2019_confirmation/protocol.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/C2019_confirmation/results.json](../../us/20261122_local_operator_exploration_v01/C2019_confirmation/results.json)（metadata；read）
- [experiments/us/20261122_local_operator_exploration_v01/C2019_replication/protocol.lock.json](../../us/20261122_local_operator_exploration_v01/C2019_replication/protocol.lock.json)（protocol；read）
- [experiments/us/20261122_local_operator_exploration_v01/C2019_replication/results.json](../../us/20261122_local_operator_exploration_v01/C2019_replication/results.json)（metadata；read）
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_0/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_0/status.json)（status；read）
- [experiments/us/20261122_local_operator_exploration_v01/gate_fold_1/status.json](../../us/20261122_local_operator_exploration_v01/gate_fold_1/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
