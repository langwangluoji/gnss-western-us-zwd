# europe_transfer_20261126

当前公开来源：`research/europe_transfer_20261126`；[来源目录](../../../research/europe_transfer_20261126)。

历史路径：`maintenance/europe_transfer_20261126`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第1行：# 欧洲 Model4 数据上的架构迁移

## 类别依据

未定位到明确证据。

## 状态依据

- [research/europe_transfer_20261126/review_receipt.json](../../../research/europe_transfer_20261126/review_receipt.json)，status：passed
- [research/europe_transfer_20261126/verification.json](../../../research/europe_transfer_20261126/verification.json)，status：passed

## 报告中的阶段声明

- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第3行：正式v2已完成：结果、架构与7张图件。相同273站、2020–2023训练池，2024确认人口59有效站/63494共同记录；预先选择p14＋GRU＋I1，双种子预测集成RMSE **17.1472→15.9279mm（降低7.11%）**。全部131个有效空间留出站为16.5642→14.9132mm。原对照及本模型均从欧洲数据重训，17项欧洲检查、568项工程测试和独立数组复算通过。
- [research/europe_transfer_20261126/progress.md](../../../research/europe_transfer_20261126/progress.md)，第13行：- v1先行阶段双种子base和joint均完成，各3000块；原模型双种子均完成重训。原模型最佳CNNepoch19/22，预测有效415518条/种子。尚未计算2024最终RMSE。
- [research/europe_transfer_20261126/progress.md](../../../research/europe_transfer_20261126/progress.md)，第17行：- v2双种子基础场均完成3000块，开发站宏RMSE13.623570/13.585066mm。
- [research/europe_transfer_20261126/progress.md](../../../research/europe_transfer_20261126/progress.md)，第19行：- 独立审阅发现推理未预生成全轴age，窗口首段缺测会走局部fallback。手算36h回归先失败、修正inputs缺省同样构造全轴age；正式训练已使用全轴age，训练权重不受影响。最终选择/评价尚未执行。训练时源快照按lock SHA单独保留。

## 历史证据范围与信息条件

- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第3行：正式v2已完成：结果、架构与7张图件。相同273站、2020–2023训练池，2024确认人口59有效站/63494共同记录；预先选择p14＋GRU＋I1，双种子预测集成RMSE **17.1472→15.9279mm（降低7.11%）**。全部131个有效空间留出站为16.5642→14.9132mm。原对照及本模型均从欧洲数据重训，17项欧洲检查、568项工程测试和独立数组复算通过。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第9行：- 来源：Model4 当前修复版 `euref_train_2020_2023.h5` / `euref_test_2024.h5`；2024 为历史已暴露的回看年度。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第11行：- 正式v2使用完整2020–2023训练站池，2023监控保留开发站址隔离，因此是空间开发监控；2024作年度回看。先行v1按旧CNN阶段用2020–2022神经拟合，单独保留。原系统保持自己的阶段安排：背景2020–2023、CNN2020–2022/2023早停、XGB2023。两系统总体训练站和训练时间池一致。2024不参与拟合、调参或模型选择。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第12行：- 连续6h UTC网格，缺测保留NaN及mask/age，不插值生成观测。16记录历史跨度90h（通常称4天窗口），7输出分别0/6/12/18/24/30/36h；主比较当前ZWD，另保存+6h。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第18行：1. 迁移实际p14 `FixedSupportModel`：113维多尺度点特征、30通道固定载体U-Net16、自由坐标查询；接回原GRU历史空间分支和海岸/地形边分支。从欧洲数据重新训练，不套美国归一化或权重。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第19行：2. 在训练站构造模型创新，拟合9维I1历史岭头；查询站标签及历史不进入条件。原生逐站Transformer编码15维ZWD/T/P、mask/age、创新、时钟，保留站点身份；通过地形/海岸条件查询和有向站×lag传递校正场。新增读出保持零初始化。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第20行：3. 原模型复用原项目FFT-SCHA、CNN-BiLSTM-Attention及XGBoost实现，按相同273训练站筛选后重新拟合；保留原版湿度和自身历史特征。两套结果是系统比较，不能将差异全部归因于神经结构。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第25行：P0 数据审计、源文件SHA、站点CSV、缺测与单位检查 → P1 因果/留站隔离测试与CUDA小试 → P2 p14+GRU双种子各3000连续块 → P3 I1及原生Transformer双种子各3000连续块 → P4 原流程重训 → P5 2024完整年度共同键RMSE、季节/站点图、训练曲线及报告 → P6 独立数值复算、Git同步核对。
- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)，第27行：所有训练时间按顺序遍历连续块；站点条件动态轮换。开发监控独立记录，不用2024挑checkpoint。原始数据、权重、预测数组与缓存保留本地；Git只纳入代码、协议、指标与图件。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第9行：400个训练期站和385个2024站合并为411站，374站跨年共享。按1km连通物理站址分组，再按坐标空间分层划分273训练、64开发、63最终留出站；另有11个2024新站。训练、开发和最终留站之间没有1km物理组交叉；新站如果与训练站同址，从“新站未见站址”指标中剔除。确认站BISK、CNIV、SMI2、SMID在2024源包无标签，故63注册站中59站进入实际评价。完整站点清单见stations.csv，逐站覆盖数见heldout_coverage.csv，源文件SHA和单位审计见data_audit.json。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第13行：输入是实际6h UTC记录，保留缺测，不造小时值。16记录历史跨度90h，7个输出对应0/6/12/18/24/30/36h。本报告主指标为当前ZWD的RMSE，单位mm。2024已在历史研究中暴露，本轮属于年度回看。共同键规则同时要求两系统、两个种子和标签有效；每站断测引起的旧模型连续序列缺失在共同覆盖率中如实体现。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第17行：本模型复用实际p14固定载体U-Net16、113维多尺度点读出、原8×8 GRU历史分支及海岸/地形边分支；随后拟合9维I1历史创新岭头，并训练实际NativeDirectedLagV2因果Transformer候选。逐站15维状态包含GNSS/T/P、mask、age、模型误差和时钟，身份嵌入独立保留，查询注意力用方向、高差、地形和海岸关系传递信息；新增校正头零初始化。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第21行：新模型只使用参考站GNSS ZWD、T、P及静态几何/地形/海岸和时间条件。P由ERA5原始气压、T、高差进行干空气高程修正；不读原流程WVP、TCWV、FFT-SH及VMF3特征。欧洲地形采用站点ERA5静态地形的32×32粗载体插值，梯度和粗糙度从该粗场计算；不是美国SRTM地形的直接复制。海岸距离来自Natural Earth1:5000万海岸线的EPSG3035投影线段距离；源文件名50m中的m表示million，并非50米。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第23行：原模型按相同273训练站重新拟合原FFT-SCHA、CNN-BiLSTM-Attention和XGBoost，保留其原始湿度、气象和目标站历史特征。FFT-SCHA使用2020–2023；CNN-LSTM按原流程2020–2022拟合、2023早停；XGBoost在2023年前9个月拟合/Q4早停后按全年重拟合。本模型正式v2基础场、I1和joint使用2020–2023训练站池，2023监控是独立站址开发监控。两者训练站和总体年份一致，各自特征及训练阶段不同，因此结果是系统效果比较。
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)，第25行：空间场在留出目标的整个站址历史上均不读GNSS；原模型允许读该目标站严格早于当前的历史及其气象湿度。两者最终预测的是同一份Model4 ZWD标签，标签和行筛选继承原制包谱系。以上信息条件应与RMSE同时解释。

## 原始协议、结果与报告

- [research/europe_transfer_20261126/README.md](../../../research/europe_transfer_20261126/README.md)（documentation；read）
- [research/europe_transfer_20261126/REPORT.md](../../../research/europe_transfer_20261126/REPORT.md)（report；read）
- [research/europe_transfer_20261126/comparison.json](../../../research/europe_transfer_20261126/comparison.json)（metadata；read）
- [research/europe_transfer_20261126/progress.md](../../../research/europe_transfer_20261126/progress.md)（documentation；read）
- [research/europe_transfer_20261126/review_receipt.json](../../../research/europe_transfer_20261126/review_receipt.json)（metadata；read）
- [research/europe_transfer_20261126/split_lock.json](../../../research/europe_transfer_20261126/split_lock.json)（protocol；read）
- [research/europe_transfer_20261126/verification.json](../../../research/europe_transfer_20261126/verification.json)（metadata；read）
- [research/europe_transfer_20261126/smoke/base_s42/lock.json](../../../research/europe_transfer_20261126/smoke/base_s42/lock.json)（protocol；read）
- [research/europe_transfer_20261126/smoke/base_s42/progress.json](../../../research/europe_transfer_20261126/smoke/base_s42/progress.json)（status；read）
- [research/europe_transfer_20261126/smoke/joint_s42/lock.json](../../../research/europe_transfer_20261126/smoke/joint_s42/lock.json)（protocol；read）
- [research/europe_transfer_20261126/smoke/joint_s42/progress.json](../../../research/europe_transfer_20261126/smoke/joint_s42/progress.json)（status；read）
- [research/europe_transfer_20261126/smoke/joint_v2_s42/lock.json](../../../research/europe_transfer_20261126/smoke/joint_v2_s42/lock.json)（protocol；read）
- [research/europe_transfer_20261126/smoke/joint_v2_s42/progress.json](../../../research/europe_transfer_20261126/smoke/joint_v2_s42/progress.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
