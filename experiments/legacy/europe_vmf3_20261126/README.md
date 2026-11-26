# europe_vmf3_20261126

当前公开来源：`research/europe_vmf3_20261126`；[来源目录](../../../research/europe_vmf3_20261126)。

历史路径：`maintenance/europe_vmf3_20261126`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第1行：# 欧洲基础场引入VMF3-FC背景特征

## 类别依据

未定位到明确证据。

## 状态依据

- [research/europe_vmf3_20261126/delivery_verification.json](../../../research/europe_vmf3_20261126/delivery_verification.json)，status：passed
- [research/europe_vmf3_20261126/training_verification.json](../../../research/europe_vmf3_20261126/training_verification.json)，status：passed

## 报告中的阶段声明

- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第3行：**正式结果已完成**：完整报告、模型架构与7组结果图。双种子基础场和14组概率模型训练、14组完整年度K32评价均已结束。同59有效确认站/63494共同键，新基础场RMSE由15.9279降至11.4064mm（降低28.39%）；Local的+6h CRPS为9.82785mm，相对同条件GP改善0.96%，对Affine的额外差异区间跨零。点场改进与概率收益分别报告。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第7行：**当前授权修订**：用户明确重点是把VMF3-FC引入新的“基础场＋扩散”模型，既有CNN不再训练。CNN seed17已立即停止；seed42在澄清前完成的试验保留归档，不作为本轮新对照任务。早期计划中的CNN重训安排已被本段及扩散实施合同替代。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第28行：原CNN-LSTM-XGBoost保留上轮固定预测，不再训练；澄清前完成的seed42试验及停止的seed17试验本地归档。VMF3 native和aligned自身作为直接背景基线。主评价对齐上轮确认站63494共同键。所有基础场选择只用2023开发站；2024新结果用于历史暴露年度回看。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第5行：按用户修订，已有CNN保留为固定参考，后续研究只训练新融合模型；澄清前已完成的CNN seed42尝试归档，seed17已停止，均不进入本轮主比较。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第3行：**当前状态：COMPLETE。** 以下早期条目按发生顺序保留；最终状态和结果见末尾完成记录与正式报告。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第8行：- 用户随后澄清：既有CNN无需重训，主要把VMF3-FC接入扩散新模型。已停止legacy_vmf seed17进程29324，保留部分日志/锁；seed42已在澄清前完成，归档且不作为本轮新增比较主线。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第9行：- 基础场辅助消融已完成：control/vmf3双种子各3000连续块；seed42 VMF3开发站宏RMSE9.946979mm、seed17 9.927412mm，相对原I1开发12.424873mm。此为2023开发，2024新数值尚未读取。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第12行：- 12项数学/编码/单位/PSD/查询一致性检查通过；7个概率模型均完成真实CUDA四更新，实际监督时钟与Gaussian零速度初值通过。正式各3000待启动。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第16行：- control/vmf3两臂、两个种子，均完成3000连续四记录块。VMF3均值分支于2023开发站监控后封存；未来时效保留原joint预测增量，以新当前值作公共偏移，未输入未来VMF3有效时刻。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第18行：- 14组K32完整2024年度评价全部完成，Local/Affine采用Heun0 20步，Gaussian/GP直接采样。每组1464起点、138查询、7时效；主确认当前人口59有效站、63494键。base与vmf均共享新VMF3基础场，base只屏蔽额外显式VMF3概率条件。

## 历史证据范围与信息条件

- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第3行：**正式结果已完成**：完整报告、模型架构与7组结果图。双种子基础场和14组概率模型训练、14组完整年度K32评价均已结束。同59有效确认站/63494共同键，新基础场RMSE由15.9279降至11.4064mm（降低28.39%）；Local的+6h CRPS为9.82785mm，相对同条件GP改善0.96%，对Affine的额外差异区间跨零。点场改进与概率收益分别报告。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第5行：最终交付核验 · 独立数值及图件审阅 · 概率模型结果。复核命令：`python -m maintenance.europe_vmf3_20261126.verify_delivery`。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第13行：继承欧洲v2的411站元数据、273训练/64开发/63确认/11新站分组、2020–2023训练池、2023留站开发选择及2024年度回看。保留现有模型、权重、科学锁和15.927922mm主结果。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第17行：本模型保持已有参考站GNSS/T/P条件：aligned中的站压由条件站T/P按干空气高程传递并空间加权估计，缺测回退到VMF3自身静力项；不读取目标站T/P、WVP、TCWV或FFT-SCHA。anomaly减本模型I1场。原CNN保留自身站压与FFT-SCHA定义，因此三特征算法同源、对齐参考各自遵循原模型。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第26行：- vmf3：输入当前与历史VMF3三特征及有效mask，检验VMF3额外收益。
- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)，第28行：原CNN-LSTM-XGBoost保留上轮固定预测，不再训练；澄清前完成的seed42试验及停止的seed17试验本地归档。VMF3 native和aligned自身作为直接背景基线。主评价对齐上轮确认站63494共同键。所有基础场选择只用2023开发站；2024新结果用于历史暴露年度回看。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第3行：VMF3-FC已同时进入新基础场和残差扩散条件。2024完整年度回看、相同59有效确认站/63494共同起点上，双种子预测集成的当前场RMSE由15.9279降至11.4064mm，降低28.39%；相对VMF3 aligned直接背景的12.6661mm降低9.95%。固定基础场上，Local相对同条件天气GP的+6h CRPS改善0.96%。非Gaussian相对Affine的额外收益按下方区间单独解释。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第11行：VMF3-FC1°预报归档按原CNN的算法处理：四格点分别进行Kouba高程修正，再双线性到查询点。三个背景特征是native ZWD、与本模型干湿口径对齐的aligned ZWD、aligned减原I1场的异常。aligned中的P由原有参考站T/P按干空气高程传递和空间加权得到；目标站GNSS/T/P、WVP、TCWV和原FFT-SCHA均不作为新条件。缺2023-12-18 00UTC的原始背景用mask保留，不从未来有效时刻填充。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第23行：表格均为两个训练种子的预测先等权平均，再计算共同键RMSE。native/aligned是直接背景，不含新神经训练；control是相同新校正器、无VMF3输入。CNN来自上轮固定273站模型，使用其原气象湿度和同站历史特征，未启用VMF3，仅保留为固定系统参考。主要归因比较原I1/control与直接VMF3背景。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第38行：VMF3背景加入了新的外部物理信息，收益包含条件信息扩充；control显示新增校正器本身的变化较小。native与aligned的差异也说明干湿分离口径对齐的重要性，融合场在对齐背景之上继续改善。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第44行：下面分数为两个固定训练种子的平均，未拼接为64成员分布。每模型K32独立成员；Local/Affine采用Heun0 20步，Gaussian/GP直接采样。GP/Local/Affine共享同源GP噪声路径以作配对比较。当前/未来共用上轮固定起点，未来mask仅用于监督和评分。Gaussian与GP具有相同解析边际，其采样CRPS微差来自有限成员；GP主要检验联合相关结构。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第58行：Local相对同VMF条件天气GP的+6h CRPS降低0.09481mm。Local相对Affine的+6h差值为-0.00162mm，95%共同月份重采样区间[-0.02264,+0.02238]，区间跨零；当前Local比Affine高0.01719mm。当前证据支持强Gaussian基础上的传输收益，未建立非Gaussian模块的独立优势。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第64行：K32的第3/30秩区间在交换性条件下参考覆盖率为27/33=81.82%，不是精确80%。Local VMF的当前/+6h秩覆盖为84.32%/82.71%；解析Gaussian80%覆盖另存JSON。月份区间固定本次两个训练种子和采样流，表示该回看年份内的过程变化。场能量采用独立配对估计并除以观测维数平方根；各模型使用相同观测维度。近邻站对的几何选择、数量和分数见spatial_contrasts.json。
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)，第85行：继承273训练/64开发/63注册确认/11新站的1km物理组隔离；2020–2023拟合、2023开发为留站监控，2024为历史已暴露年度回看。确认63站中59有本轮共同有效记录。原Model4标签和湿度参与制包筛选的谱系保留；strict_oof=false。原始NWP归档仅验证有效时刻映射，未逐文件追溯发布时间，不将本轮称为业务实时预报验证。滚动起点间独立采样，每起点内生成相关7时效场。
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)，第10行：- 当前重点转为VMF3条件的Gaussian→天气GP→Affine/Local残差扩散；全部以欧洲新残差重训，保留显式最终7时效均值与VMF3历史条件。

## 原始协议、结果与报告

- [research/europe_vmf3_20261126/README.md](../../../research/europe_vmf3_20261126/README.md)（documentation；read）
- [research/europe_vmf3_20261126/REPORT.md](../../../research/europe_vmf3_20261126/REPORT.md)（report；read）
- [research/europe_vmf3_20261126/delivery_verification.json](../../../research/europe_vmf3_20261126/delivery_verification.json)（metadata；read）
- [research/europe_vmf3_20261126/point_results.json](../../../research/europe_vmf3_20261126/point_results.json)（metadata；read）
- [research/europe_vmf3_20261126/probability_lock.json](../../../research/europe_vmf3_20261126/probability_lock.json)（protocol；read）
- [research/europe_vmf3_20261126/probability_results.json](../../../research/europe_vmf3_20261126/probability_results.json)（metadata；read）
- [research/europe_vmf3_20261126/progress.md](../../../research/europe_vmf3_20261126/progress.md)（documentation；read）
- [research/europe_vmf3_20261126/review_receipt.json](../../../research/europe_vmf3_20261126/review_receipt.json)（metadata；read）
- [research/europe_vmf3_20261126/training_verification.json](../../../research/europe_vmf3_20261126/training_verification.json)（metadata；read）
- [research/europe_vmf3_20261126/smoke/s42/base/gaussian/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/base/gaussian/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/base/gaussian/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/base/gaussian/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/base/gp/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/base/gp/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/base/gp/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/base/gp/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/base/local/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/base/local/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/base/local/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/base/local/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/affine/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/affine/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/affine/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/affine/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/gaussian/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/gaussian/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/gaussian/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/gaussian/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/gp/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/gp/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/gp/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/gp/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/local/lock.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/local/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/s42/vmf/local/progress.json](../../../research/europe_vmf3_20261126/smoke/s42/vmf/local/progress.json)（status；read）
- [research/europe_vmf3_20261126/smoke/vmf3_s42/lock.json](../../../research/europe_vmf3_20261126/smoke/vmf3_s42/lock.json)（protocol；read）
- [research/europe_vmf3_20261126/smoke/vmf3_s42/progress.json](../../../research/europe_vmf3_20261126/smoke/vmf3_s42/progress.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
