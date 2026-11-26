# 20261122_generation_postrun_research_v01

当前公开来源：`experiments/us/20261122_generation_postrun_research_v01`；[来源目录](../../us/20261122_generation_postrun_research_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261122_generation_postrun_research_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第1行：# 本批生成模型的外层复核与后续研究结论

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json)，status：COMPLETE
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json)，status：COMPLETE
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json)，status：COMPLETE
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json)，status：COMPLETE
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json)，status：COMPLETE
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json)，stage：generation_refine
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json)，status：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第57行：Slurm3645090确认COMPLETED、0:0，ElapsedRaw=11363秒；前次预算门槛退出3644321为131秒。累计单A800分配3.1928卡时，低于原4卡时上限。查询时没有本项目GPU作业在队列；其他CPU作业未作改动。本轮研究没有提交新的服务器训练。

## 历史证据范围与信息条件

- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第9行：数据为B2018增量开发，含历史基础场及先验对B资料的暴露。每折训练5472小时，一遍684更新，三个模型各两折，共4104次正式更新。内层96小时使用拟合站，外层96小时为12个固定天气周各8个时刻并留出物理站组。第0折实际评分204站、19131站时刻；第1折244站、22844站时刻。两折分别报告空间联合指标。C2019、D2020没有用于本批新评价；此处数值不能直接与C2019全年RMSE排序。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第20行：所有产品均由内层规则选择，point/joint本批恰好选中同一步684，但两个产品仍保留独立清单。学习均值修正的RMS为第0折2.09—2.36mm、第1折3.30—3.33mm。在12个抽样天气周上进行2000次同步配对块重采样，原生成器相对基础场的RMSE差值区间均为正；该区间限于本轮12簇的探索性判断，未做密集物理组重采样及多重比较调整。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第28行：为排除该依赖，另做前16成员估计平移、后16成员评分，再互换的诊断；两次K16分数分别计算，未拼成K32，也未作为独立重复。小幅概率改善依然存在。但依据独立先导成员预测方差构造的逐小时单标量高斯缩放，对ES、CRPS、VS均表现更好。它使用预测值，不使用当前目标标签。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第37行：这里α只缩放潜在空间场；τ仍为原6.1827mm，并在站端独立添加一次。它不同于缩整个观测集合的先导幅度a≈0.87—0.91。完整数据见gaussian_scale_control.json；其中total_error是另列的补充对照，不替代本表的field_only主对照。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第43行：直接用K个样本的线性分位数构造区间，覆盖率本身依赖K。完全正确的标准正态模型模拟中，标称90%区间的平均实际覆盖为K16：80.65%、K32：85.04%、K64：87.31%、K128：88.64%；不同种子直接取目标及均匀次序统计量复核通过。这个参考不作为所有非高斯模型的校正表。评价保留历史CRPS-V，同时补fair CRPS-U、随机化秩和解析Gaussian区间。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第51行：通过既有内置浏览器Pro会话提供真实协议和数值，完成一次方法复核及一次新增单参数证据复审。采用：校准Gaussian强对照、空间内层选模、点/概率职责分开、独立成员评分及条件偏态前置检查。反对称随机残差仅作为可证伪的均值保护候选，保留其不能表示条件偏态的限制；不直接宣称为最终架构。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第53行：会话：https://chatgpt.com/c/6af6fa29-6a4c-83e8-a0a7-15d62767383b 。网页答复不作为独立实验证据；本记录中的数值均来自客户端复算。下一轮详细安排见NEXT_STEP.md。
- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)，第59行：下载126个文件的逐文件哈希及归档SHA256通过，归档：27a4ca243dbc6117f2a26f0f4b44ed73ea2414b5fb01fcab38d9bd9f06c5d28a。旧发布包、训练检查点及环境保持可追溯。完整代码测试310项通过；新诊断覆盖配对人口、非有限预测、平移分解、独立先导、不偏CRPS与一维ES一致性、秩并列处理和固定站群块重采样。

## 原始协议、结果与报告

- [experiments/us/20261122_generation_postrun_research_v01/RESULTS.md](../../us/20261122_generation_postrun_research_v01/RESULTS.md)（report；read）
- [experiments/us/20261122_generation_postrun_research_v01/gaussian_scale_probe.lock.json](../../us/20261122_generation_postrun_research_v01/gaussian_scale_probe.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/member_stability_probe.lock.json](../../us/20261122_generation_postrun_research_v01/member_stability_probe.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/RESULTS.md](../../us/20261122_generation_postrun_research_v01/downloaded/RESULTS.md)（report；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/allocation.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/allocation.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/budget.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/budget.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/data.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/data.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/summary.json](../../us/20261122_generation_postrun_research_v01/downloaded/summary.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/engineering/verification.json](../../us/20261122_generation_postrun_research_v01/downloaded/engineering/verification.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/diffusion/status.json)（status；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/gan/status.json)（status；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_0/stochastic/status.json)（status；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/diffusion/status.json)（status；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/gan/status.json)（status；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/results.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/results.json)（metadata；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/selection.lock.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/selection.lock.json)（protocol；read）
- [experiments/us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json](../../us/20261122_generation_postrun_research_v01/downloaded/fold_1/stochastic/status.json)（status；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
