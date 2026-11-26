# 20261123_observation_transport_v11

当前公开来源：`experiments/us/20261123_observation_transport_v11`；[来源目录](../../us/20261123_observation_transport_v11)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_observation_transport_v11`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第1行：# v11 实践结果：观测编码改善点值，扩散呈现局地空间关联

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第3行：完成日期：2026-11-24。10组本机训练，共24,000次参数更新；六组确定性编码对照与四组概率模型均完成。结果支持继续研究局地空间联合生成，同时把下一轮点值优化集中到空间支持分布与查询一致性。

## 历史证据范围与信息条件

- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第5行：## 1. 评价人口与比较口径
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第7行：沿用美国西部最新允许变量QC合同：训练410站、验证137站；2016/2017共72个训练窗口。模型条件为稀疏站ZWD、T/P和静态几何，保留12小时可观测条件历史，训练与推理均无目标站历史。不输入PWV/IWV/TCWV/Tm，也没有v04均值背景或插值教师。确认138及原C/D标签未读取。历史站点队列仍受到早期PWV完整率筛选影响。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第9行：选模为2018年1/4/7/10月1日前4小时、3种固定观测布局；冻结权重后扩展到12个月月1日、每段24小时，每布局288小时。扩展集与选模集部分重叠，属于开发期稳定性检查，并非完整逐小时全年或独立盲测。宏RMSE先按站计算，再对布局和种子平均；概率结果用fair CRPS，越低越好。B选择与扩展均K16，另以K32扩散/解析高斯复核C门槛。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第31行：对137个不同物理组的验证站构造站间ZWD差值，0–50km共有720站对。扩展范围高斯两种子差值CRPS为4.997/5.033 mm，扩散4.720/4.805 mm，平均改善5.0%。保持每站所有成员取值完全不变，逐站独立打乱成员编号后，扩散差值CRPS增加0.135/0.131 mm；高斯变化约0.001 mm。这支持扩散已经学到对局地差值分布有益的空间对应。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第33行：50–150km的519站对上，扩散打乱后反而改善0.029/0.026 mm，远距离相关结构还需校准。此结果为选模结束后的探索性分析，3次shuffle重复不是置信区间；站对之间也非独立样本。它验证观测站对的联合变化，不等价于无人测量格网上的细尺度精度。后续应在锁定距离分箱与独立时段上复核，并加入相关高斯对照，区分一般空间协方差与扩散独有贡献。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第51行：C门槛复核使用冻结权重、独立噪声、K32扩散，以及解析高斯均值/CRPS。扩散两种子CRPS为7.041/6.877，高斯6.829/6.810 mm；扩散相对A点值差为+0.238/−0.109 mm。两种子均未满足全部条件，因此本轮C自生成历史训练不启动。概率收益明确解释为“超过配对高斯且两种子成立”，以及K32复核，是B期间经审查补充的裁定，未冒称全部预注册。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第53行：下一轮建议按顺序建立独立假设，每次两种子、相同预算：
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第57行：3. **局地相关生成。** 在前两项稳定后比较独立高斯、相关高斯与扩散；把站对差值CRPS与点值CRPS同时锁定为评判项，限制远距离错误关联。增加时间关联作为单独假设测试，而非把当前空间证据当作时序证据。
- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)，第60行：本轮随小时更新可观测条件，未训练目标自生成历史，也不是24小时预报。连续场演变仍是下一阶段任务。
- [experiments/us/20261123_observation_transport_v11/REVIEW.md](../../us/20261123_observation_transport_v11/REVIEW.md)，第5行：独立只读审查未发现高差符号、单位或目标标签泄漏阻塞问题。发现评估默认目录名all_months24可能错误描述季度范围：已改为按实际日历生成名称，报告额外核验12个月和24小时清单。已有A扩展调用均显式指定all-months，数值不受此问题影响；回归测试经历RED→GREEN。
- [experiments/us/20261123_observation_transport_v11/REVIEW.md](../../us/20261123_observation_transport_v11/REVIEW.md)，第11行：学习曲线和扩展比较图已生成；训练、评价日历、预训练预算、历史PWV队列筛选、未训练时序历史等解释边界记录于RESULTS.md。

## 原始协议、结果与报告

- [experiments/us/20261123_observation_transport_v11/RESULTS.md](../../us/20261123_observation_transport_v11/RESULTS.md)（report；read）
- [experiments/us/20261123_observation_transport_v11/REVIEW.md](../../us/20261123_observation_transport_v11/REVIEW.md)（report；read）
- [experiments/us/20261123_observation_transport_v11/summary.json](../../us/20261123_observation_transport_v11/summary.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/verification.json](../../us/20261123_observation_transport_v11/verification.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s17/config.json](../../us/20261123_observation_transport_v11/A_base_s17/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s17/metrics.json](../../us/20261123_observation_transport_v11/A_base_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s17/all_months24/config.json](../../us/20261123_observation_transport_v11/A_base_s17/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s17/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_base_s17/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s42/config.json](../../us/20261123_observation_transport_v11/A_base_s42/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s42/metrics.json](../../us/20261123_observation_transport_v11/A_base_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s42/all_months24/config.json](../../us/20261123_observation_transport_v11/A_base_s42/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_base_s42/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_base_s42/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s17/config.json](../../us/20261123_observation_transport_v11/A_raw_s17/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s17/metrics.json](../../us/20261123_observation_transport_v11/A_raw_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s17/all_months24/config.json](../../us/20261123_observation_transport_v11/A_raw_s17/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s17/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_raw_s17/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s42/config.json](../../us/20261123_observation_transport_v11/A_raw_s42/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s42/metrics.json](../../us/20261123_observation_transport_v11/A_raw_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s42/all_months24/config.json](../../us/20261123_observation_transport_v11/A_raw_s42/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_raw_s42/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_raw_s42/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s17/config.json](../../us/20261123_observation_transport_v11/A_transport_s17/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s17/metrics.json](../../us/20261123_observation_transport_v11/A_transport_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s17/all_months24/config.json](../../us/20261123_observation_transport_v11/A_transport_s17/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s17/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_transport_s17/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s42/config.json](../../us/20261123_observation_transport_v11/A_transport_s42/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s42/metrics.json](../../us/20261123_observation_transport_v11/A_transport_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s42/all_months24/config.json](../../us/20261123_observation_transport_v11/A_transport_s42/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/A_transport_s42/all_months24/metrics.json](../../us/20261123_observation_transport_v11/A_transport_s42/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s17/config.json](../../us/20261123_observation_transport_v11/B_diffusion_s17/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s17/metrics.json](../../us/20261123_observation_transport_v11/B_diffusion_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s17/all_months24/config.json](../../us/20261123_observation_transport_v11/B_diffusion_s17/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s17/all_months24/metrics.json](../../us/20261123_observation_transport_v11/B_diffusion_s17/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s42/config.json](../../us/20261123_observation_transport_v11/B_diffusion_s42/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s42/metrics.json](../../us/20261123_observation_transport_v11/B_diffusion_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s42/all_months24/config.json](../../us/20261123_observation_transport_v11/B_diffusion_s42/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_diffusion_s42/all_months24/metrics.json](../../us/20261123_observation_transport_v11/B_diffusion_s42/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s17/config.json](../../us/20261123_observation_transport_v11/B_gaussian_s17/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s17/metrics.json](../../us/20261123_observation_transport_v11/B_gaussian_s17/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s17/all_months24/config.json](../../us/20261123_observation_transport_v11/B_gaussian_s17/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s17/all_months24/metrics.json](../../us/20261123_observation_transport_v11/B_gaussian_s17/all_months24/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s42/config.json](../../us/20261123_observation_transport_v11/B_gaussian_s42/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s42/metrics.json](../../us/20261123_observation_transport_v11/B_gaussian_s42/metrics.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s42/all_months24/config.json](../../us/20261123_observation_transport_v11/B_gaussian_s42/all_months24/config.json)（metadata；read）
- [experiments/us/20261123_observation_transport_v11/B_gaussian_s42/all_months24/metrics.json](../../us/20261123_observation_transport_v11/B_gaussian_s42/all_months24/metrics.json)（metadata；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
