# weather_native_20261126

当前公开来源：`research/weather_native_20261126`；[来源目录](../../../research/weather_native_20261126)。

历史路径：`maintenance/weather_native_20261126`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第1行：# 执行记录：weather-native-diffusion

## 类别依据

未定位到明确证据。

## 状态依据

- [research/weather_native_20261126/event_audit_lock.json](../../../research/weather_native_20261126/event_audit_lock.json)，status：FROZEN_BEFORE_EVENT_SCORE_READ

## 报告中的阶段声明

- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第7行：- 算子RED：缺少native_weather模块时4测试收集失败；GREEN：4通过。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第8行：- 时间/均值/扩散RED：缺少模型模块时收集失败；GREEN：10通过。重排和因果测试作用于中间状态及非零均值头。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第9行：- 评估RED：缺少评估模块时收集失败；GREEN：3通过。宏指标保持每布局每站人口，日历空档不连增量。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第10行：- Copula RED：缺少实现时收集失败；GREEN：2通过。线性因子精确复原空间×时间协方差、边际方差1。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第12行：- 正式seed42缓存已启动3000连续12h块；主日历另建缓存，不读取新确认集。两个大缓存同时运行出现显存接近8GB和吞吐下降，故停止开发缓存进程并保留块级检查点，后续原协议续跑；科学配置未改。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第15行：- 额度接口初始和最近复核均已用13%、剩余87%；停止线已用25%。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第18行：- 旧seed42缓存1685块后因2018-04-29原始产品缺测停止。四个原始gz文件第119天0记录、前后各24记录；另2019-12-03局部缺测。全训练时间轴45个起点不足128条完整7h轨迹、6个起点无完整轨迹。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第22行：- seed42 v2训练缓存3000块已完成；主日历v2从原34块前缀续建中（进程由campaign calendar管理）。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第23行：- seed42主日历216块现已完成并逐值审计；I1最大复现差0.00022316 mm。全训练3000块的标签及物理组隔离审计通过。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第30行：- 独立float64分析核对全部已完成均值输出一致；旧curve缺新字段以NaN显示、不补造。月份bootstrap以有效平方误差和/计数重采样，缺测覆盖不同时的回归测试RED→GREEN；公开JSON空组和开放边界写null。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第32行：- seed17训练3000块及主日历216块均完成，逐值审计通过，I1主日历复现最大差0.00016785 mm。seed17首轮六模型已启动。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第33行：- seed42十二组均值模型共36000连续块已完成，24组raw/EMA评价完成。完整联合方案raw：TF R/F6=9.865004/14.174626，GRU=9.869019/14.241080，LSTM=9.858661/14.326339 mm。

## 历史证据范围与信息条件

- [research/weather_native_20261126/event_audit_lock.json](../../../research/weather_native_20261126/event_audit_lock.json)，scope：post-model-selection exposed-development diagnostic; all thresholds and signs retained; no multiplicity adjustment
- [research/weather_native_20261126/weather_gp_supplement_lock.json](../../../research/weather_native_20261126/weather_gp_supplement_lock.json)，development_exposed：True
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第5行：- 用户补充目标：逐站误差、GNSS/T/P、mask/age、身份与方向→相同历史的GRU/LSTM/因果Transformer→地形/海岸条件查询注意力；均值头零初始化。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第6行：- 沿用现有`codex/p14-continuous-fusion`工作分支及原路径数据，新增独立文件和结果目录，旧模型/科学锁不改写。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第16行：- 独立审阅发现均值训练缺开发角色guard、cache loader缺协议/几何核验；4项RED复现后已GREEN。未有正式均值/概率模型在修复前训练；此前均为明确标注的小试。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第30行：- 独立float64分析核对全部已完成均值输出一致；旧curve缺新字段以NaN显示、不补造。月份bootstrap以有效平方误差和/计数重采样，缺测覆盖不同时的回归测试RED→GREEN；公开JSON空组和开放边界写null。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第37行：- 新增节点单位/mask与fair energy直接成员和核验测试，维护目录合计9项通过；独立概率联合评分已开始对Gaussian实际样本复算。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第42行：- 续接核验：同历史12配置、C/H/S、原GP过程及GraphAffine双种子均已完成；更新统一入口2026-11-26-weather-projective-trajectory.md。Full近邻收益13%–15%，图Gaussian约11%，边际/energy代价单列。
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)，第45行：- 独立review数学通过；补G265/seed、最终权重seal、完整评价marker与训练样本可用性描述。研究168测试49.90s、默认568测试293.82s、维护12测试0.45s通过。新守卫真实缺测两臂4更新1435轨迹及137-query一致性复核通过，旧源码按SHA归档。
- [research/weather_native_20261126/engineering_snapshots/weather_gp_v1/README.md](../../../research/weather_native_20261126/engineering_snapshots/weather_gp_v1/README.md)，第5行：后续新增mask-poison回归测试发现：直接统计原始15维状态，会把缺测变量的占位值计入全局摘要；特别是旧缓存的缺测GNSS后I1误差通道可以保留非零校正占位。下一版摘要先按各变量mask清理数值与两类GNSS误差，再统计均值/总体标准差，mask与age仍保留。初版代码与证据按原哈希保留，正式GP训练使用修正后源码和独立小试目录。

## 原始协议、结果与报告

- [research/weather_native_20261126/directed_lock.json](../../../research/weather_native_20261126/directed_lock.json)（protocol；read）
- [research/weather_native_20261126/directed_v2_lock.json](../../../research/weather_native_20261126/directed_v2_lock.json)（protocol；read）
- [research/weather_native_20261126/event_audit_lock.json](../../../research/weather_native_20261126/event_audit_lock.json)（protocol；read）
- [research/weather_native_20261126/increment_lock.json](../../../research/weather_native_20261126/increment_lock.json)（protocol；read）
- [research/weather_native_20261126/independent_pair_centering_lock.json](../../../research/weather_native_20261126/independent_pair_centering_lock.json)（protocol；read）
- [research/weather_native_20261126/inducing_flow_lock.json](../../../research/weather_native_20261126/inducing_flow_lock.json)（protocol；read）
- [research/weather_native_20261126/innovation_panel_lock.json](../../../research/weather_native_20261126/innovation_panel_lock.json)（protocol；read）
- [research/weather_native_20261126/lead_mean_audit_lock.json](../../../research/weather_native_20261126/lead_mean_audit_lock.json)（protocol；read）
- [research/weather_native_20261126/progress.md](../../../research/weather_native_20261126/progress.md)（documentation；read）
- [research/weather_native_20261126/supplement_lock.json](../../../research/weather_native_20261126/supplement_lock.json)（protocol；read）
- [research/weather_native_20261126/verification.json](../../../research/weather_native_20261126/verification.json)（metadata；read）
- [research/weather_native_20261126/weather_calibration_lock.json](../../../research/weather_native_20261126/weather_calibration_lock.json)（protocol；read）
- [research/weather_native_20261126/weather_gp_supplement_lock.json](../../../research/weather_native_20261126/weather_gp_supplement_lock.json)（protocol；read）
- [research/weather_native_20261126/engineering_snapshots/event_audit_v1/README.md](../../../research/weather_native_20261126/engineering_snapshots/event_audit_v1/README.md)（documentation；read）
- [research/weather_native_20261126/engineering_snapshots/weather_gp_masked_v2/README.md](../../../research/weather_native_20261126/engineering_snapshots/weather_gp_masked_v2/README.md)（documentation；read）
- [research/weather_native_20261126/engineering_snapshots/weather_gp_v1/README.md](../../../research/weather_native_20261126/engineering_snapshots/weather_gp_v1/README.md)（documentation；read）
- [research/weather_native_20261126/engineering_snapshots/weather_transport_v1/README.md](../../../research/weather_native_20261126/engineering_snapshots/weather_transport_v1/README.md)（documentation；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
