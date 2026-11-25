# Weather Projective GP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. 用户已授权自主设计并执行；本计划不要求重复确认。

**Goal:** 实测天气条件、非分离时空相关且查询人口一致的Gaussian先验，再评价Local扩散增益。

**Architecture:** 冻结源均值/S175边际；逐站/逐lead相关核＋可见历史90维摘要生成时间相关；SPD分解采样。GP冻结后逐站7h白化、Local/Affine扩散、逆变换mm。

**Tech Stack:** 现有Windows pytorch环境、RTX4060、NumPy/SciPy/Matplotlib。

**Spec:** [精确公式、初始化与实验合同](../specs/2026-11-26-weather-projective-gp.md)。

## Global Constraints

- 不修改已运行的native_process、GraphAffine或均值v2源码/科学锁；只新增模块和独立runner。
- 同3000连续12h块、72h状态、146h原始跨度；未来mask仅用于监督；推断137查询。
- 输入仅GNSS/T/P、mask/age、身份、时钟、地形/海岸及历史误差。禁止PWV/IWV/TCWV/Tm。
- 2020已暴露、strict_oof=false；确认138及原C/D隔离；75%预算停止线；轻量证据Git同步。

## Review Focus

1. cw90及时间T不能读取查询人口统计、未来标签或监督mask。
2. 固定输入后的核主子矩阵一致；同坐标不同ID的时间块nugget正确，重复ID拒绝。
3. 初始化Ks⊗T_ref、单位对角、SPD及有限梯度；显式5%时间收缩与rho上限不是临时jitter。
4. 不跨矩阵交换逐元素sigma；Local逐query的7×7根白化顺序正确。
5. 区分解析Gaussian、K32分位覆盖、固定superset路径和不同根同IID seed；记录真实NFE。

## Task 1：核与CPU数学合同

**Files:** 新建`tools/terrain_diffusion/native_weather_gp.py`和`test_native_weather_gp.py`。

**Interfaces:** `weather_summary(history[B,L,C,15])->[B,90]`；`WeatherProjectiveGP(view,copula).correlation(c175,history,static,query_ids)->[B,7Q,7Q]`。view为175或265；类别同时提供`query_time_blocks(...)->[B,Q,7,7]`，不得通过全Q归一化计算。

- [ ] RED：摘要形状/历史截断、初始Kronecker参考、单位对角、SPD、重排/主子矩阵/重复坐标ID、非零梯度测试缺实现。
- [ ] GREEN：实现源L0、两种节点网络、可见历史全局时间头、显式PSD核；参数应为43893/44027，若实现需要增参先登记理由。
- [ ] 对学习后非零参数，改变其他查询及未来监督mask仍保持目标站K块；用float64小矩阵直接NLL复核。
- [ ] 运行新测试与现有native数学/模型相关测试，核验后提交。

## Task 2：独立数据与训练入口

**Files:** 新建`native_weather_gp_experiment.py`、相应test，以及`maintenance/weather_native_20261126/weather_gp_campaign.py`。

**Interfaces:** 复用`native_process_experiment.Sources`取得冻结c175/gm/gs/base与来源receipt；通过其CachedData.batch取得同一history。实现`masked_gp_nll(model,u,c175,history,static,query_ids)`，先分组完整轨迹再构核，返回loss/轨迹数/起点数。

- [ ] RED：开发数据训练拒绝、非训练归一化拒绝、缺测不传播、只读源哈希改变拒绝、全缺测不更新优化器。
- [ ] GREEN：独立协议/代码哈希、3000连续块、500步checkpoint、STOP文件、实际步数/耗时/曲线；source Gaussian不参与梯度。
- [ ] 真实gap cache上两臂各4更新，完整训练归一化复用；参数、warm covariance差与梯度实测归档。
- [ ] 测试与源权重guard通过后提交，再执行两臂seed42各3000；完成后按固定方案复核seed17。

## Task 3：采样、区域与Gaussian机制结论

**Files:** 同独立runner的evaluate及`maintenance/.../weather_gp_summary.py`。

- [ ] 固定137查询，float64相关Cholesky、K32 IID；恢复`s*d*u`时保留原f/b/s/g/d元素次序；解析Gaussian边际应与S175逐值一致。
- [ ] 共同superset噪声限制后的GP/Local路径、K主子矩阵及不同时效相关图；不同根同seed不作为路径一致性判断。
- [ ] 保存同格式evaluation.npz与lineage，复用现有joint audit，给四档站对、增量、energy、边际覆盖及月份配对区间。
- [ ] 主要判断天气条件相关是否复现近邻收益并保持边际/查询一致性；冻结候选后追加日期验证，更新图与Git来源。

## Task 4：Local非线性与Affine增量

**Files:** 新建`native_weather_gp_diffusion.py`及test、独立runner模式；不编辑旧过程模块。

- [ ] RED：逐站Aq白化/恢复、零v继承GP、非零Local共同路径一致性、Affine映射Gaussian性、来源与参数匹配。
- [ ] GREEN：c265条件、同噪声端点/共同训练流、Heun0×20与DDIM40。训练目标仅7维边际score，初值来自完整天气GP。
- [ ] 真实4更新后各3000连续块；报告相对天气GP和近等参数Affine的额外收益，不能将Gaussian先验收益写成扩散贡献。

## 后续均值任务（独立计划）

先根据已完成H/S对照新建H+S的零影响组合，验证当前R、F6和小时增量；胜出后再重制训练embedding/归一化并重训概率头。该工作不与核实现混合，避免改变本轮冻结来源。

## 执行细化与已验证证据

初版真实缺测两臂各4更新通过；随后mask-poison回归测试RED→GREEN，先按mask清理全局摘要，再重跑两臂各4更新，每臂1435条完整轨迹，warm covariance最大差7.07515e-8。初版与mask版源码按协议哈希分别保存在engineering_snapshots。主日历参考137查询、K32完整采样与解析站对/增量已真实通过。

lead依赖诊断显示固定300km/1500m尺度无法同时匹配当前与6h依赖，新增21个按lead的基础长度参数；归一化Gaussian卷积核及平移不变性测试RED→GREEN。新参数量43893/44027，后续正式训练使用最后核验的metric版本，不混用早期小试权重。metric学习率1e-3，其余3e-4；每组最低率为初值0.1倍。
