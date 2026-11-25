# GP参考轨迹过程实施计划

> **For agentic workers:** Use superpowers:executing-plans。用户已授权自主研究与实验。

**Goal:** 在强相关Gaussian参考上验证完整白化和查询集合一致的局部轨迹过程，区分条件Gaussian与非线性分布收益。

**Architecture:** 冻结I1/Transformer均值＋S175 Gaussian → 训练copula对称根 → Full/Local/Affine v → Heun0 → 逆变换mm。

**Tech Stack:** 现有pytorch环境、RTX4060、NumPy/SciPy/Matplotlib。

**Spec:** [公式、输入与对照合同](../specs/2026-11-26-gp-trajectory-process.md)。

## Global Constraints

3000连续12h块；只抽站身份；禁止PWV/IWV/TCWV/Tm。c=S175，源均值/Gaussian/copula冻结。2020已暴露开发、strict_oof=false；确认/C/D隔离。剩余额度75%停止新增研究。只接受唯一站ID，nugget作为观测残差解释。

## Review Focus

1. 缺测分组后白化，sigma元素运算与矩阵顺序正确。
2. 零速度与同根/同噪声Gaussian参考逐元素相同，不重复施加空间相关。
3. Local无query间归一化，路径一致检验使用共同superset GP初值。
4. Affine固定c严格线性于状态，参数接近Nonlinear；报告39/40实际NFE。
5. 未来标签/监督mask不改变推断条件、核参数和固定查询人口。

## Task 1：矩阵、采样与局部模型

- [x] 新建`native_process_ops.py`、`native_process_models.py`和测试；RED验证缺模块。
- [x] 主值根/逆根、双向恢复、异方差、Q1/重复坐标/SPD测试GREEN。
- [x] Heun0拒绝k0神经调用、零速度保持参考；Local非零权重子集路径一致，Affine采样映射仿射。
- [x] 逐完整轨迹分组loss与手动计算一致，NaN不传播。

## Task 2：独立真实运行入口

- [x] 新建`native_process_experiment.py`，复用缓存/源embedding；训练角色、源权重、copula和归一化强核验。
- [x] 协议登记三臂参数、精确noise端点、共同k[B]/epsilon[B,Q,7]及初始化后噪声seed。
- [x] reference/evaluate固定全部137查询，无未来mask选点；保存同格式概率数组、指标及来源哈希。
- [x] 三臂真实缺测块各4更新、真实冻结S175 Gaussian小试通过后提交代码；代数正确不能替代精度验证。

## Task 3：配对3000块与区域证据

- [ ] 先S175训练copula及G-ref对称根评价，再三臂seed42各3000块和Heun0/DDIM对照。
- [ ] float64复算、共同月份区间、尾部/覆盖/站对/增量/energy、Q人口诊断。
- [ ] 候选seed17及冻结补充日期；归档曲线/流程图/参数与来源；Git推送后核对远端一致。

## 当前执行证据

真实3块缺测缓存、完整TF冻结均值、完整训练S175 Gaussian归一化：Full/Local/Affine各4更新全部通过；每臂共1435条完整7h轨迹，mask只参与监督。新入口CPU测试先3项RED，再3项GREEN；相关代数与模型共10项通过。正式3000块及137查询评价由独立process_campaign.py执行，结果另行登记。
