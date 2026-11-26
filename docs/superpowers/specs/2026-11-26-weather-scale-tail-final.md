# 最后一轮：天气条件Gaussian尺度与同协方差Student-t

用户要求完成本轮后停止新增实验，汇总全部成果、最佳架构和精度。本合同收束为已有Affine time完整Gaussian场上的天气尺度/尾部配对，不新增频域或新均值训练。原75%剩余额度停止线继续约束新增研究。

## 来源与阶段

冻结两种子Affine time原权重、TF3000均值mu和完整协方差C。历史天气为原72h允许观测的90维摘要；无未来气象、PWV/IWV/TCWV/Tm或确认/C/D标签。训练2016–2019原3000连续块，Q32按原固定随机规则选站；旧未来可用性Q128筛选、strict_oof=false和末91块复用限制保留。

1. 预计算每起点q=e^T C^-1 e、logdet(C)、有效维数d、90维天气与原始起点。mask仅用于评分子向量，全空起点不制造标签。
2. G_weather用天气MLP学习s²(c)，分布N(mu,s²(c)C)，3000块。
3. 冻结G_weather，新增同参数MLP分别训练G_extra和T_weather，各3000块。G_extra进一步学习k²(c)，分布N(mu,s²k²C)；T_weather学习nu(c)，分布为均值mu、协方差s²C的Student-t。T_weather与G_weather严格同均值/协方差；G_extra提供相同新增参数/预算的Gaussian控制。
4. 加入固定Affine、训练内常数尺度Gaussian及在s²C上拟合的常数nu探针，分开登记解析拟合与神经3000块。

三个新增天气网络均为90→64→64→1，SiLU，参数完全相同；天气归一化统计仅用完整训练历史、不按目标标签可用性筛选。s²/k²=exp(tanh(raw))，初始化1；nu=3+997sigmoid(raw)，初始化20。均值始终冻结。AdamW lr3e-4、weight_decay0、cosine3000至3e-5、clip1，raw final3000，不选EMA。实际监督次数由d>0重建，预计2999/3000，与原Q32一致。

损失为有效起点的联合NLL/d。所有计数进入公式前转float64；Student shape=(nu−2)/nu×s²C，而非把shape直接当协方差。

## 评价与主比较

2020每月8–10日三布局，全137注册查询、K32 IID；已开发暴露，不称盲测。复用原Affine成员，确定性天气尺度作用于残差；Student另以独立封存Gamma噪声构造sqrt((nu−2)/chi-square(nu))，在同一成员/起点内共享所有查询和0–6h。总体均值mu、协方差s²C保持，不裁剪负值。

主比较为T_weather相对G_weather的六小时fairCRPS；G_extra为额外Gaussian预算控制，常数nu用于天气尾部消融。联合NLL/d、小时增量/站对CRPS、energy、20mm事件Brier、秩覆盖和负样本比例并列解释，不能用对数评分替代精度改善。

本轮不新增补充日期；按用户要求完成本轮后统一总结。全部模型/来源/缓存/代码和采样种子绑定，结果生成前固定主比较，不按结果挑选候选。独立起点成员不拼成24h天气轨迹。

## 完成要求

数学与mask测试、真实4更新、双种子3000块、原点均值逐值核验、匹配Gaussian/Student概率评价、独立只读复核及Git同步。最终报告覆盖此前全部研究方向，区分最佳确定性架构与不同来源的概率分支，不把未重训的9.806mm均值与旧概率成绩相加。
