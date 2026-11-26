# 有向时滞v2：变量独立归一化修正

## 继承的科学合同

沿用[原架构与比较合同](2026-11-26-directed-lag-transport.md)：H_delta、factorized、joint，两个新臂同参数117757，各从原TF3000来源新增3000，保持原连续块、loss/aux/Adam/RNG及raw主EMA并列。该版本修复独立mask数值归一化，不改变研究假设、变量、lag、网络参数或主指标。

## v1工程停止依据

只读审阅复现：唯一GNSS站距查询350km、创新恒1mm时，正确额外输出为3*tanh(1/3)=0.964538mm。增加近处仅温度有效且增量为零的站，v1输出变为0.004579mm。共用指数最大值使GNSS权重和过小，再被1e-8下限稀释。

两个v1 factorized任务在1532/1530更新停止，原源码、lock、checkpoint和负工程证据保留，不作为正式3000成绩。旧状态名STOPPED_AT_BUDGET源于复用现有控制标记，engineering_stop.json明确这是工程纠错停止，非账户75%阈值。joint v1未正式开始。

## 数值修正

每个变量v在自己有效的站点×lag集合上选最大score，计算指数权重并归一。存在支持时最大权重为1、分母至少1；无支持时分子/分母均0并显式返回0。温压的可用性不会单凭指数尺度下限改变GNSS常量读出。数学上仍是同一score下的变量独立masked softmax。

新模型文件native_directed_lag_v2.py独立继承v1的H前向和edge定义，覆写数值池化。构造参数次序与v1一致，初始state_dict逐tensor相同；旧源码不回改。训练/导出/评价采用新的native_directed_lag_v2_experiment.py及20261126_directed_lag_v2结果目录。重新从原TF3000训练，不续v1中间权重。

## 验收与结果范围

350km异步缺测回归在v1先失败、v2通过；全空变量输出和反传有限；warm、查询子集、站点排列、参数初始化及激活梯度保持。真实4更新的disabled共同H_delta状态、按名称Adam、EMA、scheduler和RNG再次精确等价，两启用臂梯度非零。joint峰值分配约1504MiB。

主评价、补充触发、来源与数据角色、确认/C/D隔离均沿原合同。源码和新lock在v2正式训练前固定；独立审阅确认修复后再执行正式3000。原缓存限制、已暴露2020、strict_oof=false及剩额75%停止线保持。
