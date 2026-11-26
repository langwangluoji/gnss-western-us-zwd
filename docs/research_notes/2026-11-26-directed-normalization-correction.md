# 有向时滞分支：异步缺测归一化修正

在最终只读审阅中，真实PyTorch反例定位到变量独立归一化的数值问题。GNSS唯一有效站距350km、创新为1mm；新增近处仅温度有效、增量为0的站后，v1的GNSS常量读出从0.964538mm变为0.004579mm。原因是所有变量共享指数最大值，各变量分母再使用1e-8下限，稀释了低score但合法的GNSS支持。

原单测覆盖全空支持和均匀缺测，未覆盖这种空间距离与变量异步可用性的组合。新的回归先复现该失败，再在独立v2中按每变量有效支持选择最大值、归一化，空支持显式为零，修复后测试通过。

两个v1正式队列在1532/1530更新工程停止，checkpoint、源码和lock原样留存。旧status名STOPPED_AT_BUDGET来自共用停止控制接口，外层engineering_stop.json记录实际原因及quota未达到；不能解释成已达到账户75%停止线。v1没有完整正式指标，joint v1尚未开始。

v2保留相同参数创建顺序、H前向、全部输入与实验比较，仅替换有问题的数值池化。真实4更新重新核验disabled共同H_delta权重、EMA、AdamW、scheduler与RNG精确等价，两启用模式新参数有梯度。新的正式任务将从原TF3000重新完整训练，不续用v1中间权重。架构PNG/SVG仍对应相同数学合同。

- [v1工程停止记录](../../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_directed_lag/engineering_stop.json)
- [v2原理、版本与来源](../superpowers/specs/2026-11-26-directed-lag-v2.md)
- [v2执行记录](../superpowers/plans/2026-11-26-directed-lag-v2.md)
- [v2真实4更新等价与显存](../../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_directed_lag_v2_smoke/equivalence.json)

旧科学文件、数组与权重保留本机，Git同步复现代码、轻量证据和修正原理；确认及原C/D未读取。
