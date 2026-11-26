# 20261123_simple_weather_unet_v02

当前公开来源：`experiments/us/20261123_simple_weather_unet_v02`；[来源目录](../../us/20261123_simple_weather_unet_v02)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_simple_weather_unet_v02`。

用途类别：`diagnostic`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第1行：# 本机简单U-Net：现有气象变量的配对验证

## 类别依据

- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第17行：输出锁文件、数据来源审计、训练日志、权重、逐站预测、损失/验证曲线、误差与场图。月份事件块置信区间用于本轮小样本诊断，区别于完整全年7日bootstrap；若收益方向不稳定，不扩展编码器。候选参数与优化预算已在训练前登记，不根据最终留出结果反复调参。

## 状态依据

- [experiments/us/20261123_simple_weather_unet_v02/results.json](../../us/20261123_simple_weather_unet_v02/results.json)，status：COMPLETE
- [experiments/us/20261123_simple_weather_unet_v02/status.json](../../us/20261123_simple_weather_unet_v02/status.json)，stage：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第11行：固定种子42及17，每组最多1000次更新，AdamW、学习率0.001随本轮预算余弦下降至0.00002，batch8，宽度16/32/64、双层卷积及GroupNorm/SiLU、镜像U-Net。以内部B站验证选模，C/D不参与选模。暂停文件为结果目录内PAUSE；每100步保存状态与曲线。
- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第15行：最终分别报告：新站同训练天气、新天气拟合站、新站新天气；C2019固定月度小时可在B选模结束后作开发确认，D2020保持封存。比较A粗格双线性、A直接IDW格网、ERA5站位PWV/Tm换算插值和两组U-Net。全图先发布再统一站高查询，非有限预测触发失败。

## 历史证据范围与信息条件

- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第5行：v02在跨年度训练时刻上统一应用18小时验证缓冲；v01因独立审计发现2017年末边界缺口已标记无效并保留追溯。v02在重新读标签、重新训练前锁定新时刻，不继承v01权重。
- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第7行：采用原生PyTorch与本机RTX4060，独立试验脚本，从头训练。气象来源使用现有GPAC-Repro产品在A站保存的ERA5温度、气压、Tm和PWV；用户已确认“就是现在这个数据”。这些是气象产品站位采样，不能称为独立完整气象格网。
- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第9行：配对比较：GNSS＋地形的小U-Net，以及增加四项气象变量的同规模U-Net。保留四帧0/6/12/18小时，144×96完整发布格与已有公共高度查询。所有模型从A站IDW粗格起步，零初始化改正头；不使用历史学习底座、岭回归、空间基或权重。气象删除组保留同一输入宽度，将对应通道置零，使初始化和参数量相同。
- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)，第15行：最终分别报告：新站同训练天气、新天气拟合站、新站新天气；C2019固定月度小时可在B选模结束后作开发确认，D2020保持封存。比较A粗格双线性、A直接IDW格网、ERA5站位PWV/Tm换算插值和两组U-Net。全图先发布再统一站高查询，非有限预测触发失败。
- [experiments/us/20261123_simple_weather_unet_v02/RESULTS.md](../../us/20261123_simple_weather_unet_v02/RESULTS.md)，第25行：- C2019为每月4个固定时刻，共48小时；不是全年，不能与旧全年数字直接排序。
- [experiments/us/20261123_simple_weather_unet_v02/RESULTS.md](../../us/20261123_simple_weather_unet_v02/RESULTS.md)，第26行：- 月份事件块配对区间用于小试诊断，不代替完整全年7日块bootstrap。两种子是优化稳定性检查，不是独立气象样本。

## 原始协议、结果与报告

- [experiments/us/20261123_simple_weather_unet_v02/PLAN.md](../../us/20261123_simple_weather_unet_v02/PLAN.md)（documentation；read）
- [experiments/us/20261123_simple_weather_unet_v02/RESULTS.md](../../us/20261123_simple_weather_unet_v02/RESULTS.md)（report；read）
- [experiments/us/20261123_simple_weather_unet_v02/experiment.lock.json](../../us/20261123_simple_weather_unet_v02/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_simple_weather_unet_v02/results.json](../../us/20261123_simple_weather_unet_v02/results.json)（metadata；read）
- [experiments/us/20261123_simple_weather_unet_v02/status.json](../../us/20261123_simple_weather_unet_v02/status.json)（status；read）

## 待核查项

已定位基本导航证据；科学结论以原报告、协议与审阅为准。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
