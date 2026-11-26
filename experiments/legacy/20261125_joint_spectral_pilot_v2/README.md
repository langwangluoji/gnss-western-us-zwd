# 20261125_joint_spectral_pilot_v2

当前公开来源：`experiments/us/20261125_joint_spectral_pilot_v2`；[来源目录](../../us/20261125_joint_spectral_pilot_v2)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261125_joint_spectral_pilot_v2`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261125_joint_spectral_pilot_v2/REPORT.md](../../us/20261125_joint_spectral_pilot_v2/REPORT.md)，第1行：# 时空融合与谱残差：真实本机工程验证

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261125_joint_spectral_pilot_v2/protocol_all.json](../../us/20261125_joint_spectral_pilot_v2/protocol_all.json)，status：diagnostic pilot
- [experiments/us/20261125_joint_spectral_pilot_v2/status.json](../../us/20261125_joint_spectral_pilot_v2/status.json)，status：COMPLETE_ALL
- [experiments/us/20261125_joint_spectral_pilot_v2/status.json](../../us/20261125_joint_spectral_pilot_v2/status.json)，stage：local engineering pilot
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/status.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/status.json)，status：COMPLETE_4_VELOCITY_CONTROLS

## 报告中的阶段声明

- [experiments/us/20261125_joint_spectral_pilot_v2/REPORT.md](../../us/20261125_joint_spectral_pilot_v2/REPORT.md)，第39行：- 综合收集测试时定位到旧test_fixed_support向sys.path插入子目录、遮蔽标准库profile的错误；改成包相对导入。第一次合并运行88项失败的原始日志保留，修正后另跑完整测试。

## 历史证据范围与信息条件

- [experiments/us/20261125_joint_spectral_pilot_v2/REPORT.md](../../us/20261125_joint_spectral_pilot_v2/REPORT.md)，第27行：全部为mm。R为当前场重建，F为起点持久性均值上的未来6小时预测；两任务人口与信息条件不同。当前表示均未满足相应正式门槛，概率训练使用显式diagnostic覆盖，不能据此宣布新模型优于原p14。L1与L0的容量及训练状态也应在正式重复中匹配。
- [experiments/us/20261125_joint_spectral_pilot_v2/REPORT.md](../../us/20261125_joint_spectral_pilot_v2/REPORT.md)，第41行：保存预测独立复算共26组，RMSE/CRPS最大差3.77604e-05mm。训练曲线、分层与站对误差见相应JSON；小时增量只在连续6小时块内计算，跨块首态重置。当前片段仅1月，夏季误差和全年结果尚无此轮证据。

## 原始协议、结果与报告

- [experiments/us/20261125_joint_spectral_pilot_v2/REPORT.md](../../us/20261125_joint_spectral_pilot_v2/REPORT.md)（report；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/comparison.json](../../us/20261125_joint_spectral_pilot_v2/comparison.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/protocol_all.json](../../us/20261125_joint_spectral_pilot_v2/protocol_all.json)（protocol；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/status.json](../../us/20261125_joint_spectral_pilot_v2/status.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/verification.json](../../us/20261125_joint_spectral_pilot_v2/verification.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/A0/results.json](../../us/20261125_joint_spectral_pilot_v2/A0/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/A1/results.json](../../us/20261125_joint_spectral_pilot_v2/A1/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/A2/results.json](../../us/20261125_joint_spectral_pilot_v2/A2/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/B0/results.json](../../us/20261125_joint_spectral_pilot_v2/B0/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/B1/results.json](../../us/20261125_joint_spectral_pilot_v2/B1/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/J0/results.json](../../us/20261125_joint_spectral_pilot_v2/J0/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/J1/results.json](../../us/20261125_joint_spectral_pilot_v2/J1/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/L0/results.json](../../us/20261125_joint_spectral_pilot_v2/L0/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/L1/results.json](../../us/20261125_joint_spectral_pilot_v2/L1/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T0/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T0/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T1/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T1/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T2/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T2/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T3/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T3/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T4/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T4/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/foundation/T5/progress.json](../../us/20261125_joint_spectral_pilot_v2/foundation/T5/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/oof/fold0/progress.json](../../us/20261125_joint_spectral_pilot_v2/oof/fold0/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/oof/fold1/progress.json](../../us/20261125_joint_spectral_pilot_v2/oof/fold1/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/oof/fold2/progress.json](../../us/20261125_joint_spectral_pilot_v2/oof/fold2/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/oof/fold3/progress.json](../../us/20261125_joint_spectral_pilot_v2/oof/fold3/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/oof/fold4/progress.json](../../us/20261125_joint_spectral_pilot_v2/oof/fold4/progress.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/protocol.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/protocol.json)（protocol；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/status.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/status.json)（status；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/summary.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/summary.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/A0/results.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/A0/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/A2/results.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/A2/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/B1/results.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/B1/results.json)（metadata；read）
- [experiments/us/20261125_joint_spectral_pilot_v2/velocity_control/J1/results.json](../../us/20261125_joint_spectral_pilot_v2/velocity_control/J1/results.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
