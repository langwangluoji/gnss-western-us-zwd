# 20261121_grid_generation_v01

当前公开来源：`experiments/us/20261121_grid_generation_v01`；[来源目录](../../us/20261121_grid_generation_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261121_grid_generation_v01`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第1行：# 随机残差、Diffusion、GAN实施交接

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第3行：当前状态：已核验旧作业COMPLETED0:0、完整结果及下载SHA256，服务器新训练待部署。外层U-Net未晋级，共同均值明确锁定M0+IDW，旧U-Net权重只作审计。
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第13行：原5472训练小时完整覆盖一遍，每组684更新，batch8，运行时8小时并行；两折三臂上限4104更新。内层96小时K16，外层预登记96小时K32；C/D封存。单A800，累计分配上限4卡时，基准不容纳时停止，不自动追加预算。全部运行保持独立目录，不修改旧代码发布、旧缓存或环境。
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第15行：## 已完成验证
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第49行：提交程序核验旧队列COMPLETE、旧入选checkpoint哈希、新发布每个文件，申请单A800后先基准，再真实100+100对连续200恢复验收，再顺序训练fold0随机/DF/GAN及fold1随机/DF/GAN。任何失败保留checkpoint，不重复提交活动任务。
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第51行：暂停：在新GRID_RUN创建PAUSE；恢复：

## 历史证据范围与信息条件

- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第11行：两折共同冻结已核验M0+IDW基础场。新增独立随机ResU-Net、Diffusion采样参数化、GAN。共同69条件+相关先验场+白化状态，共71通道。Gaussian先验旁路使第0步三臂一致；tau仅在站端加入一次。Diffusion明确采用真实站ES监督的18步Heun采样，不声称独立HR真值去噪预训练。
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第13行：原5472训练小时完整覆盖一遍，每组684更新，batch8，运行时8小时并行；两折三臂上限4104更新。内层96小时K16，外层预登记96小时K32；C/D封存。单A800，累计分配上限4卡时，基准不容纳时停止，不自动追加预算。全部运行保持独立目录，不修改旧代码发布、旧缓存或环境。
- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)，第22行：- 独立审阅完成，基准恢复误计时问题已修复并回归，未解决Critical/Important=0。

## 原始协议、结果与报告

- [experiments/us/20261121_grid_generation_v01/HANDOFF.md](../../us/20261121_grid_generation_v01/HANDOFF.md)（documentation；read）
- [experiments/us/20261121_grid_generation_v01/local_engineering/data.lock.json](../../us/20261121_grid_generation_v01/local_engineering/data.lock.json)（protocol；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_0/mlp69/results.json](../../us/20261121_grid_generation_v01/previous/fold_0/mlp69/results.json)（metadata；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_0/mlp69/selection.lock.json](../../us/20261121_grid_generation_v01/previous/fold_0/mlp69/selection.lock.json)（protocol；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_0/resunet69/results.json](../../us/20261121_grid_generation_v01/previous/fold_0/resunet69/results.json)（metadata；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_0/resunet69/selection.lock.json](../../us/20261121_grid_generation_v01/previous/fold_0/resunet69/selection.lock.json)（protocol；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_1/mlp69/results.json](../../us/20261121_grid_generation_v01/previous/fold_1/mlp69/results.json)（metadata；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_1/mlp69/selection.lock.json](../../us/20261121_grid_generation_v01/previous/fold_1/mlp69/selection.lock.json)（protocol；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_1/resunet69/results.json](../../us/20261121_grid_generation_v01/previous/fold_1/resunet69/results.json)（metadata；read）
- [experiments/us/20261121_grid_generation_v01/previous/fold_1/resunet69/selection.lock.json](../../us/20261121_grid_generation_v01/previous/fold_1/resunet69/selection.lock.json)（protocol；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
