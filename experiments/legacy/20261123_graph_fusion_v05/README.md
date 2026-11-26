# 20261123_graph_fusion_v05

当前公开来源：`experiments/us/20261123_graph_fusion_v05`；[来源目录](../../us/20261123_graph_fusion_v05)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261123_graph_fusion_v05`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第1行：# 图编码观测残差小试

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第10行：- 四次训练全部完成且检查点锁定后，评价2018每月22/27日原固定 B-outer 人口。该日历已有开发暴露，属于适应性探索，不作为盲测。不得按外层成绩选择超参或延长训练。

## 历史证据范围与信息条件

- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第3行：2026-11-23，训练前锁定。目的：检验 A 站之间的图消息传递是否在 v04 历史创新校正之后仍有增益。原始投稿稿件只在本地读取，不上传原文、图片或抽取文本。
- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第5行：- 复用 v03 的训练/内层时段与 B 空间分组，A174 为唯一输入。Z0/T/P 与当前、1h、3h 观测创新；历史队列含 PWV 筛选限制继续适用。
- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第6行：- 冻结 v03 均值与随机分支以及 v04 历史 ridge。新网络只学习额外的共同高度校正，先在统一格网节点输出，再按四角查询和 exp(-H/3000) 变换。
- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)，第7行：- 两臂：相同参数的节点独立编码（self），以及一层 12 邻居含自身的边条件注意力（graph）。两者共同使用 12 邻居 A→格网距离读出、地形查询特征及 sigmoid 门控。区别仅为 A→A 消息来源与对应边。
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)，第3行：本轮测试了一层图注意力是否能继续改善已有历史创新校正。两种子结果支持保留原 v04 候选，没有证据支持将此次图模块扩成全量训练。当前固定人口尚未达到 8 mm。
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)，第21行：输入为 A174 的 Z0/T/P、当前与 1h/3h 创新及有效掩码。新模块学习格网节点上的额外校正，保留 v03 均值、v04 历史 ridge 及相关随机残差生成器。A→A 图采用 12 邻居、一层单头注意力；dx、dy 和高度差分别缩放。A→格网使用距离权重及地形条件门控，统一四角查询后恢复站点高度。
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)，第27行：生成分支保持冻结，K64 与独立噪声 K128 均加入同一确定性校正。概率评分见 results.json 和 verification.json。此操作保留随机时间相关结构，区间宽度保持不变；不表示图模块改善了生成机制，也不是新 GAN、VAE 或 Diffusion 训练。
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)，第35行：这仍是已暴露日历上的适应性探索。模型值输入排除 PWV，但历史制包的队列筛选曾依赖 PWV；正式训练仍须重建允许变量质控。
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)，第43行：论文目标需要区分站点精度、格网间距和实际可恢复的细尺度结构。固定站点 RMSE 是必要证据；高分辨率输出还需通过空间整块留出、站网稀疏度变化及天气过程增量验证。当前没有独立稠密真值，不能把更密格网或更丰富纹理直接解释为真实细节。
- [experiments/us/20261123_graph_fusion_v05/REVIEW.md](../../us/20261123_graph_fusion_v05/REVIEW.md)，第1行：# 独立审查
- [experiments/us/20261123_graph_fusion_v05/REVIEW.md](../../us/20261123_graph_fusion_v05/REVIEW.md)，第5行：初审发现协议要求 K128，但主训练程序仅评价 K64。通过独立 verify_graph_fusion.py 重生成固定噪声11703的128成员并施加同一校正补齐；不改变训练和检查点选择。复审确认缺口关闭。
- [experiments/us/20261123_graph_fusion_v05/REVIEW.md](../../us/20261123_graph_fusion_v05/REVIEW.md)，第9行：复审提出补充协议、数据、v04选择文件和源检查点哈希核验，已加入独立核验程序。实际核验结果见 verification.json。

## 原始协议、结果与报告

- [experiments/us/20261123_graph_fusion_v05/PROTOCOL.md](../../us/20261123_graph_fusion_v05/PROTOCOL.md)（documentation；read）
- [experiments/us/20261123_graph_fusion_v05/RESULTS.md](../../us/20261123_graph_fusion_v05/RESULTS.md)（report；read）
- [experiments/us/20261123_graph_fusion_v05/REVIEW.md](../../us/20261123_graph_fusion_v05/REVIEW.md)（report；read）
- [experiments/us/20261123_graph_fusion_v05/experiment.lock.json](../../us/20261123_graph_fusion_v05/experiment.lock.json)（protocol；read）
- [experiments/us/20261123_graph_fusion_v05/results.json](../../us/20261123_graph_fusion_v05/results.json)（metadata；read）
- [experiments/us/20261123_graph_fusion_v05/selection.lock.json](../../us/20261123_graph_fusion_v05/selection.lock.json)（protocol；read）
- [experiments/us/20261123_graph_fusion_v05/verification.json](../../us/20261123_graph_fusion_v05/verification.json)（metadata；read）

## 待核查项

- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
