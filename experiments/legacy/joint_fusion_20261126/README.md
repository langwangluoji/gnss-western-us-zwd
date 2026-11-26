# joint_fusion_20261126

当前公开来源：`research/joint_fusion_20261126`；[来源目录](../../../research/joint_fusion_20261126)。

历史路径：`maintenance/joint_fusion_20261126`。

用途类别：`uncertain`。状态索引：`reported_prose`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第1行：# SDD ledger — plan: docs/superpowers/plans/2026-11-26-best-base-diffusion-fusion.md

## 类别依据

未定位到明确证据。

## 状态依据

未定位到明确证据。

## 报告中的阶段声明

- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第13行：Ruling: 首锁v1在正式训练前保留为lock_v1_before_review.json；审阅新增spec哈希、复用采样器/GP来源、实际标签时钟守卫。新增2项测试先失败后修正，再封存当前锁；旧v1未用于任何正式训练。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第15行：Task 1: complete — 当前锁SHA256=138d8fadd91dcbaf8a4a3681b5b18c8164cd8b7da6349cdec1d37a9ab92075bf；7项测试通过；双方训练导出完整，源joint开发导出身份与哈希通过。独立预审无数值/兼容性阻碍。GP和两种传输的真实四更新安排在各自冻结上游正式拟合之后，保留工程状态，不伪造3000步。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第19行：Task 2: in progress — 140bd62已推送并核对远端后启动两个原队列。S175双种子3000/3000完整监督均完成，来源/训练掩码/全部最终Adam时钟通过；copula均仅使用训练残差拟合。G265真实4更新6144完整轨迹通过，双种子正式G265在运行。RTX4060 Laptop运行正常，主要计算量为float64完整协方差分解。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第23行：Task 2 milestone — G265双种子3000/3000监督全部完成，完整轨迹数分别4602209/4602175。每个模型11个优化器参数全部有3000步Adam状态；最终权重与checkpoint、scheduler通过。Affine/Local按既定队列继续，每臂3000，核心锁138d8fad保持。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第25行：Task 2 training complete — 8/8正式模型全部完成。独立只读审阅从原truth重算支持、逐tensor权重和全部Adam状态（每种子S175/G265/Affine/Local为4/11/6/6组张量），均为3000，copula与锁来源一致。正式原队列session95277(seed42)、4929(seed17)继续评价两日历，禁止重复启动。主日历K32生成已推进至Local。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第29行：Task 2: complete — 原session95277/4929均正常退出0，两日历×两种子×三模型共12份K32评价及joint评分全部完成。主日历双种子F6 CRPS：G265=7.67524161、Affine=7.67298325、Local=7.65755655mm；Local增量CRPS3.14035754高于Affine3.11962944。此处为原始分数，主比较区间及K16尚待完整汇总。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第39行：Task 3: complete — 训练、两日历raw/pair评价、来源/实际点场/独立成员核对、区间、事件、分层、站点与PNG/SVG报告均完成。报告明确raw相对G265/强Affine的不同收益、K16后处理和单次MC界限。最终Git操作由交付会话执行并核对HEAD，不以后台启动代替完成。原队列与后处理进程均正常结束，没有追加实验。

## 历史证据范围与信息条件

- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第7行：Ruling: 普通生成K32为主；不重叠独立双场差分K16为中心保持二级对照，Gaussian/Affine/Local全部同规则，不以同批去均值伪造固定中心。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第9行：Task 1: in progress — 已读代码与数据合同；当前分支codex/p14-continuous-fusion已有隔离于main的研究开发历史。原目录存在无关untracked证据，不删除、不整体stage。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第11行：Task 1 evidence: 两种子3000块joint训练嵌入完整导出；原开发导出复用前核验。5项数学/来源测试RED→GREEN，seed42 Gaussian真实4更新6144完整轨迹通过。独立只读预审确认Sources兼容新joint与数值还原、差分概率机制。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第15行：Task 1: complete — 当前锁SHA256=138d8fadd91dcbaf8a4a3681b5b18c8164cd8b7da6349cdec1d37a9ab92075bf；7项测试通过；双方训练导出完整，源joint开发导出身份与哈希通过。独立预审无数值/兼容性阻碍。GP和两种传输的真实四更新安排在各自冻结上游正式拟合之后，保留工程状态，不伪造3000步。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第17行：Ruling: 独立审阅提出的K16秩覆盖、原joint数组直接核对、几何/nearest复验、重点站和训练曲线作为同实验只读补充交付，不改训练合同或主评分。正式队列启动前源码/来源提交推送。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第25行：Task 2 training complete — 8/8正式模型全部完成。独立只读审阅从原truth重算支持、逐tensor权重和全部Adam状态（每种子S175/G265/Affine/Local为4/11/6/6组张量），均为3000，copula与锁来源一致。正式原队列session95277(seed42)、4929(seed17)继续评价两日历，禁止重复启动。主日历K32生成已推进至Local。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第27行：Task 3 review note — diagnostics的K16秩混合、原点场及几何/站距合同独立审阅通过；图件manifest补绑定8份训练曲线。锁定campaign.py / fusion_contract.py / summary.py不再改写；额外后处理不改变科学合同。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第31行：Task 3: in progress — 原锁定summary正在session15210执行，逐个生成独立K16配对及joint评分，不得重复启动。接续顺序：summary退出0→diagnostics.py→render_delivery.py→独立终审与verify_delivery.py→核验Git同步。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第33行：Task 3 result — session15210正常退出0，diagnostics/render的session56117退出0。24组点场/真值直接等于原joint；12份K16与raw不重叠成员逐值重算一致。独立审阅重算F6、增量、近邻三指标及2000次共同月份区间通过（误差≤1e-8mm），无阻断问题。K32 Local−Affine主F6区间负、补跨零；K16单站两区间均跨零；近邻相对Affine两日历改善，但对G265须单独解释。增量与部分减湿事件有代价。
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)，第39行：Task 3: complete — 训练、两日历raw/pair评价、来源/实际点场/独立成员核对、区间、事件、分层、站点与PNG/SVG报告均完成。报告明确raw相对G265/强Affine的不同收益、K16后处理和单次MC界限。最终Git操作由交付会话执行并核对HEAD，不以后台启动代替完成。原队列与后处理进程均正常结束，没有追加实验。

## 原始协议、结果与报告

- [research/joint_fusion_20261126/delivery_verification.json](../../../research/joint_fusion_20261126/delivery_verification.json)（metadata；read）
- [research/joint_fusion_20261126/lock.json](../../../research/joint_fusion_20261126/lock.json)（protocol；read）
- [research/joint_fusion_20261126/progress.md](../../../research/joint_fusion_20261126/progress.md)（documentation；read）

## 待核查项

- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
