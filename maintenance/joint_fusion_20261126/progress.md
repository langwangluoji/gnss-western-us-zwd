# SDD ledger — plan: docs/superpowers/plans/2026-11-26-best-base-diffusion-fusion.md

新授权为完成最佳载体与扩散的有界融合实验；旧阶段已经收束，旧结果/锁不改写。

Ruling: 采用joint v2/last作为预报载体，I1保留当前重建参考；不直接复用旧概率权重，以免残差来源错配。复用已验证神经代码，以新路径和来源锁执行。

Ruling: 普通生成K32为主；不重叠独立双场差分K16为中心保持二级对照，Gaussian/Affine/Local全部同规则，不以同批去均值伪造固定中心。

Task 1: in progress — 已读代码与数据合同；当前分支codex/p14-continuous-fusion已有隔离于main的研究开发历史。原目录存在无关untracked证据，不删除、不整体stage。

Task 1 evidence: 两种子3000块joint训练嵌入完整导出；原开发导出复用前核验。5项数学/来源测试RED→GREEN，seed42 Gaussian真实4更新6144完整轨迹通过。独立只读预审确认Sources兼容新joint与数值还原、差分概率机制。

Ruling: 首锁v1在正式训练前保留为lock_v1_before_review.json；审阅新增spec哈希、复用采样器/GP来源、实际标签时钟守卫。新增2项测试先失败后修正，再封存当前锁；旧v1未用于任何正式训练。

Task 1: complete — 当前锁SHA256=138d8fadd91dcbaf8a4a3681b5b18c8164cd8b7da6349cdec1d37a9ab92075bf；7项测试通过；双方训练导出完整，源joint开发导出身份与哈希通过。独立预审无数值/兼容性阻碍。GP和两种传输的真实四更新安排在各自冻结上游正式拟合之后，保留工程状态，不伪造3000步。

Ruling: 独立审阅提出的K16秩覆盖、原joint数组直接核对、几何/nearest复验、重点站和训练曲线作为同实验只读补充交付，不改训练合同或主评分。正式队列启动前源码/来源提交推送。

Task 2: in progress — 140bd62已推送并核对远端后启动两个原队列。S175双种子3000/3000完整监督均完成，来源/训练掩码/全部最终Adam时钟通过；copula均仅使用训练残差拟合。G265真实4更新6144完整轨迹通过，双种子正式G265在运行。RTX4060 Laptop运行正常，主要计算量为float64完整协方差分解。

Task 3 preparation — 新增只读diagnostics.py，直接核对原joint NPZ、几何/nearest与K16/K32覆盖，保留137站及五重点站；两项秩覆盖测试RED→GREEN。render_delivery.py准备完整融合图、曲线与报告；这些后处理不改已封存训练或主评分源码。
