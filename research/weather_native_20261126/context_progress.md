# SDD ledger — plan: docs/superpowers/plans/2026-11-26-uncertainty-context-bypass.md

- Pre-flight: Task1的视图/工厂决定Task2所有训练/采样入口及预条件器维度；以同一condition-view合同为准，latent64默认兼容旧权重。
- Pre-flight: Task2协议字段决定Task3审计和容量比较；旧协议缺字段时显式按latent/64解释，新增协议必须登记实际视图和宽度。
- 沿用用户授权的当前分支、原缓存与结果路径，独立新增视图文件；原模型类文件保持冻结。
- Task 1: complete。视图/工厂缺失时3项RED；GREEN后补充非默认条件维度的空域/联合/频率token采样，研究测试118通过。S175/Lwide参数12174/12180；默认五类模型逐参数/缓冲量与旧工厂完全相同。
- Task 2: complete。训练/评价/copula/校准入口按协议构造条件视图和宽度，旧协议默认latent/64；预条件视图不一致拒绝。真实缺测3块上skip Gaussian及skip预条件频率token各4更新通过，标签未改，仅工程小试。
- Task 3: pending。本次独立输出20261126_probability_context，seed42先执行R111/S175/Lwide各3000连续块；L64复用已冻结Gaussian，不计新重复。
