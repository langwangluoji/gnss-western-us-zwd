# 完整周评价与配对短续训实施清单

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用固定结构、配对学习率和完整留出周，判断局部残差模型是否值得继续投入。

**Architecture:** 复用已验核的局部特征、PCA对照、格网查询、Gaussian噪声和训练引擎，另建扩展评价适配器和续训阶段。不修改上一轮源码快照或checkpoint。

**Tech Stack:** 现有PyTorch2.8、NumPy、FP64统计及本机RTX4060；不安装依赖。

**Spec:** `docs/superpowers/specs/2026-11-20-local-optimization-and-weekly-evaluation-design.md`

## Global Constraints

- 原run `results/20261120_local_context_three_arm_v01`只读。
- 新run `results/20261120_local_optimization_weekly_v01`。
- M0、空间基、输入五变量共同有效规则及四帧不变。
- 训练384小时、内层96小时、两物理组折不变。
- 两结构×两学习率×两折×500更新，最多4000次诊断更新，batch8，ES K4。
- 完整周2016小时；补1728小时M0缓存；统计沿用原训练核心。
- 本机总墙钟上限45分钟；不提交服务器作业；D2020封存。

## Review Focus

- 从COMPLETE checkpoint创建新阶段时，不跳过新阶段也不丢失优化器/RNG/游标。
- 候选checkpoint中的pending_evaluation不引发旧阶段评价重复或污染新阶段。
- LR副本的相同随机噪声序列不因评价、拆批或模型装载改变。
- 扩展时刻与原训练索引、noise缓存索引不能混用，杜绝同索引对应不同小时。
- 负值或非有限值不能通过删站、改人口或临时裁剪取得改善。

### Task 1: 扩展开发周与输入缓存

**Files:** Create `tools/local_context_weekly_data.py`; Test `tests/test_local_context_weekly.py`.

**Interfaces:** `weekly_hours(anchor_hours) -> sorted int64 hours`; `prepare_weekly_dataset(source_run, target_root) -> manifest_path`。

- [ ] 写失败测试：12个完整周合计2016小时，与384核心和96内层无交集；训练前后18小时缓冲保持；小时按实际值索引，不按旧数组位置。
- [ ] 实现按周/分片的断点缓存，复用原288小时与冻结M0；记录数据、M0、PCA、几何和原normalizer哈希。扩展标签由评分入口单独载入。
- [ ] 对共同288时刻复现原均值/条件/评分掩码，查询差≤0.001mm；扰动评价标签不改变条件、缓存和支持域；空站/单站沿用原合同。
- [ ] 生成`weekly_eval.lock.json`，明确已暴露锚点与新增时刻，记录880702个目标站时刻的实际复现数。

### Task 2: 可恢复的配对学习率阶段

**Files:** Create `tools/local_context_continue.py`; Test `tests/test_local_context_continuation.py`.

**Interfaces:** `make_continuation(source_checkpoint, arm, fold, learning_rate, directory) -> engine, objective, evaluator`。

- [ ] 先写失败测试：新阶段0步模型/raw/EMA分别等于源1000步状态；优化器、RNG和数据游标相同；COMPLETE/pending_evaluation/候选列表不沿用旧阶段语义。
- [ ] 实现独立阶段编号、继承来源哈希和总步数记录。禁止把EMA当raw载入原优化器；起点是warm start，不能标作零修正。
- [ ] 固定LR为1e-6或1e-4，其他合同一致；保留warm-start与真正零修正候选；只按内层ES选EMA。
- [ ] 验证两LR副本前100次小时/噪声序列一致；同一副本100+恢复100与连续200一致；测试首次选模中断与预算退出。
- [ ] 分别记录真实监督站时刻数，接通旧日志未使用的计数器；沿用只读追加日志监控。

### Task 3: 旧产品周评价、短续训与候选评分

**Files:** Create `tools/local_context_weekly_evaluate.py`; outputs under newrun only.

**Interfaces:** `evaluate_locked(checkpoint_lock, weekly_manifest) -> per_fold metrics/predictions`; `run_registered_queue(plan_lock) -> status`。

- [ ] 在读取新增周评分前登记唯一队列、续训合同、LR分支及暂停/恢复方式。
- [ ] 用冻结起点先复算完整周旧产品及原线性对照；不以新周结果修改预登记分支。
- [ ] 顺序执行最多8个500步分支。每100步评价原96小时；原288锚点及扩展周不参与新步数选取。
- [ ] 内层锁定后才评价选中产品；每折整场ES单列，汇总点值不伪装为单场联合概率产品。
- [ ] 非有限预测停止并保留失败状态；45分钟耗尽保存cursor，状态与科学门槛失败分开。

### Task 4: 结果判定与交接

**Files:** Create `tools/report_local_context_weekly.py`; Test `tests/test_local_context_weekly_report.py`.

- [ ] 测试：3%点估计、两折同向、零底座及CI条件缺一不得晋级；同一7/14日分区不能计作两次稳健性验证。
- [ ] 输出训练/内层/完整周三种口径的RMSE、ES、概率与结构诊断；对比高LR与下限分支的同起点变化。保留实际评分人口和有效站对数。
- [ ] 输出完整周曲线、站点收益与校准图，支持域和负值计数单列。保留全部入选权重、配置及源哈希。
- [ ] 运行项目完整测试与真实查询/恢复验核；通过门槛才制定单A800基准，不自动提交收费作业。

## 当前交付

已完成现有checkpoint梯度/raw/EMA诊断以及2016小时扩展清单审计，产物位于`results/20261120_postrun_optimization_diagnosis_v01`。本清单尚未实施新训练或新缓存。
