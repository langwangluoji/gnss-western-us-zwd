# Mean收敛日程试验实施计划

**Goal:** 用一次单A800有限试验，区分EMA追赶、新优化日程收益与B到C迁移。

**Architecture:** 冻结全部现有底座，只从seed42零输出初始化训练mean头。沿用原生PyTorch与Slurm，新增显式登记的5000步日程及B站门槛；原1572步协议及服务器发布保持只读。

**Tech Stack:** 既有ZhouSiyu环境、PyTorch、NumPy、JSON锁与SHA256。

## 决策及依据

采用单mean试验；同时扩mean/EDM不能区分EMA滞后，直接扩GAN也缺少现有分布形状收益证据。全年归因与Pro复核记录见`results/20261120_postcoverage_analysis_v01/`。本方案是根据已授权的“继续规划和安排训练”制定的执行选择。

科学合同：B2018的6288训练小时、288开发锚点及整周隔离不变；固定seed42、batch8、ES K4、开发K16、EMA=.999、AdamW、lr1e-4、warmup100、余弦按5000步下降至1e-6。每100步同时记录EMA/raw ES，唯一选模目标为EMA的B开发ES，零候选保留。至少1600步后，连续10次正式评价未改善则早停。raw不参与候选或patience。5000步后终止，不自动续跑。

新日程改变前1572步的学习率，结果解释为“优化日程试验”，不把收益单独归因于步数。旧raw只用作固定诊断参照。

## 文件与验收

- [ ] `training/observable.py`增加显式协议分派，旧默认仍1572/300/patience3；新协议5000/1600/patience10，校验manifest一致。raw指标为joint产品的诊断字段。
- [ ] `training/observable_convergence_gate.py`只读取B和锁定权重。固定独立种子20261122、K64，比较唯一入选EMA、旧EMA、零候选；旧raw与新raw仅诊断。2000次整开发周配对bootstrap，上界<0且ES低于旧EMA才允许C评价。开发区间承认选模影响，不作为独立检验。
- [ ] 预算工具支持manifest登记的mean-only允许列表，默认旧行为保留；新增Slurm队列：真实A800基准→预算锁→mean→B门槛→条件C评价。单卡并发1。
- [ ] TDD覆盖旧协议、硬上限、门槛失败、raw不参与选模；现有恢复、ES、评分与角色隔离回归通过。
- [ ] 新发布包绑定代码、原始B/C缓存、新manifest、旧mean参考权重及登记SHA；部署新目录，不安装包。
- [ ] 基准最多5分钟、mean最多15分钟、B门槛及条件C评价最多15分钟，总GPU墙钟上限35分钟，失败无自动重试。新真实基准与预算锁未通过则停止。
- [ ] 核实Slurm ID、真实更新和有限首评后交接。若B门槛未通过，记录`SCIENTIFIC_GATE_NOT_PASSED`并跳过C，不启动EDM/GAN/搜索/额外种子。

## 记录与分流

保留训练loss、学习率、耗时/显存、B的EMA/raw ES及两者差距；C通过门槛后报告解析RMSE/MAE/偏差、ES/VS/CRPS/覆盖与宽度。B改善而C无改善时优先诊断条件和空间迁移；EMA接近旧raw但raw无进一步改善时记录EMA追赶。D2020继续封存。

Ruling：当前目录不是Git仓库，保留已封存v02发布作为恢复点，新版在独立staging、release和run下执行。Pro已完成本轮方法审阅，客户端继续验证数值和实际部署。
