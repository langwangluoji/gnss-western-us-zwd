# 时空联合残差 Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2016–2019 训练、2020 验证合同下，实现带海岸距离、地形条件和因果历史状态的基础场 + 残差 diffusion，并公平比较 GRU、LSTM 与 causal Transformer。

**Architecture:** 固定支撑空间编码器负责将稀疏 GNSS 观测转换为连续基础场均值 μ；时间模块只读取历史空间摘要与折外创新，输出因果历史状态 h；diffusion 在标准化残差上生成下一时刻或多步残差，最终输出 μ+r̂。三种时间模块共享空间主干、数据采样、损失权重、随机种子和评价合同。

**Tech Stack:** Python、PyTorch、现有 gnss_field_server 工程、NumPy、Matplotlib、Mermaid/Markdown。

**Spec:** `docs/superpowers/specs/2026-11-25-spatiotemporal-residual-diffusion-temporal-ablation.md` 与 `docs/superpowers/specs/2026-11-25-spatiotemporal-architecture-diagrams.md`

## Global Constraints

- 训练时间：2016–2019；开发验证：2020；C/D/确认站在配置冻结前不可读取。
- 新轨道禁止 PWV、IWV、TCWV、Tm；只使用 ZWD、温度、气压、几何和地形/海岸特征。
- 所有时间模块只允许读取目标时刻之前的信息；未来标签不得进入历史状态。
- 残差统计量只由训练折计算，并写入 `normalization.json`。
- 原始数据、权重、大数组和凭据不提交 Git；代码、锁文件、轻量指标、图件和报告必须提交。
- 每个正式实验双种子；短训只用于筛选，不写成最终精度。

## Review Focus

- 日期边界泄漏：测试 `2020` 日期不进入训练索引、归一化或采样器。
- 时间泄漏：测试 causal mask、历史窗口和自由滚动不读取未来 token。
- 站点泄漏：目标站标签不进入条件站聚合。
- 海岸距离单位：测试米/千米转换、缺值和边界站点。
- 反标准化一致性：测试 `ZWD_hat = mu + scale * u_hat` 与残差指标复算一致。
- 查询一致性：固定支撑输入下 64/137 查询子集同点漂移 <1e-5 mm。

---

### Task 1: 锁定 2016–2019 / 2020 数据合同

**Files:**
- Create: `research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/contracts/temporal_2016_2019_train_2020_dev.json`
- Create: `.../results/20261125_spatiotemporal_contract/contract.json`
- Create: `.../results/20261125_spatiotemporal_contract/qc_receipt.json`
- Test: `.../tests/test_temporal_contract.py`

**Interfaces:** `load_contract(path) -> Contract`; `assert_no_date_overlap(train, dev, confirm) -> None`。

- [ ] **Step 1:** 写失败测试，断言训练日期最大值为 2019-12-31、开发日期最小值为 2020-01-01，且 C/D 不在训练索引。
- [ ] **Step 2:** 运行 `pytest tests/test_temporal_contract.py -q`，确认测试因合同文件不存在而失败。
- [ ] **Step 3:** 生成合同、站点角色、日期掩码、质量控制摘要和支持距离分布。
- [ ] **Step 4:** 运行同一测试并保存 `qc_receipt.json`；输出必须包含样本数、站点数、变量白名单和禁用变量扫描结果。
- [ ] **Step 5:** 提交 `data: lock temporal train-dev contract`。

### Task 2: 海岸距离与地形特征管线

**Files:**
- Create: `.../tools/terrain_diffusion/coast_distance.py`
- Modify: `.../tools/terrain_diffusion/feature_encoder.py`
- Create: `.../tests/test_coast_distance_features.py`
- Create: `.../results/20261125_coast_feature_audit/feature_summary.json`

**Interfaces:** `compute_coast_distance(lon, lat, coast_geometry) -> float`; `build_edge_features(query, station) -> Tensor`，输出固定顺序 `[dx,dy,log_dist,dH,dSlope,dRough,dCoast]`。

- [ ] **Step 1:** 写测试，验证同点距离为 0、单位为 km、边消息维度为 7、缺失地形值有 mask。
- [ ] **Step 2:** 运行测试确认失败。
- [ ] **Step 3:** 实现海岸距离栅格/站点查询和边消息构造；保存来源、坐标系、单位和哈希。
- [ ] **Step 4:** 运行测试并对 0–100、100–300、>300 km 分层统计。
- [ ] **Step 5:** 提交 `feat: add coast distance edge features`。

### Task 3: 连续历史窗口和折外残差

**Files:**
- Create: `.../data/temporal_windows.py`
- Create: `.../data/oof_residuals.py`
- Create: `.../tests/test_causal_windows.py`
- Create: `.../results/20261125_oof_residuals/manifest.json`

**Interfaces:** `make_history_window(events, lookback=12, horizons=(1,3,6)) -> Batch`; `build_oof_residuals(foundation_checkpoint, split) -> ResidualStore`。

- [ ] **Step 1:** 写测试，验证窗口只包含 `t-11...t`，目标为 `t+1/t+3/t+6`，缺测包含 mask 与 age。
- [ ] **Step 2:** 运行测试确认未来值泄漏检查失败。
- [ ] **Step 3:** 实现窗口生成和基础场折外预测；不得使用验证目标训练基础场。
- [ ] **Step 4:** 复算 `r = zwd - mu`，保存训练尺度 `b_r/s_r` 和输入哈希。
- [ ] **Step 5:** 运行窗口、泄漏和反标准化测试；提交 `feat: build causal windows and oof residuals`。

### Task 4: 共享时空基础场

**Files:**
- Create/modify: `.../models/spatiotemporal_foundation.py`
- Create: `.../tests/test_foundation_query_consistency.py`
- Create: `.../results/20261125_foundation_coast_ablation/`

**Interfaces:** `SpatialEncoder.forward(stations, grid, terrain, coast) -> (mu, q)`；`read_query(latent, query) -> Tensor`。

- [ ] **Step 1:** 写测试，断言输出形状、无目标站标签、查询子集漂移 <1e-5 mm。
- [ ] **Step 2:** 运行测试确认新模块未定义而失败。
- [ ] **Step 3:** 实现 50/150/300 km 聚合、固定支撑网格、两级 U-Net 和 terrain/coast FiLM。
- [ ] **Step 4:** 运行 C0/C1/C2 海岸距离消融，双种子短训；只保留 formal 扩展指标。
- [ ] **Step 5:** 对最优配置双种子 600 s 正式训练，保存 `summary.json`、曲线和误差分层表；提交 `feat: add coast-aware spatial foundation`。

### Task 5: GRU 与 LSTM 时间模块

**Files:**
- Create: `.../models/temporal_gru.py`
- Create: `.../models/temporal_lstm.py`
- Create: `.../tests/test_temporal_causality.py`

**Interfaces:** `TemporalGRU.forward(history, mask) -> State`; `TemporalLSTM.forward(history, mask) -> State`；State 必须含 `h`, `scale_logits`, `valid_mask`。

- [ ] **Step 1:** 写测试，置换未来输入不得改变当前 hidden state；缺测 mask 不得产生 NaN。
- [ ] **Step 2:** 运行测试确认失败。
- [ ] **Step 3:** 实现单向 2 层 hidden=128 GRU/LSTM；历史输入为 `q, innovation, T/P diff, increment`。
- [ ] **Step 4:** 在相同空间主干、采样器、种子和更新预算下运行 S1/S2。
- [ ] **Step 5:** 生成 1/3/6 小时、夏冬、海岸距离和站对分层指标；提交 `feat: add causal gru and lstm branches`。

### Task 6: Causal Transformer / GPT 式分支

**Files:**
- Create: `.../models/causal_temporal_transformer.py`
- Create: `.../tests/test_causal_transformer_mask.py`
- Create: `.../results/20261125_temporal_s3_causal_transformer/`

**Interfaces:** `CausalTemporalTransformer.forward(tokens, key_padding_mask) -> State`；`make_causal_mask(length) -> BoolTensor`。

- [ ] **Step 1:** 写测试，检查上三角 mask 全为禁用，并用未来 token 改写测试确认当前输出不变。
- [ ] **Step 2:** 运行测试确认失败。
- [ ] **Step 3:** 实现 4 层、8 heads、hidden=128、context=12 的 causal Transformer；加入随机历史 span mask，不允许未来泄漏。
- [ ] **Step 4:** 训练 teacher-forcing direct horizon 版本。
- [ ] **Step 5:** 运行自由滚动 1/3/6 小时评价，保存误差增长曲线；提交 `feat: add causal transformer branch`。

### Task 7: 标准化残差 Diffusion 与自适应噪声

**Files:**
- Create/modify: `.../models/residual_diffusion.py`
- Create: `.../tests/test_residual_scaling.py`
- Create: `.../results/20261125_diffusion_temporal_ablation/`

**Interfaces:** `ResidualDiffusion.loss(u_target, condition, sigma) -> Tensor`; `ResidualDiffusion.sample(condition, steps, members) -> Tensor`；`denormalize(u, stats) -> residual`。

- [ ] **Step 1:** 写测试，验证标准化—反标准化闭环误差 <1e-6，并验证 `sigma=1` warm-up 时行为稳定。
- [ ] **Step 2:** 运行测试确认失败。
- [ ] **Step 3:** 实现固定尺度 residual diffusion，先预测 v/epsilon 之一并锁定噪声日程。
- [ ] **Step 4:** 加入 `sigma_t=softplus(head(h, season, terrain, coast))+sigma_min`，前 20% 更新 stop-gradient warm-up。
- [ ] **Step 5:** 对 S1/S2/S3 分支分别运行固定尺度和自适应尺度，对比 CRPS、覆盖率、区间宽度；提交 `feat: add conditional residual diffusion`。

### Task 8: 正式评价、图件与 Git 归档

**Files:**
- Create: `.../tools/terrain_diffusion/evaluate_spatiotemporal.py`
- Create: `.../tools/terrain_diffusion/plot_spatiotemporal.py`
- Create: `.../results/20261125_spatiotemporal_final/REPORT.md`
- Modify: `docs/CURRENT_RESEARCH.md`, `docs/EXPERIMENT_INDEX.md`

**Interfaces:** `evaluate_split(predictions, labels, strata) -> metrics.json`; `render_report(metrics, curves, maps) -> REPORT.md`。

- [ ] **Step 1:** 写评价测试，验证宏/微 RMSE、CRPS、覆盖率、站对距离和小时增量分层均可复算。
- [ ] **Step 2:** 运行评价测试并保存零误差复算记录。
- [ ] **Step 3:** 对冻结模型读取 2020 全部 formal 数据，生成总体、季节、海岸距离、地形、高程差和站对表格。
- [ ] **Step 4:** 生成训练曲线、空间误差图、CRPS 热图、季节图、海岸距离曲线和 rollout error growth 图。
- [ ] **Step 5:** 写采用/拒绝结论：时间模块必须同时看点值、概率质量和自由滚动稳定性，不以单一 monitor 分数选模。
- [ ] **Step 6:** 运行 `git diff --check`、完整测试、指标复算和远端一致性检查；提交 `docs: archive spatiotemporal final evaluation` 并推送。

## Stop Conditions

- 任一分支出现未来信息泄漏，立即停止比较并修复数据管线。
- 训练指标改善但 formal 2020 指标不改善，不晋级架构。
- 只改善总体 RMSE、却恶化夏季、远距离或大高差组，不作为候选冠军。
- 自适应噪声导致区间覆盖率异常或尺度发散，回退到固定尺度版本。
- 任何结果没有协议、输入哈希、指标 JSON、图件和复算命令，不称为完成。

## Commit Cadence

每完成一个 Task，先运行该 Task 的测试和 `git diff --check`，再单独提交；禁止把多个未验证实验混成一个提交。每次推送后执行：

```powershell
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

两者完全一致才报告 GitHub 已同步。
