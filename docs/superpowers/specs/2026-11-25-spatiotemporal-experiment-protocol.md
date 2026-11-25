# 时空联合基础场与残差 Diffusion：完整实验协议

> 执行前必读：[下一阶段入口与审阅修订](../../NEXT_STAGE.md)。本文件保留早期设计；时间边界、自适应尺度和执行路径以该入口修订为准。

日期：2026-11-25  
协议状态：下一阶段正式实验入口  
相关架构图：[时空联合架构图](2026-11-25-spatiotemporal-architecture-diagrams.md)  
相关实施计划：[逐步实施计划](../plans/2026-11-25-spatiotemporal-diffusion-implementation-plan.md)

## 1. 研究主张与实验职责

本阶段围绕一个明确主张组织：**稀疏 GNSS 观测经过海岸—地形感知的固定支撑编码后，可以形成稳定的连续基础场；在此基础上，因果历史状态条件的残差 diffusion 能够学习下一时刻的空间残差分布，改善小时变化、远距离站对关联和概率覆盖。**

每个实验只承担一个职责：

| 实验类别 | 需要回答的问题 |
|---|---|
| 数据合同 | 2016–2019 训练、2020 验证是否严格可复现且无泄漏？ |
| 海岸特征 | 海岸距离是否解释海岸带和内陆过渡误差？ |
| 基础场 | 稀疏站点怎样编码为连续空间场？ |
| 时间模块 | GRU、LSTM、causal Transformer 哪一种最适合历史状态？ |
| 残差扩散 | diffusion 是否为基础场补充空间联合不确定性？ |
| 自适应噪声 | 夏季和快速天气过程是否需要状态依赖的噪声尺度？ |
| 自由滚动 | 历史状态是否能支持多小时连续预测？ |

不把所有实验混成一个“总分冠军”。基础场看均值精度，diffusion 看概率质量和联合结构，时间模块看条件残差与滚动稳定性。

## 2. 已核验基线

### 2.1 p14 固定支撑基础场

- 训练站：410；验证站：137；确认站及 C/D 标签保持隔离。
- 双种子、每种子 600 秒正式训练。
- 扩展宏 RMSE：10.806 mm。
- CRPS：8.286 mm。
- 小时增量宏 RMSE：5.809 mm。
- 查询 64/137 子集同点漂移：RMS 约 (5.99\times10^{-8})，最大漂移约 (2.38\times10^{-7})。

p14 的核心价值是固定支撑表示带来的查询一致性和稳定空间读出，作为所有新实验的基础场起点。

### 2.2 V15/P15 支撑采样

- hole100 正式续训平均宏 RMSE：10.804 mm。
- random 正式续训平均宏 RMSE：11.042 mm。
- monitor 分数不能替代 formal expanded evaluation。

因此下一阶段优先使用 hole100 作为空间外推主采样，random 作为控制臂。

### 2.3 扩散基线

历史实验中相关高斯宏 RMSE 约 11.880 mm，残差 diffusion 约 12.055 mm，v11 直接 diffusion 约 12.339 mm。这些结果用于确定评价方法和失败模式，不能直接称为 p14 基础场上的配对 diffusion。

## 3. 数据合同

### 3.1 时间分割

```text
训练：2016-01-01 至 2019-12-31
开发验证：2020-01-01 至 2020-12-31
确认：C/D 和确认站，全部模型与超参数冻结后读取
```

训练和验证分别生成：

- 日期掩码；
- 站点角色清单；
- 有效小时清单；
- QC 统计；
- 变量白名单；
- 归一化统计；
- 最近支撑距离和高程差分布；
- 输入文件哈希。

任何归一化参数、采样权重或站点统计不得由 2020 标签计算。

### 3.2 允许变量

输入允许：ZWD、温度、气压、站点质量标记、经纬度、相对几何、高程、坡度、地形粗糙度、海岸距离和时间周期编码。

新轨道禁止 PWV、IWV、TCWV、Tm 作为条件或教师。所有禁用变量执行自动扫描并写入 QC receipt。

### 3.3 连续历史窗口

每个样本以目标时刻 (t) 为边界：

```text
历史输入：t-11, t-10, ..., t
预测目标：t+1、t+3、t+6
```

历史变量：

- 条件站 ZWD；
- 基础场创新 (r_t=ZWD_t-\mu_t)；
- 温度和气压；
- 温度/气压 1 小时和 3 小时差分；
- 小时增量；
- 缺测 mask；
- 距上次有效观测的时间 age；
- 小时和年内日的正余弦编码。

目标站标签只在损失和评价阶段出现，不进入条件聚合。

## 4. 海岸—地形特征实验

### 4.1 特征定义

查询点和条件站分别计算海岸距离：

[
d_c(p)=\operatorname{dist}(p,\text{coastline}).
]

站点边消息为：

[
e_{ij}=[\Delta x,\Delta y,\log(d_{ij}+1),\Delta H,
\Delta slope,\Delta roughness,\Delta d_c].
]

其中距离单位统一为 km，高程单位为 m，差分方向固定为“查询点减条件站”。

### 4.2 消融实验 C0/C1/C2

| 实验 | 海岸信息 | 作用 |
|---|---|---|
| C0 | 无海岸距离 | 控制基础场 |
| C1 | 查询点 (d_c) | 检查绝对海岸带效应 |
| C2 | 查询点、条件站及 (Delta d_c) | 检查海岸梯度和站对传递 |

固定其他变量、采样、种子和训练预算。报告：总体 RMSE、海岸 0–100/100–300/>300 km 分层 RMSE、夏冬 CRPS、站对 CRPS。

### 4.3 通过条件

C1 或 C2 至少在海岸带或海岸—内陆过渡带显著改善，同时总体 formal RMSE 不恶化超过 0.1 mm；如果只改善少数样本而扩大其他区域误差，不晋级为默认特征。

## 5. 共享空间基础场

### 5.1 编码流程

```text
条件站观测
  ↓
站点 MLP
  ↓
站点—查询点边消息
  ↓
50/150/300 km 多尺度聚合
  ↓
固定支撑网格 C=32
  ↓
两级 U-Net：32→64→128→64→32
  ↓
terrain/coast FiLM 调制
  ↓
查询点局部读出
  ↓
μ_t、q_t
```

基础场头输出均值 (\mu_t)，空间条件头输出 (q_t)。后续 diffusion 不重新计算完整大尺度均值，而是在 (\mu_t) 上建模残差。

### 5.2 基础场损失

[
L_{base}=L_{Huber}(\mu,y)
+\lambda_{inc}L_{Huber}(\Delta\mu,\Delta y)
+\lambda_{remote}L_{remote}
+\lambda_{terrain}L_{terrain}.
]

其中 (L_{remote}) 对最近支撑距离 >100 km 的目标加权，(L_{terrain}) 对高程差 >1500 m 或地形边界样本加权。权重必须写入实验锁，禁止训练后调权重解释结果。

### 5.3 验证

- 固定支撑输入下，改变查询集合不影响原有点输出；
- 站点留出和目标站标签隔离；
- 输出不存在 NaN/Inf；
- 保存总体、距离、高程、地形和海岸分层表；
- 双种子正式评估使用同一 2020 合同。

## 6. 折外残差与标准化

基础场完成训练后，针对训练样本生成折外预测：

[
r_{i,t}=y_{i,t}-\mu_{i,t}^{OOF}.
]

只使用训练折估计：

[
b_r=\operatorname{median}(r),
\qquad s_r=1.4826\operatorname{MAD}(r)+\epsilon.
]

标准化残差：

[
u_{i,t}=\frac{r_{i,t}-b_r}{s_r}.
]

Diffusion 训练在 (u) 上进行，采样后恢复：

[
\hat r=s_r\hat u+b_r,
\qquad \hat y=\mu+\hat r.
]

必须验证标准化闭环误差、残差均值、残差标准差和反标准化后的物理单位。不得用验证集重新估计 (b_r,s_r)。

## 7. 三种历史状态模型

### 7.1 GRU：短期连续状态

输入为过去 12 小时的 (q_t)、创新、T/P 差分和小时增量。两层单向 GRU，hidden=128：

[
h_t^{GRU}=GRU(x_{t-11:t}).
]

GRU 输出进入 diffusion cross-attention 和条件尺度 head。它承担 1–3 小时短期天气持续性实验。

### 7.2 LSTM：显式长记忆

两层单向 LSTM，hidden=128，同时输出 (h_t,c_t)：

[
(h_t,c_t)=LSTM(x_{t-11:t}).
]

cell state 进入尺度和残差条件头，重点观察 6 小时预测和持续性天气过程。

### 7.3 Causal Transformer：GPT 式历史预测

每个小时的空间摘要作为一个 token：

[
X=[x_{t-11},...,x_t].
]

采用上三角 causal mask，位置 (t) 只能访问不晚于 (t) 的 token。训练使用 teacher forcing 预测 (t+1) 残差，推理时采用自由滚动。

Transformer 设置：4 层、8 heads、hidden=128、context=12。随机 span mask 只模拟历史缺测，不遮盖未来后再把未来信息泄漏回当前状态。

## 8. Residual Diffusion

### 8.1 固定尺度版本

以标准化残差 (u_{t+1}) 为目标：

[
z_s=\sqrt{\bar\alpha_s}u_{t+1}
+\sqrt{1-\bar\alpha_s}\epsilon.
]

条件为：

```text
基础场空间条件 q_t+1
历史状态 h_t
地形和海岸特征
预测 horizon embedding
```

网络预测 v 或 ε，二者只选一种并在整个比较中锁定。采样使用 DDIM 20/50 步，生成 K=32/64 个成员。

### 8.2 时间自适应噪声版本

时间状态预测条件尺度：

[
\sigma_t=softplus(g(h_t,season,terrain,d_c))+\sigma_{min}.
]

前 20% 更新固定 (\sigma_t=1) 并 stop-gradient，之后逐步开放尺度 head。该设计用于表示夏季快速变化和冬季稳定状态的不同残差方差。

### 8.3 Gaussian 对照

固定尺度 diffusion 前必须运行：

1. 独立高斯残差；
2. 相关高斯残差；
3. 固定尺度 residual diffusion；
4. 自适应尺度 residual diffusion。

对照共享同一折外残差和标准化统计。

## 9. 实验矩阵

| 编号 | 空间条件 | 时间模块 | diffusion | 研究职责 |
|---|---|---|---|---|
| S0 | p14/P15 | 无历史 | Gaussian | 基线 |
| S1 | C2 基础场 | GRU | 固定尺度 | 短期状态 |
| S2 | C2 基础场 | LSTM | 固定尺度 | 长记忆 |
| S3 | C2 基础场 | causal Transformer | 固定尺度 | GPT 式因果状态 |
| S4 | C2 基础场 | 最优 GRU/LSTM | 自适应尺度 | 状态噪声 |
| S5 | C2 基础场 | causal Transformer | 自适应尺度 | 最终时空候选 |
| S6 | 最优基础场 | 最优时间模块 | direct horizon | 区分滚动误差 |
| S7 | 最优基础场 | 最优时间模块 | causal rollout | 检验连续预测 |

每个实验至少双种子。短训只用于淘汰明显不合格配置；S4–S7 使用完整 2020 formal 评价。

## 10. 训练步骤与检查点

### Phase 0：数据合同

1. 生成训练/验证日期清单；
2. 生成站点角色和目标站掩码；
3. 生成变量白名单和禁用变量扫描；
4. 生成归一化统计；
5. 生成最近支撑距离、海岸距离、高程差和季节分布；
6. 保存 contract、QC receipt 和输入哈希。

通过条件：日期无重叠、禁用变量扫描为零、所有站点角色可复算。

### Phase 1：海岸特征

1. 生成海岸线距离源和坐标系说明；
2. 计算站点和查询点海岸距离；
3. 构造 C0/C1/C2 三组输入；
4. 运行特征单位、缺值、边消息维度测试；
5. 进行双种子短筛；
6. 对候选配置执行 formal 扩展评价。

### Phase 2：基础场

1. 复现 p14/P15 输入和指标；
2. 接入 C2 海岸边消息；
3. 加入分层空间采样；
4. 训练基础场均值和小时增量头；
5. 运行查询一致性、站点留出和正式评价；
6. 冻结最佳基础场权重和配置。

### Phase 3：折外残差

1. 使用冻结基础场生成训练折外预测；
2. 计算残差并保存样本索引；
3. 只用训练折估计 (b_r,s_r)；
4. 生成历史窗口和缺测 mask；
5. 复算标准化闭环；
6. 冻结残差数据版本。

### Phase 4：时间模块比较

1. 在相同残差数据上训练 GRU；
2. 保存 1/3/6 小时 direct horizon 指标；
3. 在相同预算下训练 LSTM；
4. 保存同样指标；
5. 训练 causal Transformer；
6. 做 teacher forcing 与自由滚动对照；
7. 按 CRPS、夏季、远距离和滚动稳定性选择时间候选。

### Phase 5：Diffusion 比较

1. 训练独立高斯和相关高斯对照；
2. 训练固定尺度 residual diffusion；
3. 训练自适应尺度 residual diffusion；
4. 生成 K=32/64 成员；
5. 计算点值、CRPS、站对 CRPS、小时增量 CRPS 和覆盖率；
6. 检查反标准化、成员方差和极值。

### Phase 6：正式分析

1. 读取 2020 全部 formal 数据；
2. 输出季节、海岸距离、地形、高程差和站对距离分层表；
3. 输出训练曲线和 formal 曲线；
4. 输出基础场误差图和 diffusion 中位误差图；
5. 输出站对 CRPS 热图；
6. 输出覆盖率—区间宽度可靠性图；
7. 输出 direct horizon 与 rollout error growth；
8. 形成报告和可复算 JSON。

## 11. 性能评价

### 点值

- station-macro RMSE；
- micro RMSE；
- MAE；
- Bias；
- P90/P95 绝对误差。

### 概率

- CRPS 与 fair-CRPS；
- 90% coverage；
- 区间宽度；
- reliability curve；
- ensemble spread 与真实残差标准差比值。

### 空间

- 0–50、50–150、>150 km 站对 CRPS；
- 最近支撑距离分层；
- 高程差和地形粗糙度分层；
- 海岸距离分层。

### 时间

- 1、3、6 小时 RMSE/CRPS；
- 小时增量 RMSE/CRPS；
- 夏季、冬季和过渡季；
- teacher forcing 与自由滚动误差增长。

## 12. 图件规范

每个正式实验生成：

1. `training_curves.png`：训练损失、monitor 和 formal RMSE；
2. `foundation_error_map.png`：基础场误差空间图；
3. `diffusion_median_error_map.png`：diffusion 中位预测误差；
4. `seasonal_rmse_crps.png`：季节分层；
5. `coast_distance_profile.png`：海岸距离分层；
6. `support_distance_profile.png`：支撑距离分层；
7. `station_pair_crps.png`：站对距离 CRPS；
8. `terrain_elevation_strata.png`：高程与地形分层；
9. `coverage_reliability.png`：覆盖率和区间宽度；
10. `rollout_error_growth.png`：1/3/6 小时自由滚动。

图件必须附 `figure_manifest.json`，记录数据合同、模型版本、指标来源和生成命令。

## 13. 选择与停止规则

- 未来信息泄漏：立即停止当前分支，修复数据管线后重新训练。
- monitor 改善而 formal 2020 不改善：不晋级。
- 总体 RMSE 改善但夏季、远距离或大高差组退化：不作为默认候选。
- 只改善点值而概率覆盖恶化：保留为确定性基线，不作为正式 diffusion。
- 自适应尺度发散或区间覆盖异常：回退固定尺度版本。
- GRU/LSTM/Transformer 的比较必须使用相同空间主干、样本、种子和更新预算。
- 没有输入哈希、实验锁、指标 JSON、图件和复算命令的实验不称为完成。

## 14. 归档规范

每个实验目录必须包含：

```text
protocol.json
config.json
input_manifest.json
normalization.json
status.json
summary.json
metrics.json
figure_manifest.json
REPORT.md
```

Git 同步内容：协议、代码、轻量指标、图件、报告、哈希清单。  
本地保留内容：原始数据、模型权重、大数组、缓存和凭据。

每个阶段独立提交，提交前执行：

```powershell
git diff --check
pytest <relevant tests> -q
python <recompute script>
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

只有本地 HEAD 和远端 main 完全一致，才报告该阶段已同步。

## 15. 最终研究叙事

研究主线聚焦三个可验证贡献：

1. 用固定支撑和海岸—地形边消息，把稀疏 GNSS 观测转换为查询一致的连续基础场；
2. 用 GRU、LSTM 和 causal Transformer 比较历史状态编码，明确短期持续性、长记忆和 GPT 式因果状态的适用边界；
3. 用基础场条件残差 diffusion 建模下一时刻的空间联合不确定性，使模型从“静态图像生成”转向“给定过去大气状态的连续场预测”。

所有结果围绕这三点组织，避免把单一 monitor 分数或单一季节指标扩大为整体结论。
