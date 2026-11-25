# 时空联合基础场与残差 Diffusion 架构图

> 执行前必读：[下一阶段入口与审阅修订](../../NEXT_STAGE.md)。本文件保留早期设计；时间边界、自适应尺度和执行路径以该入口修订为准。

版本：2026-11-25  
对应方案：[时间模块公平比较方案](2026-11-25-spatiotemporal-residual-diffusion-temporal-ablation.md)

## 1. 总体架构

```mermaid
flowchart LR
    A["稀疏 GNSS 条件站<br/>ZWD / T / P / 质量"] --> B["站点特征编码"]
    G["几何与地形<br/>x y H 坡度 粗糙度"] --> B
    C["海岸距离<br/>d_coast"] --> B
    B --> E["站点边消息 MLP<br/>Δx Δy log d ΔH Δterrain Δd_coast"]
    E --> M1["50 km 聚合"]
    E --> M2["150 km 聚合"]
    E --> M3["300 km 聚合"]
    M1 --> S["固定支撑空间潜场"]
    M2 --> S
    M3 --> S
    S --> U["轻量 U-Net<br/>32→64→128→64→32"]
    G --> F["Terrain FiLM"]
    C --> F
    F --> U
    U --> Q["查询站局部读出"]
    Q --> MU["基础场均值 μ_t"]
    Q --> QS["空间条件 q_t"]

    H["过去 12 小时<br/>GNSS / 创新 / T P变化率 / 小时增量"] --> T["时间编码器<br/>GRU 或 LSTM 或 Causal Transformer"]
    T --> HT["历史状态 h_t"]
    QS --> T
    MU --> R["折外残差 r_t = ZWD_t - μ_t"]
    R --> N["残差标准化<br/>u_t=(r_t-b)/s"]
    N --> D["条件 Residual Diffusion"]
    HT --> D
    F --> D
    C --> D
    D --> RU["生成标准化残差 û_t+1"]
    RU --> INV["反标准化<br/>r̂=s·û+b"]
    MU --> OUT["最终 ZWD 预测"]
    INV --> OUT
```

## 2. 共享空间编码器

```mermaid
flowchart TB
    O["条件站集合 O_t"] --> P["站点投影 MLP"]
    X["查询点 / 支撑点坐标"] --> EDGE["构造边消息"]
    O --> EDGE
    EDGE --> E["Δx Δy log距离 ΔH Δ坡度 Δ粗糙度 Δ海岸距离"]
    P --> CAT["拼接观测特征与边消息"]
    E --> CAT
    CAT --> A50["局部注意力 / scatter<br/>半径 50 km"]
    CAT --> A150["局部注意力 / scatter<br/>半径 150 km"]
    CAT --> A300["局部注意力 / scatter<br/>半径 300 km"]
    A50 --> GRID["固定支撑网格 C=32"]
    A150 --> GRID
    A300 --> GRID
    TERR["高程 / 坡度 / 粗糙度 / 海岸距离栅格"] --> FILM["FiLM 调制"]
    GRID --> UNET["两级 U-Net<br/>下采样×2 + skip connection"]
    FILM --> UNET
    UNET --> LATENT["空间潜场"]
    LATENT --> READ["查询点双线性 + 局部核读出"]
    READ --> MU["μ_t"]
    READ --> Q["q_t"]
```

这里的关键是：观测站不直接拼到固定长度向量，而是通过**站点—查询点边消息**进入空间场。边消息包含海岸距离差，因此模型可以区分海岸带、海岸—内陆过渡带和内陆站点。

## 3. GRU 时间分支

```mermaid
flowchart LR
    Q0["q_t-11"] --> G["GRU × 2<br/>hidden=128"]
    Q1["q_t-10"] --> G
    Q2["..."] --> G
    Q3["q_t"] --> G
    I["历史创新 r_t-11:t<br/>T/P及变化率<br/>小时增量"] --> IN["历史特征拼接"]
    IN --> G
    G --> H["h_t^GRU"]
    H --> C1["Diffusion cross-attention"]
    H --> C2["自适应尺度 head σ_t"]
    C1 --> D["残差 Diffusion"]
    C2 --> D
    D --> R["生成 r̂_t+1"]
```

GRU 只使用单向历史状态，适合先验证短期天气持续性。训练和推理均保持因果顺序。

## 4. LSTM 时间分支

```mermaid
flowchart LR
    QL["过去 12 小时空间摘要 q"] --> L["LSTM × 2<br/>hidden=128"]
    IL["过去 12 小时创新、T/P变化率、小时增量"] --> L
    L --> HC["h_t^LSTM"]
    L --> CC["c_t^LSTM"]
    HC --> CA["Diffusion 条件"]
    CC --> MEM["长期记忆门控"]
    MEM --> CA
    HC --> SCALE["季节/状态尺度 head"]
    SCALE --> D["自适应残差 Diffusion"]
    CA --> D
    D --> OUT["生成未来残差"]
```

LSTM 的 cell state 用于检验持续性天气过程是否需要比 GRU 更长的记忆。

## 5. Causal Transformer / GPT 式时间分支

```mermaid
flowchart TB
    T1["token t-11<br/>空间摘要+创新"] --> POS["时间位置编码"]
    T2["token t-10<br/>空间摘要+创新"] --> POS
    T3["token ..."] --> POS
    T4["token t<br/>空间摘要+创新"] --> POS
    POS --> MASK["因果 Attention Mask<br/>只能访问过去和当前"]
    MASK --> TR["Transformer Encoder ×4<br/>8 heads, hidden=128"]
    TR --> HT["h_t^causal"]
    HT --> NEXT["下一时刻残差 head"]
    HT --> SCALE["条件噪声尺度 head"]
    HT --> D["Diffusion 条件"]
    NEXT --> D
    SCALE --> D
    D --> U1["û_t+1"]
    U1 --> APPEND["加入历史状态<br/>自由滚动时使用预测值"]
    APPEND --> T4
```

训练阶段使用 teacher forcing；自由滚动阶段生成 (t+1) 后将预测状态加入历史，再预测 (t+2)。未来 token 永远不可见。

## 6. 残差 Diffusion 机制

```mermaid
flowchart LR
    Y["真实 ZWD y_t+1"] --> SUB["减去基础场 μ_t+1"]
    MU["基础场 μ_t+1"] --> SUB
    SUB --> RES["残差 r_t+1"]
    RES --> STD["训练集统计量标准化<br/>u=(r-b)/s"]
    STD --> SCALE["条件标准化 v = (u - m) / a"]
    SCALE --> FWD["固定日程加噪<br/>z = sqrt(alpha) * v + sqrt(1-alpha) * noise"]
    H["h_t + 地形 + 海岸距离"] --> SIGMA["冻结的位置 m 和正尺度 a"]
    SIGMA --> SCALE
    FWD --> NET["条件 Diffusion U-Net/DiT<br/>预测 v 或 ε"]
    NET --> REV["反向采样 DDIM 20/50步<br/>K=32/64样本"]
    REV --> U["生成 v_hat"]
    U --> INV["恢复残差 r_hat = b + s * (m + a * v_hat)"]
    SIGMA --> INV
    INV --> ADD["μ_t+1 + r̂_t+1"]
    MU --> ADD
    ADD --> PRED["最终 ZWD 样本与区间"]
```

采用条件残差标准化：位置与尺度头先拟合并冻结，扩散保持标准高斯终点和固定噪声日程。固定尺度对照取 m=0、a=1。详见统一入口的数学修订。

## 7. 三种时间模块的比较重点

```mermaid
flowchart TB
    SHARED["相同空间主干、数据合同、残差标准化、随机种子"] --> GRU["GRU<br/>短期持续性"]
    SHARED --> LSTM["LSTM<br/>cell state 长记忆"]
    SHARED --> GPT["Causal Transformer<br/>GPT式因果历史"]
    GRU --> M["1h / 3h / 6h direct horizon"]
    LSTM --> M
    GPT --> M
    M --> R1["总体 RMSE / CRPS"]
    M --> R2["夏季 vs 冬季"]
    M --> R3["海岸距离分层"]
    M --> R4["站对距离 / 高程差 / 地形分层"]
    M --> R5["teacher forcing vs 自由滚动"]
    M --> R6["90%覆盖率与区间宽度"]
```

## 8. 训练与推理边界

```mermaid
graph LR
    H["历史窗口<br/>t-11 到 t"] --> S["空间编码器"]
    S --> T["时间模块"]
    T --> D["残差 Diffusion"]
    S --> MU["基础场均值 mu_t1"]
    D --> R["生成残差 r_hat_t1"]
    MU --> OUT["最终 ZWD_hat_t1"]
    R --> OUT
    OUT --> NEXT["自由滚动时更新下一历史窗口"]
    NEXT --> H
```

未来预报图中的 mu_t1 必须由截至 t 的历史预测得到，不读取 t+1 的真实观测。GitHub 支持时序图；此前版本未验证时序图兼容性就是失败原因。本次实际修正了流程图标签中未加引号的括号表达式。

## 9. 需要生成的结果图

- `architecture_overview.png`：总体架构；
- `space_encoder_coast_distance.png`：海岸距离与地形边消息；
- `temporal_module_comparison.png`：GRU/LSTM/causal Transformer；
- `residual_diffusion_flow.png`：标准化残差 diffusion；
- `seasonal_rmse_crps.png`：夏季/冬季分层；
- `coast_distance_profile.png`：海岸距离分层；
- `rollout_error_growth.png`：1/3/6 小时自由滚动误差增长。

GitHub 页面可以直接渲染以上 Mermaid 图；正式运行后再从同一结构导出 PNG/SVG，避免图示与代码架构漂移。
