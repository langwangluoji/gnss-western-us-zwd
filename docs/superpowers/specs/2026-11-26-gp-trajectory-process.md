# GP参考与时空残差轨迹扩散

## 目标和起点

现有单点Gaussian预条件仍有边际尺度与联合结构的优化空间；seed42 DDIM40当前/F6 fair CRPS为5.570/7.998mm，内部增量3.171mm。相关Gaussian控制的增量为3.190mm，主日历没有形成5%概率收益。S175显式信息直连Gaussian当前解析CRPS5.507mm，参数匹配Lwide为5.587mm；S175–Lwide当前K32 fair CRPS差95%月份配对区间为[−0.1052,−0.0567]mm，F6区间跨零。

本轮以冻结S175 Gaussian及其训练copula为强参考，比较完整空间/时间白化与查询集合分布一致的局部轨迹过程。均值源仍为冻结p14+GRU+I1与因果Transformer；输入、原数据和旧模型不改。

## 数据与单位

2016–2019各3000连续12h块，82可见、128隐藏、72h逐站状态；只抽站身份，时间不抽样。训练/开发缓存和均值embedding复用。2020主日历3布局×12月×72h为已暴露开发。确认138及原C/D隔离。输入仅GNSS ZWD/T/P、mask/age、身份、地形/海岸及历史模型误差；禁止PWV/IWV/TCWV/Tm。

先令`r=(y-mu_frozen-b)/s`，b、s为训练残差逐lead统计。冻结Gaussian输出`g(c)`、`d(c)>0`，定义`u=(r-g)/d`。c固定为S175：原64维状态与111维因果查询向量直连。

时间相关矩阵Kt和空间核Ks均来自训练copula。空间核为`rho*k(q,j)+(1-rho)*I(id_q=id_j)`，k使用已冻结300km/1500m几何核。仅接受唯一查询站ID；重复坐标不同ID允许。该随机过程对应含nugget的观测残差，不宣称光滑物理场。

## 主值根与恢复

取对称主值根As=Ks^(1/2)、At=Kt^(1/2)。float64分解，float32网络运算；不默认加jitter或截断特征值。现有rho上限与时间5%收缩提供正定余量。若实际矩阵不正定，停止并留存诊断，不静默更换参考。

完整白化：`z=As^(-1) u At^(-1)`。

逆变换：`y=mu_frozen+b+s*[g+d*(As z At)]`，星号表示逐元素乘法。d不能穿过As或At。

缺测先按完整7h轨迹分组，再对子集构造Ks、图及白化；不对含NaN的全Q数组左乘。监督mask不进入c/Gaussian/核参数，推断查询集合不由未来标签可用性选取。记录监督起点/轨迹数和缺测分组规模。

## 四臂及Gaussian机制控制

| 臂 | 训练目标与生成路径 | 新增预算 |
|---|---|---:|
| G-ref | 同S175 Gaussian＋对称根copula：`As eta At` | 0 |
| D-full | 完整白化z，图谱joint v，最后完整着色 | 3000块 |
| D-local | 仅时间白化`w=u At^-1`，逐query非线性7维v；初值`As eta`，最后只乘At | 3000块 |
| G-local | 与D-local相同，但`v=A(c,t)x+b(c,t)` | 3000块 |

D-full使用冻结现有NativeDiffusion joint64（c175时45639参数）；D-local两隐藏层宽135，约44692；G-local系数网络宽124，约44944。最终以代码实计参数为准，同时报告冻结Gaussian12174参数及共同基础场参数，避免只报score。

G-local固定c后是仿射ODE，固定步长离散后仍为条件Gaussian过程；它可学习逐query的均值和7维时间协方差，空间形式受参考Ks和局部映射约束。D-local若超过它，表述为局部非线性轨迹分布收益，结合尾部/覆盖/增量判断，不直接声称学习了真实场的联合非Gaussian动力学。

三臂噪声时间均抽k[B]再广播到Q，epsilon为[B,Q,7]；初始化模型后重置共同训练噪声seed，缺测分组顺序一致。各零输出初始化，源Gaussian和基础均值冻结。新过程使用相同余弦中间日程，alpha[0]=1、alpha[100]=0，精确对应clean/white-noise两端；v参数化不除以alpha，旧日程及旧权重保留。

## 查询集合与训练解释

D-local/G-local逐query运算，不含query间BatchNorm、注意力或池化；c和均值/尺度仅依赖固定可见信息与该query。空间GP初值的子集分布一致，逐query确定性映射保留这种一致性。共同superset GP初值限制后可以检验路径逐元素相同；分别取不同根后重置相同IID seed不要求路径相同。

D-local训练提供7维轨迹边际监督，空间耦合来自参考GP和局部映射；不把它称为真实联合score的反向动力学。D-full可以读取其他query条件，其与Local的差异同时包含坐标、结构与条件聚合能力，不把两者之差单独归因为projectivity。

G-ref、D-full、D-local和G-local在Heun0零速度/零系数时对同一IID eta逐元素恢复同一Gaussian样本。学习后D-full没有查询子集分布一致性保证；固定Q1/64/128/137检查边际、CRPS、覆盖及条件CDF。Local用共同GP状态另做路径一致性工程检验。

## 采样与评价

Heun0在theta=0直接取条件速度0，不调用未训练端点网络；20步实际39 NFE。DDIM40为40 NFE诊断，旧Heun20/40 NFE结果保留，称近似预算匹配。G-ref单独报告解析Gaussian均值/CRPS/80%覆盖与K32样本均值；四臂共享K32、IID seed261126及同参考根。

R/F分开，报告RMSE/MAE/Bias/尾部、fair CRPS、有限K秩区间、解析参考、站对、预报内部增量、energy及六类分层。不同起点的第k成员不是连续72h轨迹。先seed42小试及3000块，候选再seed17和冻结补充日历；达到原协议门槛后才谈明显收益。当前strict_oof=false，正式空间折外来源仍需独立重训。
