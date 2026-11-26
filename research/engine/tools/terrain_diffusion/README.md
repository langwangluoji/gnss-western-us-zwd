# 地形时序图扩散：代码与训练安排

本实现直接生成地表 ZWD。条件为可见站过去12小时的 ZWD、温度、气压，以及位置和现有地形特征；生成目标为整组查询点的当前 ZWD。每个成员把自身生成值放入6小时历史，逐小时形成连续序列。没有接入 v04 均值场、PWV 条件或插值教师。

## 数据与张量

新合同采用410训练站、137验证站。138确认站和原C/D标签保持隔离。原始压缩文件只将时间、站名、ZWD、T、P列转换为数值；各变量独立质控。站点队列继承历史筛选，历史PWV完整率对队列选择的影响仍需在论文数据章节交代。

训练缓存包含2016/17每月8、18、25日，72段，每段前12小时加24小时输出。验证缓存为2018每月1日。标准化只拟合训练缓存。时间使用产品名义小时，具体来源时制保存在 `contract.json`。

| 输入 | 形状 | 组成 |
|---|---|---|
| 条件站 | B × 12 × Nc × 38 | ZWD/T/P、独立mask、age，7地形/位置，16 Fourier，4时间，2空间支撑 |
| 查询点 | B × Nq × 41 | 7地形/位置，16 Fourier，4时间，14条件天气与支撑 |
| 当前扩散状态 | B × Nq × 1 | 标准化地表ZWD的加噪值 |
| 生成历史 | B × 6 × Nq × 3 | 同成员ZWD、有效标记、时间间隔 |
| 边几何 | B × Nq × Nc × 5 | 水平位移、高差、粗糙度差、水平距离 |

查询天气用50/150/300 km三种带宽聚合条件站温度与气压，并携带支持量。条件站身份没有可学习编号；目标站的ZWD不进入条件编码器。基础去噪先筛出当前有标签的目标节点，再构图，缺测值不以零标签参与目标图。

## 网络与训练目标

条件MLP → 两层几何注意力 → 每站两层因果Transformer。查询MLP接收41维特征和当前噪声标量，GRU编码目标历史，四层去噪块交替进行条件交叉注意力、目标图注意力和FFN。宽度64，4头。去噪输出为v，100步余弦噪声日程；推理采用20步确定性DDIM更新，随机性来自各小时各成员的初始噪声。

`v = sqrt(alpha_bar) × epsilon − sqrt(1−alpha_bar) × x0`。

恢复式为 `x0_hat = sqrt(alpha_bar) × x_t − sqrt(1−alpha_bar) × v_hat`。基础损失为有效节点的v均方误差。训练目标历史50%采用真实过去标签、50%缺失；评价全部冷启动，随后使用自身生成历史。

`trajectory_loss`提供4小时、K4、DDIM20的完整可微生成。主损失为标准化ZWD fair CRPS，加0.5倍标准化一小时增量fair CRPS，加0.05倍v保持损失。增量标准差由训练缓存计算；辅助v分支对生成历史停止梯度，主轨迹梯度贯穿全部小时与DDIM步骤。`profile.py`用真实数据检查整条链。

预训练通过训练池动态遮站的确定性重建实现；`--encoder`只加载条件MLP、空间层和时间层，新去噪器重新初始化。`--initial`加载完整候选用于时间微调。预训练头不会成为扩散背景。

## 当前本机筛查配置与正式方案的差别

本机筛查使用batch1、32个有效目标、随机82条件站、AdamW 3e-4、EMA0.95、FP32，每臂1000步。对照为B1确定性、B2高斯、D0扩散、D1迁移B1编码器的扩散，种子42/17。D1额外使用B1训练预算，比较时应同时列出预训练成本。验证选点使用1月/7月1日前4小时、三布局、K4；这是机制筛查日历，不代表完整2018成绩。

原v2正式P2建议B1/B2各2000步、D0/D1各3000步、D1预训练1000步；当前筛查不标记为完整P2。正式版的空间均衡/空洞抽样、64–160目标、warmup/cosine和EMA0.999尚待以同预算配对验证。当前条件图取600km内最近12站，查询图取300km内最近12点，交叉图取600km内最近16站；无支撑时保留最近点。两个全局摘要用均值与逐特征最大值。它们是本机原型对原设计局地/远程分组和可学习汇聚的简化，不能把结果归因于尚未实现的机制。

地形从现有25km格网最近节点提取坡度/粗糙度，保留测站高程和相对格网高差。海岸距离未接入，因为本轮没有核验可用海岸资料。任意查询坐标需使用现有geometry的投影坐标系，查询高程采用米；生成密度可变，真实细尺度精度需通过留站及查询集扩展实验检验。

## 可运行命令

在 `gnss_field_server` 目录运行，Python为当前pytorch环境。数据准备只执行一次，新目录保证合同不可覆盖。

```powershell
python -m tools.terrain_diffusion.data
python -m pytest tests/test_terrain_diffusion.py -q
python -m tools.terrain_diffusion.profile --out results/20261123_terrain_diffusion_v10/full_chain_profile.json
python -m tools.terrain_diffusion.screen --root results/20261123_terrain_diffusion_v10/screen --steps 1000
python -m tools.terrain_diffusion.report results/20261123_terrain_diffusion_v10/screen
```

扩大验证窗口，使用独立生成噪声，保存连续24小时与所有137站预测：

```powershell
python -m tools.terrain_diffusion.evaluate --checkpoint results/20261123_terrain_diffusion_v10/screen/d0_s42/best.pt --out results/20261123_terrain_diffusion_v10/validation24_d0_s42 --hours 24 --members 16 --rows 0 6 --seed 19007
```

任意位置联合生成：同一命令加入 `--query-csv queries.csv`。CSV列为 `x,y,height`，一次提交完整查询集合；不要把独立生成的多个小块拼成同一联合样本。这个接口复用缓存中的可见站窗口，实时数据接入需另提供相同12小时输入合同。

时序微调示例（选择机制后执行）：

```powershell
python -m tools.terrain_diffusion.train --arm diffusion --initial results/20261123_terrain_diffusion_v10/screen/d0_s42/best.pt --temporal --queries 16 --members 4 --ddim-steps 20 --steps 200 --validate-every 50 --out results/20261123_terrain_diffusion_v10/temporal_s42
```

## 评估与训练记录

主指标为每站先算RMSE、再对站等权平均，三条件布局分别计分后等权平均，另报最差布局。保存micro RMSE、MAE、Bias、P90/P95绝对误差。概率评价为站均衡fair CRPS、90%区间覆盖和宽度、负值比例；K4区间只用于早期筛查，最终需K32/64稳定性检查。一小时增量只在连续序列内部计算，两个时刻都有效才计分。

确定性模型按验证macro RMSE选点，概率模型按验证macro fair CRPS选点。概率候选需与同种子B1核对0.2mm点值容忍线；未达者作为概率/点值权衡候选，不自动替换B1。确认集只在所有模型、消融和脚本统一冻结后启用。

每次训练保存配置、代码和数据SHA256、逐更新history.csv、选中步、三布局指标、连续成员数组、最佳与最后权重。CSV含训练目标、当前随机训练批的恢复误差、验证macro RMSE、学习率、耗时和峰值显存；随机批曲线不是固定训练集成绩。`report.py`输出真实训练曲线、指标比较、散点、逐站高程误差、固定站轨迹和概率校准图。原始数据、预测NPZ和权重只保留本机。

## 后续推进门槛

1. P1：角色/遮站/因果/置换/梯度测试通过，两段小样本能够拟合，完整生成链显存可控。
2. P2：按正式预算扩展B1/B2/D0/D1，并扩展到12个月连续24小时、K16以上；比较同站人口与同布局，检查训练覆盖、地形、距离与天气过程分组。
3. P3：最多两候选做200–500步完整时序微调，检验CRPS、增量、点值和时间成员打乱试验；两种子都记录。
4. P4/P5：通过查询集合扩展一致性、密度10%/20%/40%与72小时滚动后，再进行训练池内部空间折和全量训练。新旧站人口不同，不能直接用旧v04约9mm作为本轨道胜负基准。
