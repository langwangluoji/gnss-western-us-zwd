# 实施记录
2026-11-21用户授权三种生成分支设计并启动；已保存spec/plan。
Ruling: 第0步使用同一冻结相关Gaussian旁路，新的自由格网分支末层零初始化；Diffusion白化采样状态作解析Gaussian初始化，采用ES监督，不编造HR真值。
Ruling: 保留旧69通道语义；新输入另加先验场和白噪声状态为71通道。旧输入缓存只读，新增运行独立。
Ruling: pilot每臂每折一遍全训练池；完整外层K64不纳入本轮4卡时，以预登记96外层小时K32作配对验收。
服务器连接阻碍：logini.tongji.edu.cn当前解析192.168.212.240，SSH握手超时；非密码错误。等待校园网/VPN恢复，继续本机实现。
Ruling: 全尺寸GPU测试确认4成员35调用可反传，峰值355.4MiB（RTX4060，本机数据为合成输入，非A800预算）。正式入口按8小时并行、每小时K4执行，科学有效batch仍为8；评价把同一小时成员成批前向。此运行批量调整提高A800利用率，不改变小时池、成员数或更新上限，服务器短测仍需验收真实显存与耗时。

## 审阅与验证

独立审阅gen3_final_review完成：一个重要问题（部分基准恢复时计时0更新）已通过RED→GREEN测试修复，逐臂持久保存计时，并断言真实新增3次更新。无未解决Critical/Important。
Minor修复：报告加入匹配Gaussian基线与差值；补充pending_evaluation与三个选模产品的中断恢复测试；实际入口开始读取配置中的优化器、EMA、尺度、宽度和有效批量。正式工程恢复仍使用真实数据100+100对连续200，服务器未运行前不声称已验收。
已归档空间Gaussian因子只读复算：32共享成员站端最大差0.00000159mm，优于0.001mm门槛（审阅者只读验证，非新训练）。
TCP转接只用于本任务SSH字节流，保留原主机密钥验证，不处理/保存密码，不改全局SSH/VPN配置；BatchMode握手已稳定返回需要密码认证。

Ruling: Successful authenticated download verified SHA256 9b3d0102957e462b0b6d8e59caa5c71674bc43788889bc0e7957788ccf9aea21. Old Slurm COMPLETED0:0,3:10:49. Full outer B2018 RMSE: base7.577401/7.631033, MLP7.556044/7.642464, U-Net9.774505/9.615438. Do not promote U-Net mean. New three generators retain the small U-Net architecture but share frozen M0+IDW mean and Gaussian. Old selected weights remain archived for audit, not used as new baseline. This evidence-based routine correction is within authorized generation objective; no new search or extra budget. Earlier outer cohorts are now exposed development evidence, not fresh independent confirmation.
