# 执行记录

Plan: docs/superpowers/plans/2026-11-21-grid-processor-pair-v01.md

2026-11-21：用户授权编写文件并开始训练。正在实现，尚未启动本轮科学更新。

Ruling: 非Git工程采用新增文件、独立实验目录和锁定源码快照隔离，旧文件不覆盖。
Ruling: 新69通道使用已有训练统计及固定密度/坐标缩放，两臂共用，无开发标签统计；避免因构建归一化提前要求全量M0前向，懒缓存保留全部训练时间覆盖。
Pre-flight: 模型消费[B,69,144,96]，发布[B,1,144,96]；查询权重含exp(-H/3000)，禁止先查询特征再过非线性。

## 2026-11-21 启动核验

- 独立审阅后的六类重要问题已修复：发布依赖完整性、恢复与累计预算、缓存内容校验、冷热基准分离、真实恢复路径验证、监控状态与远端快照接入。
- 全工程回归结果：286 passed in 32.51s，见 pytest_final_verified.log。
- 最终锁定版本在本机真实数据完成400次工程更新：连续200步，与100步＋恢复100步逐位一致；模型权重真实变化。最终证据位于 scientific/engineering_resume/verification.json，耗时98.203秒。工程更新不计入科学实验结果。
- 锁定release清单SHA256：1e01c3bdefb04f1f14f2bb813764829d0a0c69bd0dd23ef0994f2a0440e89035。
- 服务器已核验同一release并提交作业3633416；单张A800，节点gpu8006，累计分配墙钟上限14400秒。既有CPU任务3632785保持运行，未作修改。
- 服务器状态已由BENCHMARK、ENGINEERING_RECOVERY_CHECK进入RUNNING，说明实测预算门槛与工程恢复门槛已放行。正式科学步数与第200步开发检查继续由实时日志验收。
- 本机实时监控：http://127.0.0.1:8781/，只监听127.0.0.1；5秒刷新。可见日志窗口：GNSS Grid69 Live Training Log。
- SSH流日志：C:/Users/user/AppData/Local/Temp/gnss_grid69_login/submit.log；每5秒同步到本实验server_live.log。关闭SSH仅中断本机快照同步，不影响Slurm。
- 本机runtime显式指向原始processed数据目录与锁定release assets，未修改已发布代码。旧原型工程缓存完成行已压缩保存在engineering_cache_rows.npz；原型巨型缓存已移除，旧研究目录不变。

启动验收完成：服务器正式MLP第200步，RMSE 7.3987451451 mm、ES 5.3744170070 mm，日志由100步增长到200步，页面已实际显示；证据startup_acceptance.json。启动检查不参与选模，队列继续后台运行。
