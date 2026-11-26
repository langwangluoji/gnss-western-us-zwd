# 2026-11-26最终收束记录

用户要求完成当前实验后停止新增实验，总结全部成果、最佳架构和精度。最后一轮维持原冻结v2合同，没有启动新均值、相关物理状态、额外频域或新日期研究。

## 完成证据

- root启动的seed42/17原队列正常退出；G_weather、G_extra、T_weather各3000块，监督2999/3000。六种输出×两种子共十二份评价与joint评分完成。
- 原锁和模型/汇总代码未改写。weather_calibration_summary.py实际运行退出0，检查全部Adam/scheduler、源哈希、点场/真值、匹配尺度与标准差、3个查询的K32全部Gamma成员及原始CRPS。
- 现有review_inducing_flow审阅者只读终审：六训练封存与十二评价一致；独立SciPy密度、共同点场、协方差、锁定成员、2000次共同月份区间复算通过，未发现阻断问题。
- final_point_metrics.py只读重算既有两日历，不新增拟合/日期；I1合并当前10.319804，joint当前10.327014、F6 15.053927mm。各72h段内单独计算增量，不跨空档。
- final_research_figures.py生成三套PNG/SVG：完整架构、性能总览、最终训练曲线，均已实际查看，文字和数值可读；manifest绑定源JSON、绘图代码和图哈希。
- 最终数学/回归核验涵盖weather_calibration、student_process、directed_lag_v2及weather_calibration_lock：实际pytest退出0，13 passed in 5.98s。最终报告表格、49个交付链接、JSON及图源哈希也已通过专用核验。
- 总报告再次经只读终审，24个点指标、末轮全表及配对区间一致；按审阅澄清早期S175概率中心与后期固定均值的区别、原Gaussian/Gamma独立种子，以及同种子内点场一致范围。

## 结论与归档

当前重建保留I1；未来六小时采用joint v2。末轮同协方差天气t相对天气G减少F6 CRPS0.011290mm，但相对常数t额外收益区间跨零；原Affine time仍为本轮最低F6 CRPS。概率来源仍为原TF3000，不与新joint均值成绩拼接。

完整内容见docs/research_notes/2026-11-26-final-research-summary.md。README、CURRENT_RESEARCH与EXPERIMENT_INDEX指向最终报告，旧阶段状态作为历史保留。轻量清单由final_closeout_verify.py生成，只stage代码、协议/指标JSON、报告和PNG/SVG；原数据、权重、NPZ及旧无关文件留本地。Git提交推送后另实时核对本地HEAD与远端，不将仅本地内容称为已同步。

研究任务已停止新增实验；归档、核验和同步不创建新实验队列。
