# 本轮实验索引

| 结果目录（均位于工程results） | 状态 | 用途 |
|---|---|---|
| 20261119_local_condition_probe_v01 | 完成、三折选零 | 原全域线性条件头负结果及恢复测试 |
| 20261119_local_method_research_v01 | 非嵌套探索 | 最初局部模型诊断；Pro三轮输入/回复及截图保存在此 |
| 20261119_local_method_research_v02 | 完成、当前总报告入口 | 嵌套局部模型，全年混合，图像、候选配置与验证 |
| 20261119_local_joint_probability_v01 | 完成 | 固定M0均值的局部联合高斯；full2019含恢复后全年评价 |
| 20261119_local_fused_probability_v01 | 失败、未排名 | 旧IDW缓存缺station_ids引发输入校验失败，保留日志 |
| 20261119_local_fused_probability_v02 | 完成、当前概率候选 | 混合均值下B重新拟合高斯；full2019保存逐站128成员 |
| 20261119_local_student_probability_v01 | 完成、保留对照 | B拟合重尾模型；不替换高斯 |
| 20261119_local_conditional_probability_v01 | 完成、优先概率候选 | B拟合A条件尺度，C全年与时间×组复核 |

两个full2019初次程序在月活动组数变化时退出；每月成员完整保存。`aggregation.log`、`aggregation.lock.json`、`results.json`和`crossed_robustness.json`对应固定站组轴修复后的成功评价。原`experiment.log`保留初次失败事实。最终状态以验证清单及完成的报告为准，不把旧进程退出码当作科学结果。

当前入口为本目录`RESULTS.md`、`selected_method_v02.lock.json`和`verification.json`。`selected_method.lock.json`保留常数尺度前一版。两个选择文件均为2019开发候选，不授权读取2020标签。
