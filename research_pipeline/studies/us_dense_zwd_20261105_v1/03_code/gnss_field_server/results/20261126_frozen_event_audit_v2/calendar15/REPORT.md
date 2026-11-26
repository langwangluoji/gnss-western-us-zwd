# 六小时快速增湿/减湿概率诊断

阈值10/20/30mm及主阈值20mm在读取事件分数前固定。目标为同一预报起点的ZWD(+6h)−ZWD(current)，正负两侧全部保留。2020为已暴露开发诊断。

| 比较 | 阈值/mm | 方向 | 指标 | 改善 | 差值95%区间 |
|---|---:|---|---|---:|---|
| full_vs_reference | 10 | increase | fair_brier | -0.91% | [-0.000981674128100342, 0.0038314456806488035] |
| full_vs_reference | 10 | increase | tail_weighted_fair_crps_mm | -0.84% | [-0.02901065286244596, 0.06677725504169302] |
| full_vs_reference | 10 | decrease | fair_brier | -1.42% | [-0.003118165752279196, 0.007515386453389983] |
| full_vs_reference | 10 | decrease | tail_weighted_fair_crps_mm | -0.76% | [-0.08674802870393915, 0.11464677945850807] |
| full_vs_reference | 20 | increase | fair_brier | -0.64% | [-0.0015344324435588174, 0.0026048397873907085] |
| full_vs_reference | 20 | increase | tail_weighted_fair_crps_mm | -1.03% | [-0.016052807438277285, 0.040772081952269675] |
| full_vs_reference | 20 | decrease | fair_brier | -0.53% | [-0.003424450447436909, 0.003947413452183763] |
| full_vs_reference | 20 | decrease | tail_weighted_fair_crps_mm | -0.59% | [-0.05118742403596396, 0.057846464527076105] |
| full_vs_reference | 30 | increase | fair_brier | -1.18% | [-0.0006735971759251287, 0.0016566227120170755] |
| full_vs_reference | 30 | increase | tail_weighted_fair_crps_mm | -1.35% | [-0.006789067331439256, 0.02210412450214786] |
| full_vs_reference | 30 | decrease | fair_brier | -0.28% | [-0.001980518267011534, 0.0018991627982185581] |
| full_vs_reference | 30 | decrease | tail_weighted_fair_crps_mm | -0.89% | [-0.024605303841659695, 0.031855170946391106] |
| local_vs_reference | 10 | increase | fair_brier | -0.00% | [-0.0005884032087220952, 0.0005779121584495824] |
| local_vs_reference | 10 | increase | tail_weighted_fair_crps_mm | -0.04% | [-0.006460659062142655, 0.00840381621418932] |
| local_vs_reference | 10 | decrease | fair_brier | 0.41% | [-0.0011015809881779466, -0.00021361772129297566] |
| local_vs_reference | 10 | decrease | tail_weighted_fair_crps_mm | 0.32% | [-0.013702103471983852, 0.00022690406850080577] |
| local_vs_reference | 20 | increase | fair_brier | 0.01% | [-0.0002819339664370284, 0.00028374550500078887] |
| local_vs_reference | 20 | increase | tail_weighted_fair_crps_mm | -0.05% | [-0.0032276591206456275, 0.004607355591404958] |
| local_vs_reference | 20 | decrease | fair_brier | 0.33% | [-0.0005026424983764601, -3.0947229067764866e-06] |
| local_vs_reference | 20 | decrease | tail_weighted_fair_crps_mm | 0.21% | [-0.00587150041242443, 0.001515409789762702] |
| local_vs_reference | 30 | increase | fair_brier | -0.01% | [-0.00013930806115140012, 0.000159943851163976] |
| local_vs_reference | 30 | increase | tail_weighted_fair_crps_mm | -0.13% | [-0.0014413520526336, 0.0030560475486697534] |
| local_vs_reference | 30 | decrease | fair_brier | 0.25% | [-0.0002358718323190362, 4.8824840254494515e-05] |
| local_vs_reference | 30 | decrease | tail_weighted_fair_crps_mm | 0.09% | [-0.0023284017077331983, 0.001313267571803613] |

fair Brier扣除有限IID成员概率估计方差，尾部加权fair CRPS通过对增量分布做上/下阈值截尾计算。它们均在全部有效起点上评分，不只在发生事件时评分；这两个分数理论上非负，模型间配对差值可负。共同月份2000次重采样、两个种子共享月份；未做多重比较校正。

本报告检查已冻结模型，不训练、调门控或更改权重。主日历对照含GraphAffine/Affine；补充日期仅检验冻结Full/Local相对原GP。事件计数、逐月和来源见[event_scores.json](event_scores.json)。
