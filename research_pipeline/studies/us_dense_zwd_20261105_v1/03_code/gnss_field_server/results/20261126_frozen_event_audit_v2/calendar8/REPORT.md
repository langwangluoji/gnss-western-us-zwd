# 六小时快速增湿/减湿概率诊断

阈值10/20/30mm及主阈值20mm在读取事件分数前固定。目标为同一预报起点的ZWD(+6h)−ZWD(current)，正负两侧全部保留。2020为已暴露开发诊断。

| 比较 | 阈值/mm | 方向 | 指标 | 改善 | 差值95%区间 |
|---|---:|---|---|---:|---|
| full_vs_reference | 10 | increase | fair_brier | -0.13% | [-0.00187417541659044, 0.002328958470255305] |
| full_vs_reference | 10 | increase | tail_weighted_fair_crps_mm | 0.52% | [-0.02970843584198084, 0.013594692754183272] |
| full_vs_reference | 10 | decrease | fair_brier | -0.36% | [-0.0010583737687540854, 0.0019689793884612775] |
| full_vs_reference | 10 | decrease | tail_weighted_fair_crps_mm | -0.09% | [-0.01215826723246476, 0.016000022948406557] |
| full_vs_reference | 20 | increase | fair_brier | 0.27% | [-0.0011697953175769168, 0.000954923664259878] |
| full_vs_reference | 20 | increase | tail_weighted_fair_crps_mm | 0.99% | [-0.013889208802004354, 0.002150413806173947] |
| full_vs_reference | 20 | decrease | fair_brier | 0.16% | [-0.000701685730237097, 0.0005927026576127285] |
| full_vs_reference | 20 | decrease | tail_weighted_fair_crps_mm | -0.26% | [-0.006888576823061515, 0.010433765401577822] |
| full_vs_reference | 30 | increase | fair_brier | 1.14% | [-0.0006217862740350123, 0.00010985248287864777] |
| full_vs_reference | 30 | increase | tail_weighted_fair_crps_mm | 1.71% | [-0.007239414571642244, -0.0007500257424796491] |
| full_vs_reference | 30 | decrease | fair_brier | -0.41% | [-0.00025727620701616156, 0.00046296099213677844] |
| full_vs_reference | 30 | decrease | tail_weighted_fair_crps_mm | -0.72% | [-0.0024504651840985357, 0.0059589269688270895] |
| local_vs_reference | 10 | increase | fair_brier | -0.10% | [-0.0003541306941201408, 0.0006135979431128329] |
| local_vs_reference | 10 | increase | tail_weighted_fair_crps_mm | 0.11% | [-0.006556862413315532, 0.0023775033313483723] |
| local_vs_reference | 10 | decrease | fair_brier | 0.42% | [-0.0008549739511330627, -0.00034719245216336325] |
| local_vs_reference | 10 | decrease | tail_weighted_fair_crps_mm | 0.47% | [-0.011786925965945194, -0.0031641543627259315] |
| local_vs_reference | 20 | increase | fair_brier | 0.17% | [-0.00030172533320218554, 7.32965166654457e-05] |
| local_vs_reference | 20 | increase | tail_weighted_fair_crps_mm | 0.32% | [-0.0037183101588541155, -0.0005515959970602101] |
| local_vs_reference | 20 | decrease | fair_brier | 0.50% | [-0.00046177371545258656, -0.00010870601979545434] |
| local_vs_reference | 20 | decrease | tail_weighted_fair_crps_mm | 0.45% | [-0.005514562849556057, -0.0005976701490110069] |
| local_vs_reference | 30 | increase | fair_brier | 0.44% | [-0.00017375832531751898, -4.2241449215613124e-05] |
| local_vs_reference | 30 | increase | tail_weighted_fair_crps_mm | 0.37% | [-0.001452915629900133, -0.0003337988035064585] |
| local_vs_reference | 30 | decrease | fair_brier | 0.46% | [-0.00020847059322756369, -1.228930435610371e-05] |
| local_vs_reference | 30 | decrease | tail_weighted_fair_crps_mm | 0.36% | [-0.002341820794884853, 1.4673392903671849e-05] |
| full_vs_graph_affine | 10 | increase | fair_brier | -0.34% | [-0.0017484340794372013, 0.003038314062168048] |
| full_vs_graph_affine | 10 | increase | tail_weighted_fair_crps_mm | 0.58% | [-0.037595823746280406, 0.018942496466556297] |
| full_vs_graph_affine | 10 | decrease | fair_brier | -0.49% | [-0.0013278775685381964, 0.002312241996203772] |
| full_vs_graph_affine | 10 | decrease | tail_weighted_fair_crps_mm | -0.06% | [-0.03395044152431307, 0.028403343601394206] |
| full_vs_graph_affine | 20 | increase | fair_brier | 0.45% | [-0.0015108655153759567, 0.0009455065785117023] |
| full_vs_graph_affine | 20 | increase | tail_weighted_fair_crps_mm | 1.32% | [-0.02077123870185393, 0.002285012693122918] |
| full_vs_graph_affine | 20 | decrease | fair_brier | 0.04% | [-0.0016635298284518908, 0.0012834267557554904] |
| full_vs_graph_affine | 20 | decrease | tail_weighted_fair_crps_mm | 0.20% | [-0.018463628268958556, 0.013301501735844296] |
| full_vs_graph_affine | 30 | increase | fair_brier | 1.51% | [-0.0008868300426908361, 7.170735501445088e-05] |
| full_vs_graph_affine | 30 | increase | tail_weighted_fair_crps_mm | 2.09% | [-0.009415670280534956, -0.0011733338341979682] |
| full_vs_graph_affine | 30 | decrease | fair_brier | 0.02% | [-0.0006904287810069678, 0.0006191735413745141] |
| full_vs_graph_affine | 30 | decrease | tail_weighted_fair_crps_mm | 0.44% | [-0.008850922224846465, 0.004594392934789607] |
| local_vs_affine | 10 | increase | fair_brier | -0.32% | [-5.0832073567574325e-05, 0.001099084186272958] |
| local_vs_affine | 10 | increase | tail_weighted_fair_crps_mm | -0.08% | [-0.006416950642038035, 0.009591825392116074] |
| local_vs_affine | 10 | decrease | fair_brier | -0.12% | [-0.00047900184793878955, 0.0007707717540724675] |
| local_vs_affine | 10 | decrease | tail_weighted_fair_crps_mm | 0.04% | [-0.009249295483353792, 0.007051085842586362] |
| local_vs_affine | 20 | increase | fair_brier | 0.07% | [-0.00039642715297099793, 0.0003251832640731485] |
| local_vs_affine | 20 | increase | tail_weighted_fair_crps_mm | 0.17% | [-0.004429882432441873, 0.0021704767659379886] |
| local_vs_affine | 20 | decrease | fair_brier | 0.06% | [-0.0004645291150984667, 0.00032196803003656027] |
| local_vs_affine | 20 | decrease | tail_weighted_fair_crps_mm | 0.21% | [-0.004965667352680292, 0.001752245521918257] |
| local_vs_affine | 30 | increase | fair_brier | 0.35% | [-0.00020694026509565655, 4.569051713231021e-05] |
| local_vs_affine | 30 | increase | tail_weighted_fair_crps_mm | 0.11% | [-0.0017230853928906504, 0.0012722731941811618] |
| local_vs_affine | 30 | decrease | fair_brier | 0.26% | [-0.00020942550174893773, 6.629510669268296e-05] |
| local_vs_affine | 30 | decrease | tail_weighted_fair_crps_mm | 0.30% | [-0.0017063666877367622, -3.3535128656272827e-06] |

fair Brier扣除有限IID成员概率估计方差，尾部加权fair CRPS通过对增量分布做上/下阈值截尾计算。它们均在全部有效起点上评分，不只在发生事件时评分；这两个分数理论上非负，模型间配对差值可负。共同月份2000次重采样、两个种子共享月份；未做多重比较校正。

本报告检查已冻结模型，不训练、调门控或更改权重。主日历对照含GraphAffine/Affine；补充日期仅检验冻结Full/Local相对原GP。事件计数、逐月和来源见[event_scores.json](event_scores.json)。
