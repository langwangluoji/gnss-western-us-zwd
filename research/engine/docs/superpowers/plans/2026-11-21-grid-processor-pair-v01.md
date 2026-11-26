# 69通道点式MLP与ResU-Net配对训练实施计划

> **For agentic workers:** Use executing-plans; execute inline with tests and one final independent review. User authorized implementation and training on 2026-11-21.

**Goal:** 启动全资格时间覆盖下的配对空间处理器训练，提供可恢复权重、真实开发指标和实时曲线。

**Architecture:** 两臂共享固定M0＋IDW混合底座、69通道条件及联合Gaussian。MLP逐格点处理，ResU-Net使用32/64/128/128通道；两臂均经同一发布格网查询监督真实B站。新实验不自动启动EDM/GAN或新增搜索。

**Tech Stack:** 现有PyTorch/NPY/TrainEngine，本机工程验收，单A800顺序队列。

**Spec:** `results/20261121_unet_architecture_literature_v01/REVIEW.md`及该目录Pro答复；用户已同意执行。

## 约束与裁决

- 数据为B2018；C不参与本轮选模，D2020封存。M0、旧混合权重、g(A)、Gaussian均冻结。
- 两空间折沿用既有物理组分配。6288资格小时扣除既有四个内层整周及±18h缓冲，预计5472训练小时；必须实算断言。外层用已登记2016小时完整周，仅在选模后读结果。
- batch8，seed42，ES K4；EMA .99、AdamW lr1e-4、weight decay1e-4、warmup100、最低lr1e-6、裁剪1。最多4遍、2736更新/臂/折。第200步启动检查不选模，每半遍正式评价，2遍后patience3；零修正保留。
- 输入依旧五变量共同有效、四帧、75/150km核；创新相对冻结M0。固定科学通道归一化：现有训练统计标准化值与幅值，密度log1p/log1p(174)，坐标按固定投影域范围缩放。无需用标签或开发时刻重新拟合统计，两臂相同。
- 公开网格144×96全矩形，陆地/支持作为通道与标记；不把缺测区当物理ZWD零值。M0缓存与新69通道缓存按小时懒构建到单独NPY，缓存冻结输入，不截断卷积梯度。
- 当前目录不是Git仓库。裁决：新文件及独立运行目录隔离，运行前保存源码快照及SHA256清单，不初始化/改动其他仓库。
- 本轮授权单卡训练；A800先实测锁定墙钟上限，超出上限保存暂停，不无限自动重试。无活动SSH，部署包验收后请用户在可见SSH窗口登录。

## Task 1：模型和科学接口

- [x] 新建tests/test_grid_processor_pair.py，验证零初值、邻域感受野、格网查询高程、共同mask/置换、缺站、小时隔离与缓冲。
- [x] 观察测试失败，再实现src/gnss_field/models/grid_processor.py及tools/grid_pair_data.py。
- [x] 使用全矩形输入，两臂查询同一已发布格网。B标签只用于objective/evaluator，不进入feature构造。

## Task 2：训练、恢复与监控

- [x] tools/grid_pair_train.py复用TrainEngine；同小时顺序及独立Gaussian流，累计真实监督站时刻。
- [x] 检验真实数据、权重更新、启动评价有限，100＋100恢复与连续200一致；工程更新另目录。
- [x] tools/grid_pair_monitor.py仅读追加日志，5秒刷新RMSE与ES，保留最后更新时间；打开页面和日志。

## Task 3：部署与启动

- [x] 生成来源锁、配置、代码快照与Linux部署包，单A800短基准后预算门槛，四组顺序运行。
- [x] 独立审阅、测试通过后启动正式队列，核实实际更新与指标，再交接路径和暂停恢复方式。

## Review Focus

源码/缓存变化必须拒绝恢复；懒构建M0不得消耗训练随机流；所有有效站预测必须有限；零候选不得丢失；训练/内层/外层小时与角色隔离；超时和SSH断线不损坏checkpoint。
