# 20261121_unet_architecture_literature_v01

当前公开来源：`experiments/us/20261121_unet_architecture_literature_v01`；[来源目录](../../us/20261121_unet_architecture_literature_v01)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261121_unet_architecture_literature_v01`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第1行：# 稀疏 GNSS 场生成的 U-Net 适用性与架构选择

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261121_unet_architecture_literature_v01/STATUS.json](../../us/20261121_unet_architecture_literature_v01/STATUS.json)，status：REVIEW_COMPLETE_NOT_IMPLEMENTED

## 报告中的阶段声明

- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第3行：日期：2026-11-21。状态：文献核验与架构建议，未修改训练实现，未启动本机训练或收费作业。Pro 审阅另存，不以网页模型意见替代原始证据。
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第95行：本轮已在内置浏览器的“Diffusion融合研究设计”对话提交实际三臂结果、V2 草案、数据角色和原始问题，页面模式为 Pro；已观察到完整回复及“回答已完成”状态。最初菜单显示的使用限额未阻止实际提交。原文见 `PRO_RESPONSE.txt`，原请求见 `PRO_PROMPT.txt`，网页末端完成截图见 `pro_completed.jpg`。原始页面文本保留为 `PRO_PAGE_TEXT.txt`；原文可能含网页数学公式的重复表示。

## 历史证据范围与信息条件

- [experiments/us/20261121_unet_architecture_literature_v01/STATUS.json](../../us/20261121_unet_architecture_literature_v01/STATUS.json)，scope：targeted architecture literature review
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第40行：推荐先在同一信息条件与发布接口下比较 ResU-Net；如它的学习曲线和误差结构提示非局地组合不足，再以固定前端替换空间处理器为小 Swin。完整 Gridded-TNP 的自由点查询解码器会改变本项目发布契约，因此只借鉴编码/处理部分，最终仍经公共格网查询。
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第60行：标准 EDM 去噪训练需要所建模随机变量的训练样本。完整且合格的独立格网/模拟场可以支撑场先验训练；真实 B 站残差可以支撑观测空间分布的训练，但不等价于完整格网真值。IDW、克里金、岭逆算或高斯采样能作代理/先验初始化，其生成规则必须记录，并保留去掉该初始化的对照。
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第62行：外部分析场也需核对其 GNSS 同化来源、湿延迟转换和高程基准；“来自另一个数据产品”不自动等于统计独立真值。SDA 论文自身说明 HRRR 同化了其评价站资料，因此本项目应借鉴其先验与观测分层方法，不能照搬其独立性口径。
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第77行：6. 在完整 B 留出天气块上锁模型后作有限 C2019 开发评价，D2020 保持封存。单 A800 先实测再锁预算；本次调研不提交作业。
- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)，第108行：本轮产物是选型研究与下一轮有限比较建议。旧 V2 文件和历史实现保留；不把建议稿、原网页意见或合成示例记为完成的神经网络实验。

## 原始协议、结果与报告

- [experiments/us/20261121_unet_architecture_literature_v01/REVIEW.md](../../us/20261121_unet_architecture_literature_v01/REVIEW.md)（report；read）
- [experiments/us/20261121_unet_architecture_literature_v01/STATUS.json](../../us/20261121_unet_architecture_literature_v01/STATUS.json)（status；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
