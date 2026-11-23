# 时空扩散设计：网页复核交接与研究记录

状态：NOT_SUBMITTED_BROWSER_INITIALIZATION_FAILED。2026-11-23内置浏览器两次初始化失败，未发送以下材料，也没有网页回答。恢复后需提供主设计全文及本交接，不假定网页服务能读取本地路径。

## 待提交任务

请独立审阅配套设计 `2026-11-23-masked-spatiotemporal-diffusion-design.md`，目标为稀疏GNSS下地形感知连续ZWD场的可发表方法研究。不要为支持方案而补造数据、论文或性能；先找具体反例和不可实现之处。

实际数据：西部美国25–50N、125–110W，2016–2020；A174为唯一GNSS条件输入，B511分为fit357/inner77/outer77；C110历史开发、D110登记回顾角色。本机小试72个30小时窗口（6h历史+24h输出），T/P为站点采样ERA5变量，当前地形144×96、25km。标签B为地表ZWD(mm)，A首变量Z0需按exp(-H/3000)转换；时标尚是名义产品小时。历史QC曾依赖PWV完整性，新输入禁止PWV/IWV/TCWV，新正式队列必须按允许变量重建。没有稠密ZWD真值，任意位置气象格网及细DEM尚未核实可用。模型直接生成ZWD，不采用旧均值场或旧权重。

历史结果：v04约9.003mm，仅开发集；v06遮站图预训练没有超过v04；v09时间损失改善CRPS/覆盖但点精度变差。24→64维残差训练投影拟合改善不等于外层收益。目标8mm尚未达到。

设计主线：MAE式A内部遮站预训练＋地形图条件编码＋因果时间编码＋当前小时空间联合v扩散，逐小时生成时同成员历史保持；B-fit只作监督。训练初始真实历史，随后4h自生成历史微调；冷启动/24/72h自由生成评价。任意查询通过几何编码，但目标集合变化可能改变边缘分布；需专门检查，不能把节点置换等变当查询一致性。

请审查：

1. 数据角色与遮挡是否匹配A174部署？预训练是否产生隐性泄漏？
2. 因果分解、真实历史训练到生成历史部署、缺测传播和采样预算是否自洽？
3. 节点扩散到稠密格网的瓶颈是否充分暴露，是否存在更简单的可检验替代？
4. 在没有密集真值的条件下，损失和指标能支撑哪些主张？空间块留出如何设计？
5. 论文来源是否支持被借鉴的机制？特别核查NDP第5节边缘一致性限制，勿把CorrDiff密集监督成绩转称本项目依据。
6. 哪些消融最能证伪方案？本机轻训预算应如何优先排序？

输出：按重要性列问题、定位章节、修改建议、所需证据及原始论文链接；最后说明哪些结论仍依赖真实训练。不要把此次另一轮模型回答当作独立实验或数学证明。

## 本地研究记录

检索日期2026-11-23，渠道为联网检索连接器与原始论文页面/PDF；这是定向设计检索，不报告PRISMA或穷尽覆盖。

主要检索串：

- CSDI conditional score based diffusion time series imputation 2021
- Denoising diffusion models functional data / Neural Diffusion Processes
- Masked Autoencoders Are Scalable Vision Learners 2022
- Convolutional conditional neural processes local climate downscaling
- geometric neural diffusion processes
- probabilistic downscaling weather diffusion CorrDiff
- Scheuerer Hamill variogram based proper scoring rules 2015
- Roberts cross validation temporal spatial hierarchical structure 2017
- GNSS zenith wet delay graph neural network spatial interpolation terrain

采用来源及阅读范围见主设计R1–R9。特别核验：CSDI条件/目标拆分及二维注意力；NDP第5节与附录E承认一般边缘一致性限制；convCNP正文支持任意位置查询与地形消融。MAE CVF页403后核验arXiv原始摘要；CorrDiff全文重定向失败，引用限制在原始索引架构描述。不以搜索排名或摘要替代本项目实际验证。

此前用户提供的本地DOCX论文此次未重新阅读全文，因此本设计不增加对该文的新方法或性能归因。

浏览器故障原文：`failed to write kernel assets: 系统找不到指定的路径。 (os error 3)`。首次调用为单一受支持入口，重置后再次单入口失败；无网页副作用，无上传，无网页模式声明。
