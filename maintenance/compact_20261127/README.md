# 项目数据精简与恢复说明

2026-11-27；执行计划见[PLAN.md](PLAN.md)。本轮整理保留当前有结果支撑的模型链，退役旧中间数组，不启动研究训练，不改写实验锁。

## 保留什么

| 本地保留链 | 保留目的 |
|---|---|
| 美国p14＋GRU＋I1，以及I0/I2对照 | 当前重建与创新修正的关键对照 |
| TF3000、H_delta、joint/factorized双种子 | 时间编码、融合与有向时滞预报对照 |
| joint＋S175/G265/Affine/Local | 最新确定性点场与配对概率评价 |
| 美国连续时间数据、v10归一化包、v2天气缓存、核心embedding | 现有核心推理与核验入口所需数据 |
| 欧洲transfer与VMF3完整链、固定CNN权重和输出 | 欧洲独立训练、背景场增益与原模型对照 |
| 美国原始GNSS、地形、几何、站址与数据合同 | 数据来源和后续研究基础 |
| D盘旧ref最终frozen数据、原模型回放包、站点parquet | 保留旧研究最终数据和模型；旧ref气象原料不齐，不能宣称完整离线重建 |

美国完整保留目录由[compact.py](compact.py)中的`CORE`列出。模型对应的精度和适用范围继续见[论文结果指南](../../paper/README.md)，本次没有重新训练或产生新精度成绩。

## 精简范围

下表为已执行的原项目文件精简结果，见[删除回执](deletion_receipt.json)。单位GiB=2³⁰字节。按真实文件统计，不重复计算junction；排除`.git`、`.superpowers`和本次审计目录。

| 范围 | 整理前/GiB | 已删除/GiB | 原清单保留/GiB |
|---|---:|---:|---:|
| 当前项目 | 86.020 | 54.283 | 31.736 |
| `D:\SITP_cache` | 92.240 | 60.192 | 32.048 |
| 合计 | 178.259 | 114.475 | 63.784 |

实际删除3195个文件；本次新建的临时归档包另占2.681GiB，详见文末的自动审批拦截记录。该新增占用不计入上表原文件清单，因此114.475GiB是删除的旧文件体积，不是把临时包也扣除后的净缩减量。

包含Git、审计记录和上述临时包的项目实际目录约34.50GiB，D盘缓存约32.05GiB，见[整理后目录实测](disk_after.json)。

旧权重、优化器检查点、归一化/拟合尺度参数和≤4MiB的小数组归档到[GitHub Release](https://github.com/langwangluoji/gnss-western-us-zwd/releases/tag/archive%2Fpre-data-compaction-20261127)。模型压缩包内部恢复哈希、压缩包哈希、GitHub服务器资产SHA256与字节数核验一致后，才允许删除本地对应文件；轻量证据与恢复清单另回下载复核。未回下载的大包在回执中明确记录为服务器摘要验证。原始观测不上传Release，权重和GB数组不提交Git历史。

非核心的大型evaluation、embedding与训练缓存按退役处理：保留指标、曲线、代码、协议与来源哈希；这些大数组没有云端逐字节备份，重做旧逐样本分析需要按原流程重新生成，必要时重训。负结果说明继续保留，不将它们改写为当前模型的证据。

D盘清理范围为旧dense physics/validation H5、旧ref expanded中间H5和未完成partial。dense来源的60个月ERA5与静态场已逐文件校验，见[dense_source_verification.json](dense_source_verification.json)。最终ref frozen的训练、验证、UTC测试和GPST测试保留，不作为重复文件误删。

## 如何定位、核验和恢复

- [逐文件清单](inventory.csv)：原位置、字节数、SHA256、保留或退役理由。
- [逐实验空间清单](run_storage.csv)：美国各运行目录的本地保留、云归档与退役体积。
- [云端对象索引](archive_index.json)：SHA256对应的压缩包及成员名；原始路径从inventory查找。
- [审阅记录](review_receipt.json)：归一化参数恢复性问题及修复；科学依赖核对。
- [远端归档核验](remote_archive_verification.json)与[详细资产回执](remote_archive.json)：7个核心资产验证通过，另有校验和文本。
- [保留集合核验](retention_verification.json)：12352个科学文件、68303771249字节逐文件哈希一致，435份核心权重可加载；删除日志3195项与清单一致。此检查不解析目标观测值，不训练模型。
- [删除后全项目测试](postdelete_tests.json)：611 passed，286.61秒；测试覆盖本轮7项整理回归和原研究工程测试。
- [总体执行回执](completion.json)与[GitHub发布核验](publication.json)：本地执行、远端内容提交和临时包拦截分别记录。

恢复退役模型时，先按Release的`recovery_manifest.json`验证下载压缩包SHA256；从对应`retired-models-*.tar.gz`中读取`objects/<SHA256>.<扩展名>`，再按inventory中的原始路径恢复。相同内容去重保存，恢复后再次核验文件SHA256。`retired-evidence.tar.gz`保存原路径映射和未纳入网页索引的轻量记录。

旧完整历史环境使用标签`archive/pre-paper-restructure-20261126`恢复代码；本次精简前代码身份为`archive/pre-data-compaction-20261127`。Git标签保存代码，不保存已退役数组。原科学锁按原代码身份校验；不通过修改锁绕过代码哈希检查。

安全执行入口为[retire_files.ps1](retire_files.ps1)：全批预检、逐文件绝对路径边界/reparse检查、大小与SHA256校验，使用`Remove-Item -LiteralPath`删除指定文件。无递归清空；上轮被审批拒绝的build/pytest缓存不在删除范围内。

## 临时归档包的自动审批拦截

模型与证据已上传并核验。本次生成的`maintenance/compact_20261127/staging/`仍有7个压缩包/恢复清单，合计2879158305字节（2.681GiB）。原生PowerShell按清单删除、以及固定路径单文件删除均被自动审批拒绝，只返回`blocked by policy`，没有更具体理由；两次均未执行删除。见[拦截记录](staging_cleanup.json)。这些是可手工删除的上传临时副本，Git忽略该目录，模型恢复材料已在Release中。上表原项目文件的删除已独立执行并核验。
