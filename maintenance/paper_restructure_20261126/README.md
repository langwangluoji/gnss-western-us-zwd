# 现有项目整理与安全清理

用户2026-11-26要求先整理现有项目，并删除可以确认删除的文件。本阶段完成目录迁移、已有结果说明和安全清理；正式新训练另按已批准计划执行。

## 现在从哪里读

- [最佳结果](../../paper/README.md)
- [模型各模块贡献](../../paper/MODULE_CONTRIBUTIONS.md)
- [架构与流程图](../../paper/ARCHITECTURE.md)
- [147项已有公开实验索引](../../experiments/README.md)
- [34个仅本地保留运行目录](../../archive/LOCAL_RUNS.md)

核心包移至`src/gnss_field`，研究工具与测试移至`research/engine`，公开轻量结果移至`experiments`，本地数据、权重和预测移至`artifacts`。旧访问路径是可重建的本地兼容junction，不进入GitHub导航。历史支持源和迁移前导航在`archive`。

## 删除了什么

删除两个重复输入缓存，共 **225,534,208字节（215.09 MiB）**：原覆盖扩展实验的`cache/features.npy`和`cache/innovation.npy`。两文件分别与保留的`data.npz`中同名NPY成员逐字节SHA256相同，完整NPZ也符合原数据完成锁。旧缓存游标移入本目录，避免把已删副本误认为缓存完整。

恢复时从保留NPZ提取同名NPY成员即可，无需训练。完整证明见[删除清单](delete_plan.json)。原始观测、唯一权重、正式预测、负结果和科学锁保留。

命名为cache的欧洲模型缓存、折外文件、旧native_weather缓存仍被引用，全部保留。1.56GiB的旧服务器数据包包含此前已清理缓存的恢复内容，也保留。

## 身份与核验

- 清理前本地全量清单：`inventory_before.local.csv`，不进Git；摘要见[inventory_before.local.json](inventory_before.local.json)。
- [路径迁移映射](path_mapping.json)：7415项原跟踪文件。
- [完整性核验](integrity_verification.json)：16164个受保护科学文件、92,250,828,293字节通过SHA256复核；不解析观测标签。6个可重建的egg-info安装元数据及本次执行中更新的计划文件单列排除，前后身份与理由保存在回执中。
- 5098个迁移JSON与旧提交逐字节相同，历史科学锁未批量改写。
- 模型源码的数值路径保持；迁移补充了数据路径、测试UTF-8、延迟外部VMF依赖和新布局release摘要覆盖。原版源码及锁可从标签`archive/pre-paper-restructure-20261126`复现，不能跳过旧锁的源码校验。
- 原始少量改动源的副本保存在`archive/legacy/engine_source`，导航旧版在`archive/legacy/navigation_before`。

Windows长路径检查使用extended-length前缀。最初824项普通路径存在性误报已逐项确认存在，并在修正检查器后完成整套哈希复核。

[最终核验摘要](completion.json)记录本机604项通过、独立公开检出602项通过及2项缺本地数据的明确跳过，历史一致性5项全部执行；wheel安装及独立CLI通过。[独立审阅](review_receipt.json)保留已关闭问题。

临时`build`和pytest缓存的递归删除被自动审批拦截，未返回详细原因；这些忽略的可重建缓存保留，未改用其他方式递归清空。

## 重复使用

```powershell
python -m research.layout
python -m pytest -q
python -m maintenance.paper_restructure_20261126.verify_migration
```

`migrate.ps1`对已迁移布局直接返回，不再次移动；`bootstrap.ps1`检查已存在链接的目标身份。删除清单是本次已执行记录，不是通用删除规则。核验脚本依赖本机原始清单和完整产物；公开克隆可运行单元测试，但完整数据回放需先恢复本地产物。
