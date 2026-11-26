# 现有实验索引生成器审阅

`catalog.py` 整理已有目录的协议、状态和报告来源，生成可回到原文的导航。原数值、科学锁和报告保留在来源位置。未执行训练、评价、服务器任务或标签读取。

接口：`build_catalog(root: Path, output: Path, source_engine: Path | None = None) -> dict`。相对参数以 `root` 为基准。输出为 `output/index.json`、`output/README.md` 和 `output/legacy/<原目录名>/README.md`。来源名称冲突时加入 `us__` 或 `research_campaign__` 前缀。

迁移后的默认工程路径为 `research/engine`，也可通过 `source_engine` 指定。`original_path` 保存生成时的实际公开来源路径，`historical_path` 依据 `maintenance/paper_restructure_20261126/path_mapping.json` 逆映射恢复迁移前路径；各来源文件同时补充历史路径。所有文档链接按相对位置生成。

扫描优先覆盖 `experiments/us` 中的公开轻量结果，以及 `research` 中带有 Markdown 报告的 campaign，排除 engine、旧清理目录及纯代码目录。每个来源扫描至5层，只打开选定名称的 Markdown/JSON 元数据；单文件上限512 KiB。超限文件保留链接和跳过原因。Windows junction 和符号链接在遍历前剔除，避免进入本地 cache、runs、probability、evaluation。原始观测、标签目录、NPY/NPZ、模型权重和CSV逐站表不读取。文件读取成功时保留 SHA-256，原报告通过公开来源链接访问。摘录中的原报告相对超链接保留显示文字，访问统一通过明确的原文来源链接。

类别由文档标题、明确用途声明或 JSON 的 `experiment_role` / `evidence_role` 字段提供依据，分为 `formal`、`diagnostic`、`historical`、`engineering` 和 `uncertain`。多种用途并存时依次优先保留 historical、engineering、diagnostic；不据此承担正式排名判断。未找到明确用途时记为 `uncertain`。

状态保留来源中的 `status`、`stage`、`state`、`completion_status` 字段，以及带来源行号的阶段声明。子运行和历史报告中的状态分别记录，不归并为单一 COMPLETE，也不依据名称、指标存在或文件时间推断已完成。`reported_states` / `reported_prose` 表示找到了来源声明，`no_evidence` 表示没有找到，均不代表独立复算结论。

每项条目包含目标、来源路径、来源文件清单、类别依据、状态依据、历史数据角色与信息条件摘录、缺失证据清单。目标的回退来源为文档标题；未定位到目标时明确标记。所有归纳保持来源片段，不改写指标。

已完成核验：

- Python AST 语法解析通过，未生成 `.pyc`。
- 只读抽查 `20261123_simple_weather_unet_v01`，正确保留 `PAUSED` 状态及3份来源文件。
- 无元数据的模拟目录 `COMPLETE_formal_annual` 仍返回 `uncertain` 和 `no_evidence`，验证名称不构成完成或正式证据。
- 已在迁移后路径执行 `python maintenance/paper_restructure_20261126/catalog.py --root . --output experiments --source-engine research/engine`。
- 共生成147项：137个美国公开实验目录、10个研究 campaign。其余没有公开副本的本地结果目录不在本索引覆盖范围。
- 类别分布为 uncertain 133、diagnostic 13、engineering 1；未找到明确 formal 用途证据的目录保留 uncertain。
- 总索引和147份逐目录 README 共148份，3437个本地链接目标全部存在，解析后没有指向 artifacts、engine/results 或旧工程 results 的目标。
- 147项均取得历史目录逆映射；来源路径全部位于 `experiments/us` 或 `research`。

后续全量核查重点为条目数量、跨目录重名、来源链接可达性，以及原锁位于实验目录外的条目。扫描深度、文件大小和明确名称白名单限定了本次元数据覆盖范围；未扫描内容不会被写成“已验证”。
