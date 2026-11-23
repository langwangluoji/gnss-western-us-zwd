# 服务器实验索引

项目总目录为 `/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`。从这里打开 `runs/EXPERIMENTS.md` 查阅批次用途、模型和结果位置。

## 批次命名

在 `runs/` 中使用 `YYYYMMDD_<任务>_<阶段>_vNN`。日期取建批日，任务描述研究对象，阶段说明运行目的，版本从 `v01` 开始。使用字母、数字和下划线。用同一个标识填写 `campaign plan --name` 和 `--directory` 的末级目录。

| 批次名示例 | 用途 | 结果位置 |
|---|---|---|
| `20261118_us_dense_zwd_l40_bootstrap_v01` | 首次部署。第 0 阶段因 CPU 作业被投到 L40 分区而遭拒 | `preflight.json`、`manifest.json` |
| `20261118_us_dense_zwd_l40_bootstrap_v02` | 分区修订版。CPU 检查走 Intel，GPU 基准走 L40 | `benchmark/benchmark.json` |
| `20261118_us_dense_zwd_research_v01` | 美国区域 GNSS ZWD 场重建的正式模型比较 | `REPORT.md`、`seed_42/evaluate2020/summary.json`、`seed_42/export/` |

创建批次时登记实际日期和状态。更改数据、模型、训练划分、选模协议或调度配置时登记新版本。断点续跑沿用原批次名和目录，提交记录保存在 `manifest.json`。

## 正式批次中的目录

| 相对正式批次的路径 | 对应实验 |
|---|---|
| `seed_42/mean_base/` | 基础场 M0，使用 2016–2017 年数据训练 |
| `seed_42/mean_all/` | 基础场对照 M0-all，使用 2016–2018 年数据训练 |
| `seed_42/basis_cv_0/`、`basis_cv_1/`、`basis_cv_2/` | 空间基三折验证，后两项也在 `seed_42/` 下 |
| `seed_42/search/<模型>/trials/trial_0000/` | 该模型的第 0 次参数搜索 |
| `seed_42/promoted/<模型>/trial_0000/` | 该模型的第 0 号候选晋级后的完整训练。以 `promotions.json` 核对实际晋级编号 |
| `seed_42/ablations/<变体>/` | 单因素消融，如 `current_only` 表示只用当前帧 |
| `seed_17/`、`seed_101/` | 通过推进门槛的模型在种子 17、101 上的复验 |

模型名 `det` 表示确定性系数预测，`gaussian` 表示高斯系数模型，`edm` 表示扩散模型，`gan` 表示生成对抗模型。`m2_f` 和 `m2_j` 分别表示梯度隔离与联合训练的边缘高斯模型。

Slurm 队列里的 `gnss-<阶段编号>-array` 是作业数组名。通过批次的 `manifest.json` 将作业号和数组下标对应到具体任务，再用 `campaign.json` 查阶段与模型参数。同一阶段可能包含多个实验，重试时按实际作业号查对应记录。

## 后续登记

每个批次登记目录名、研究问题、模型或消融项、数据版本及时间划分、随机种子、配置或父批次、状态和结果位置。正式批次的精确配置在 `resolved.json`，代码、数据和评价协议的哈希在 `campaign.json`。

以后出图时，在项目总目录的 `results/<批次名>/figures/` 保存图件，文件名用 `<模型或对照>_<数据阶段>_<指标或内容>.<格式>`，例如 `edm_vs_gaussian_2020_joint_es.pdf`。把对应的指标文件路径和所用检查点一起登记。
