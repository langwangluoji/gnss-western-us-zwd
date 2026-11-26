# 20261121_grid_generation_v02

当前公开来源：`experiments/us/20261121_grid_generation_v02`；[来源目录](../../us/20261121_grid_generation_v02)。

历史路径：`research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261121_grid_generation_v02`。

用途类别：`uncertain`。状态索引：`reported_states`。状态表示来源中的声明，各子运行分别列出。

## 目标

- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第1行：# Generation v02 runtime handoff

## 类别依据

未定位到明确证据。

## 状态依据

- [experiments/us/20261121_grid_generation_v02/completion_snapshot.json](../../us/20261121_grid_generation_v02/completion_snapshot.json)，campaign_queue.state：COMPLETE

## 报告中的阶段声明

- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第14行：Client submission: dist/20261121_grid_generation_v02_submission; copied to C:/Users/user/AppData/Local/Temp/gnss_gen3_submit_v02. Visible legacy console title GNSS DF GAN A800 v02. Uses per-command Python TCP transport to avoid direct OpenSSH route timeouts, keeps strict host-key verification; stores no password, no global network changes. Authentication completed; job 3645090 submitted on gpu8001, one A800. Live SSH snapshot stream is active.
- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第21行：After login: deployment verifies all bytes and old audit sources, copies accounting history, submits bounded one-A800 job. Then benchmark -> budget gate -> all-three true200 vs100+100 engineering checks -> fold0 stochastic/DF/GAN -> fold1 stochastic/DF/GAN -> pilot outer96 K32 evaluations and figures -> COMPLETE.
- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第32行：Verified at 2026-11-22 00:22 Asia/Shanghai: job3645090 RUNNING on gpu8001; queue ENGINEERING_RECOVERY_CHECK. Benchmark and original-cap budget gate passed because execution reached this stage. All-three actual GPU continuous200 vs100+100 checks are in progress; scientific fold logs remain empty. Do not report engineering updates as scientific training or completed startup acceptance.
- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第54行：2026-11-22 03:15:43 Asia/Shanghai snapshot: queue COMPLETE, all six runs completed684updates each;4104scientific updates. GENERATION_QUEUE_COMPLETE present in Slurm log. Captured final sacct still RUNNING at03:09:16 because monitor sampled immediately before process exit; current scheduler completion needs fresh SSH. Snapshot archived as completion_snapshot.json, compact inner metrics completion_status.json. Remote outer result files, report and grid artifacts require download before scientific ranking. No new training submitted.

## 历史证据范围与信息条件

- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)，第3行：User authorized independent stochastic ResUNet + DF + GAN training, single A800. Original full outer results audited: old frozen M0+IDW base RMSE7.577401/7.631033mm; MLP7.556044/7.642464; U-Net9.774505/9.615438. Old U-Net mean is NOT promoted. Both gen folds share frozen M0+IDW mean, old Gaussian prior/amplitude/tau. New generator architecture remains small ResUNet,71 inputs. No C2019/D2020.

## 原始协议、结果与报告

- [experiments/us/20261121_grid_generation_v02/HANDOFF.md](../../us/20261121_grid_generation_v02/HANDOFF.md)（documentation；read）
- [experiments/us/20261121_grid_generation_v02/completion_snapshot.json](../../us/20261121_grid_generation_v02/completion_snapshot.json)（status；read）
- [experiments/us/20261121_grid_generation_v02/previous/fold_0/resunet69/results.json](../../us/20261121_grid_generation_v02/previous/fold_0/resunet69/results.json)（metadata；read）
- [experiments/us/20261121_grid_generation_v02/previous/fold_1/resunet69/results.json](../../us/20261121_grid_generation_v02/previous/fold_1/resunet69/results.json)（metadata；read）

## 待核查项

- 未在扫描范围找到协议/计划/锁文件；可能位于目录外。
- 未在扫描范围找到原结果报告。
- 未找到明确的实验用途声明，类别暂记 uncertain。

本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。
