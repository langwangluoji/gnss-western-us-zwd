# ZhouSiyu服务器部署

实际工程总目录为`/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117`。

服务器已创建Conda环境`ZhouSiyu`，Python版本3.11.18。创建时将`CONDA_PKGS_DIRS`临时设为项目中的`cache/conda_pkgs`，随后通过`unset CONDA_PKGS_DIRS`恢复默认缓存配置，终端输出已确认。

本次使用`configs/tongji_l40_zhousiyu.yaml`。它指定`ZhouSiyu`环境、CPU分区`intel`及GPU分区`L40`。GPU作业申请1张L40、7核CPU和64 GB内存。科学参数沿用已验收的主配置。

目录安排如下。

| 目录 | 内容 |
|---|---|
| uploads | 上传的代码ZIP、数据ZIP |
| code/gnss_field_server | 解压后的工程源码、配置及wheel |
| data/gnss_field_server_v1 | 解压后的分角色月片数据 |
| cache | 可重建缓存 |
| runs | 按批次归档实验清单、日志、检查点和预算锁 |
| results | 汇总与导出结果 |

先完成文件上传与解压，再安装依赖和工程。随后进行服务器预检、限定4卡时基准及预算锁定。每次规划作业均显式传入本次配置文件，Slurm脚本由配置生成。

创建批次前按[实验索引](EXPERIMENTS.md)命名，用同一个标识填写 `--name` 和 `--directory` 的末级目录。将索引放在服务器的 `runs/EXPERIMENTS.md`，按实际批次更新状态和结果位置。

## 同济分区修订与基准运行

同济平台要求L40分区作业显式申请GPU。旧批次第0阶段被拒绝，因为脚本把CPU数据包检查投到了L40。原批次保留提交记录，新批次使用调度修订版。CPU数据包检查、审计、环境预检和每个阶段的汇总使用`intel`。前3个CPU阶段各使用2核、8 GB和15分钟限时，汇总使用1核、4 GB和10分钟限时。GPU基准申请1张L40，限时4小时。

在Windows PowerShell中，从上次的`$packageDir`上传代码修订包。

```powershell
scp -P 10022 "$packageDir\ZhouSiyu_DF_GAN_20261117\slurm_v02\gnss_field_server_code_v02.zip" u23114@logini.tongji.edu.cn:/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/uploads/
```

在服务器SSH窗口安装修订包，生成新的计划与预检记录。

```bash
conda activate ZhouSiyu
export PYTHONNOUSERSITE=1
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
batch_name="20261118_us_dense_zwd_l40_bootstrap_v02"
batch_dir="$project_root/runs/$batch_name"
config="configs/tongji_l40_zhousiyu.yaml"

python -m zipfile -e "$project_root/uploads/gnss_field_server_code_v02.zip" "$project_root/code/v02" &&
cd "$project_root/code/v02/gnss_field_server" &&
python -m pip install --no-index --no-deps dist/gnss_field-0.1.1-py3-none-any.whl &&
python -m pip check &&
python -m gnss_field --config "$config" campaign plan --scope bootstrap --name "$batch_name" --directory "$batch_dir" &&
python -m gnss_field --config "$config" doctor --site Tongji --campaign-dir "$batch_dir" --output "$batch_dir/preflight.json"
```

确认新预检显示`verified: true`，然后用Slurm测试三个脚本的资源设置。测试返回的作业号和预计时间来自排程估算，实际队列以`squeue`为准。

```bash
sbatch --test-only "$batch_dir/slurm/phase_000.sh"
sbatch --test-only "$batch_dir/slurm/phase_003.sh"
sbatch --test-only "$batch_dir/slurm/phase_003_summary.sh"
```

测试通过后先提交第0阶段。

```bash
phase=0
python -m gnss_field campaign submit --directory "$batch_dir" --phase "$phase" \
  --preflight "$batch_dir/preflight.json" --benchmark-cap 4 --execute
python -m gnss_field campaign status --directory "$batch_dir" --refresh
```

每阶段的任务达到`SUCCEEDED`且作业与汇总都达到`COMPLETED`后，再依次设`phase=1`、`phase=2`、`phase=3`，重复提交与状态检查。登录后若更换了登录节点，先在当前节点重做预检。运行期间可用`squeue -u "$USER"`检查排队原因，日志位于`$batch_dir/logs/`。基准结束后读取`$batch_dir/benchmark/benchmark.json`，确认真实L40与信号恢复验证通过，再用实测结果制定正式预算并锁定。

原本地数值验收证据对应原发布版本。此次修订只更改调度和部署，模型代码与科学参数沿用已验收版本。
