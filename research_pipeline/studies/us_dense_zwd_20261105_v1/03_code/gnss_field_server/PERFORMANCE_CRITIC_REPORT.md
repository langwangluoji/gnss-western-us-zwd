# ObservationCritic 向量化验证

`ObservationCritic.forward` 将逐行执行的站点对特征网络合并为一次批量调用。四个距离分箱分别完成批量索引选择，组成 `[B, 4, 256, 36]` 特征，再统一执行原有 `self.pair`。每个样本、每个分箱仍按原始几何字典序选择前 256 个有效站点对；填充项在聚合时清零，空分箱输出零。

点特征均值和方差、几何距离分箱、掩码语义、网络层、参数以及 state_dict 键均保持原样。此变更用于训练中的观测判别器，不调整生成器、损失、梯度惩罚、采样成员数或科学评价流程。

## 数值验证

先新增测试并运行旧实现：批量特征网络调用次数测试预期 1 次、实际 12 次，出现预期失败；其余 11 项通过。随后实现向量化，运行：

```powershell
& D:/anaconda3-2025.12-2/envs/pytorch/python.exe -m pytest tests/test_full_generative_critic_vectorized.py tests/test_full_generative_architecture.py -q
```

结果：**20 passed in 4.07s**。

新增测试使用保留的旧循环参考实现，检查 FP32/FP64 判别器分数、关于观测值和局部条件的一阶导数，以及 WGAN-GP 对输入、局部条件和判别器参数的二阶导数。FP32 使用绝对/相对容差 `1e-6`，FP64 使用 `1e-11`。同时覆盖零站点、单站点、空有效集、异构有效掩码、超过 256 对的截断、无效位置 NaN/Inf、填充站点、站点置换、距离边界和有效位置非有限输入报错。

曾启动全套 pytest，收到范围限定后中止，未获得完整汇总；本报告以以上定向测试为完成依据。中止时缓存记录了 `tests/test_full_generative_evaluation.py::test_neural_hour_batch_preserves_station_arrays[diffusion]` 和 `tests/test_local_context_experiment.py::test_monitor_open_handle_allows_atomic_status_replacement` 失败，未在本任务中诊断其原因。

## 本机 CUDA 基准

环境：NVIDIA GeForce RTX 4060 Laptop GPU，PyTorch `2.8.0+cu129`，FP32，CPU 线程数 2。固定随机种子 410，`B=8`、`N=511`、局部条件维数 32，每行恰有 442 个有效站点。几何坐标由均匀随机数乘以 `[350000, 350000, 2000]` 生成。旧实现与新实现使用相同参数和输入。

每个实现、每种模式预热 10 次，随后计时 30 次；每次结束调用 `torch.cuda.synchronize()`，记录包含 CPU 调度和 GPU 执行的墙钟时间。forward 模式使用 `no_grad`；GP backward 模式计算 `score.mean() + 10 * (||d score/d values||₂ - 1)².mean()` 并反向传播，每次将参数梯度设为 None，无优化器更新。几何分箱和有效站点对选择均包含在计时内。

| 模式 | 旧中位数 ms | 新中位数 ms | 加速 | 旧均值 ms | 新均值 ms |
|---|---:|---:|---:|---:|---:|
| forward | 41.049 | 7.115 | 5.77 倍 | 41.045 | 7.321 |
| forward + GP + backward | 159.114 | 23.131 | 6.88 倍 | 162.950 | 25.661 |

| 模式 | 旧增量峰值 MiB | 新增量峰值 MiB | 旧总分配峰值 MiB | 新总分配峰值 MiB |
|---|---:|---:|---:|---:|
| forward | 9.164 | 11.724 | 18.292 | 29.217 |
| forward + GP + backward | 58.583 | 53.439 | 76.077 | 71.173 |

增量峰值为本模式峰值减去预热结束后的已分配显存；总分配峰值包含同一进程内两套模型及前序模式保留的梯度，应按增量比较工作显存。CUDA 分数最大绝对误差为 **3.7253e-9**。

以上结果衡量合成输入下的观测判别器路径。完整 GAN 更新还包含生成器、解码器及其他训练步骤，其端到端耗时需要在实际训练中另行计量。
