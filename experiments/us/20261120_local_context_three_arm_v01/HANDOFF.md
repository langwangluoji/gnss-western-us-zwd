# 本机局部条件三臂短试交接

状态：正式训练及既定B外层评价完成，未晋级服务器。六组均1000步，合计6000次正式更新；恢复验证另计800次工程更新。C2019未新增评价，D2020封存。原Conda环境、冻结M0和旧发布目录保持原样。

主要文件：

- `RESULTS.md`：结果及解释口径。
- `results.json`：点值、交叉拟合ES、配对区间、推进门槛及实际监督量。
- `training_curves.png`、`outer_comparison.png`、`station_gain_maps.png`、`field_example.png`：已渲染图件。
- `fold_0/`、`fold_1/`：每组`selection.lock.json`、入选checkpoint、最后checkpoint、日志和外层逐站预测。
- `dataset/`、`M0_cache/`：768小时缓存、训练/内层/外层索引、归一化和输入来源。
- `code_snapshot/`、`implementation_manifest.json`：本轮源码、设计清单和校验值。
- `pytest_final.log`：268项通过。

监控：http://127.0.0.1:8778，每5秒读取实际追加日志；所有模型现已完成，页面保留最后记录。当前只保留本轮只读监控进程，未运行本轮训练进程。打开Codex面板的请求已排入当前聊天窗口。

从工程根目录恢复同一队列（已完成组会跳过，不增加更新数）：

```powershell
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u tools/local_context_train.py
```

若未来中断后的同一队列需要暂停，在此run根目录创建空文件`PAUSE`；恢复前删除该文件，再使用上述命令。恢复依赖已锁定源码和数据哈希，不改变1000步上限。模型架构、样本、学习率或预算改变时另建实验，不能借恢复命令扩训。

重新启动只读监控：

```powershell
$env:PYTHONPATH=Join-Path (Get-Location) 'src'
& 'D:/anaconda3-2025.12-2/envs/pytorch/python.exe' -u tools/local_context_monitor.py
```

Windows监控只读追加日志及单次发布元数据，不打开训练循环反复原子替换的`status.json`或`last.pt`。第一组曾在第800步状态保存时触发WinError5，已从同一checkpoint恢复完成；失败日志与恢复日志均保留。

后续研究先区分优化效率与输入中可预测的信息。此次6000步结果没有支持立即扩大局部头、扩秩或自动训练生成分支的依据；既有成熟mean继续作为已验收产品。
