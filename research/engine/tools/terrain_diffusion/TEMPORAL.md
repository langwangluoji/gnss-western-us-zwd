# 连续数据与因果预报：本机运行

在gnss_field_server工程根目录执行，使用Anaconda `pytorch`环境。首次制包读取本地原始站点观测并下载Natural Earth公开海岸线，不读取确认站或C/D标签。原始产品时间保持名义小时。

```powershell
conda activate pytorch
python -m pytest tools/terrain_diffusion/test_temporal_data.py tools/terrain_diffusion/test_temporal_encoders.py tools/terrain_diffusion/test_temporal_experiment.py tools/terrain_diffusion/test_fixed_support.py -q
python -m tools.terrain_diffusion.temporal_data --out results/20261125_temporal_continuous_data --workers 4
python -m tools.terrain_diffusion.temporal_experiment --data results/20261125_temporal_continuous_data --out results/20261125_temporal_forecast_screen --steps 600 --train-events 384 --histories 12 48 72 --seeds 42 17 --kinds none mlp gru lstm transformer
python -m tools.terrain_diffusion.temporal_report --out results/20261125_temporal_forecast_screen
```

已有合同或运行协议的目录拒绝覆盖；复现实验使用新输出目录。海岸下载失败时保留已制数组和原始QC收据，恢复同一数据来源后重新制包；当前还未实现自动从断点跳过原始解析。

模型训练需要CUDA。随机种子固定，CUDA网格采样反向允许非严格确定性，因此不能承诺逐bit相同权重；重载保存的预测应精确复算现有JSON指标。

训练包`values.npy`包含训练角色410站的全部5年，窗口索引控制实际时间用途；开发标签单独存为`dev_values.npy`，只供评价。模型加载前验证合同内所有文件哈希。2020标签可预缓存，训练梯度与选模仅由2016–2019数据决定。

本轮是小型空间载体和C1海岸特征的本机预报筛查；完整计划及尚待实施部分见仓库`docs/NEXT_STAGE.md`与本轮`ANALYSIS.md`。权重、原始数组和预测数组保留本地，Git只同步轻量代码和证据。
