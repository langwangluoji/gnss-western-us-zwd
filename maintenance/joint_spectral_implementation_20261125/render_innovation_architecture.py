"""Implemented condition-innovation mechanism, including the training-only path."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('draw',ROOT/'maintenance/transformer_architecture_20261125/render_diagrams.py')
draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)
draw.OUT=ROOT/'docs/figures/observed_innovation_20261125';draw.OUT.mkdir(exist_ok=True)
n,e=draw.node,draw.edge
result=draw.render('architecture','可见站模型误差：基础场的时空创新编码',
    '已实现I0／I1／I2机制对照 · 源模型为既有GRU＋p14 · 解析头与神经网络训练分开计数',[
    n('input',50,90,90,8,'同一组82个条件站：连续ZWD／T／P＋mask／age＋地形／海岸\n源背景每次读取72小时；加入3h历史创新后总跨度75小时'),
    n('condition',26,75,43.5,15,'冻结源模型在条件站位置的预测\nr_i(t)＝可见观测z_i(t)−mu_i(t)\n当前、1h、3h均按各自过去时刻计算','space'),
    n('query',74,75,43.5,15,'同一冻结源模型\np14点特征＋两级U-Net＋GRU\n在任意查询位置输出mu_q(t)','space'),
    n('fields',50,54,90,13,'可见站模型误差 → 3 km高程输送 → 50／150／300 km软核查询\n历史由相同条件站集合独立重算；缺测不进入分子／分母\n50 km弱支持区按1e−8分母下限平滑回退','time'),
    n('training',26,34,43.5,13,'仅训练：隐藏目标残差标签\nD距离权重＋0.1小时增量\n3000连续块累计正规方程＋岭项','train'),
    n('heads',74,34,43.5,13,'I0：当前3维\nI1：当前＋1／3h，共9维\nI2：再加年度sin／cos交互，共27维','time'),
    n('output',50,15,90,12,'原物理单位输出：ZWD＝mu_q＋clip(10 mm × beta^T × phi，−5 mm，＋5 mm)\n源权重冻结；beta只由2016–2019监督目标拟合\n2020固定12个月×72h×3布局开发评价，保持原始真值与源权重配对','space')],[
    e([(26,86),(26,82.5)]),e([(74,86),(74,82.5)]),e([(26,67.5),(26,60.5)]),
    e([(26,47.5),(26,40.5)],True),e([(74,47.5),(74,40.5)]),
    e([(47.75,34),(52.25,34)],True),e([(74,27.5),(74,21)]),
    e([(95.75,75),(98,75),(98,15),(95,15)])],
    '后续：重建空间折外基础场 → 训练折残差标准化 → Gaussian／Diffusion同条件对照。')
result['status']='IMPLEMENTED_DETERMINISTIC_INNOVATION_REFINEMENT'
(draw.OUT/'manifest.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(result['text_bounds'])
