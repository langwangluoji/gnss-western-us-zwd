"""Static, GitHub-compatible diagram of the implemented anomaly encoder."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('draw', ROOT/'maintenance/transformer_architecture_20261125/render_diagrams.py')
draw = importlib.util.module_from_spec(spec); spec.loader.exec_module(draw)
draw.OUT = ROOT/'docs/figures/station_anomaly_20261125'; draw.OUT.mkdir(exist_ok=True)
n, e = draw.node, draw.edge
result = draw.render('architecture', '保留约10 mm基础场的条件站异常时间编码',
    '已实现实验A · 与原始站点时间输入S直接配对 · 当前时刻空间重建R', [
    n('input', 50, 89, 90, 9, '82／410条件站连续72小时ZWD／T／P＋mask／age\n站点坐标、高程、地形与海岸距离；目标站标签只用于训练'),
    n('base', 26, 71, 43, 15, '冻结原GRU＋p14基础场\n保留原始观测和8×8历史\n原权重、归一化及两级U-Net', 'space'),
    n('anomaly', 74, 71, 43, 15, '逐小时剔除同一物理组\n条件站空间背景→地表异常\n异常／10 mm＋有效性mask', 'train'),
    n('time', 74, 50, 43, 15, '每站独立72h×11通道\n宽32、2层4头因果Transformer\n保留T／P、年龄、日历与静态', 'time'),
    n('query', 50, 30, 90, 15, '任意查询坐标：最多16邻站的方向、高差、海岸及物理创新\n32维时间状态＋局地边→查询交叉注意力\n融合113维点特征、静态9、参考1与U-Net两级16＋32', 'space'),
    n('output', 50, 11, 90, 10, '204→64→64→1零初始化残差头；修正范围±5 mm\n最终ZWD＝冻结原参考＋新增修正；最终3000步raw与EMA同轨迹评价', 'space')], [
    e([(26, 84.5), (26, 78.5)]), e([(74, 84.5), (74, 78.5)]),
    e([(74, 63.5), (74, 57.5)]), e([(74, 42.5), (74, 37.5)]),
    e([(26, 63.5), (26, 37.5)]), e([(50, 22.5), (50, 16)]),
    e([(95, 89), (98, 89), (98, 30), (95, 30)], dashed=True)],
    '2016–2019连续训练；2020抽样开发配对。A同时替换ZWD值与mask，检验整套异常表征。')
result['status'] = 'IMPLEMENTED_MATCHED_TRAINING_NOT_ACCURACY_CLAIM'
(draw.OUT/'manifest.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
print(result['text_bounds'])
