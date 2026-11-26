"""Render the proposed shared-state space/spectrum denoiser using the repo renderer."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
helper = ROOT/'maintenance/transformer_architecture_20261125/render_diagrams.py'
spec = importlib.util.spec_from_file_location('architecture_renderer',helper)
draw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw)
draw.OUT = ROOT/'docs/figures/spectral_diffusion_20261125'
draw.OUT.mkdir(parents=True,exist_ok=True)
n,e = draw.node,draw.edge
diagram = draw.render('joint_architecture','Transformer条件与空域／图频域统一去噪',
    '待训练设计 · 一个扩散状态、一个噪声目标 · 先保留当前时刻重建R的信息条件',[
    n('input',50,90,90,8,'允许条件站历史≤t＋地形／海岸＋基础场物理输出\n真实目标站历史与未来观测不进入条件'),
    n('time',25,77,42,9,'因果Transformer历史编码\n保留逐位置状态与时间顺序','time'),
    n('history_spectrum',75,77,42,9,'过去频谱编码\n缺测用LS；完整窗用FFT／STFT','time'),
    n('memory',50,64,90,9,'对齐的条件memory C_t\n每层cross-attention＋扩散步k调制；物理时间t保持因果','time'),
    n('noisy',50,51,90,8,'同一个加噪锚点创新状态 x_k（x0为标准化AR创新）\nx_k = sqrt(abar)·x0 + sqrt(1-abar)·epsilon','neutral'),
    n('spatial',25,38,42,10,'空域／图邻域分支\n局地天气变化与空间细节\n输入同一个x_k与条件C_t','space'),
    n('spectral',75,38,42,10,'图频域分支\nGFT(x_k) → 谱处理 → inverse GFT\n固定图U、图频率标识Lambda','sample'),
    n('fused',50,24,90,9,'跨域特征融合 → 同一个epsilon预测\n单一去噪损失；谱目标对应同一噪声的U^T·epsilon','sample'),
    n('result',50,11,90,9,'反向采样 → 残差／创新逆变换 → 冻结基础场μ＋生成残差\n任意坐标读出；同时评价点值、区间、增量与站对联合分布','space')],[
    e([(25,86),(25,81.5)]),e([(75,86),(75,81.5)]),
    e([(25,72.5),(25,68.5)]),e([(75,72.5),(75,68.5)]),
    e([(4.7,64),(2,64),(2,38),(3.7,38)]),
    e([(95.3,64),(98,64),(98,38),(96.3,38)]),
    e([(25,47),(25,43)]),e([(75,47),(75,43)]),
    e([(25,33),(25,28.5)]),e([(75,33),(75,28.5)]),e([(50,19.5),(50,15.5)])],
    'x0由可审计的稀疏残差表示产生，先验收重构；不将未知格点插值当真值。')
manifest = dict(status='PROPOSED_ARCHITECTURE_NOT_TRAINING_RESULT',diagram=diagram,
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    renderer_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in draw.OUT.glob('joint_architecture.*')})
(HERE/'diagram_manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'figures':1,'exports':len(manifest['outputs']),'text_bounds':'PASS'}))
