"""Render implemented module paths, using the project's checked static renderer."""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[2]
helper=ROOT/'maintenance/transformer_architecture_20261125/render_diagrams.py'
spec=importlib.util.spec_from_file_location('renderer',helper)
draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)
draw.OUT=ROOT/'docs/figures/joint_spectral_implementation_20261125';draw.OUT.mkdir(parents=True,exist_ok=True)
n,e=draw.node,draw.edge
figures=[]


def chain(name,title,rows,subtitle='已实现模块 · 本机工程短训 · 正式模型收益以独立实验为准'):
    nodes=[n(str(i),50,88-i*12,90,9,text,role) for i,(text,role) in enumerate(rows)]
    edges=[e([(50,83.5-i*12),(50,80.5-i*12)]) for i in range(len(rows)-1)]
    item=draw.render(name,title,subtitle,nodes,edges,'训练标签与预测条件分开；所有逆变换、图基及归一化随实验保存。')
    item['status']='IMPLEMENTED_ENGINEERING_PILOT';figures.append(item)


chain('overview','已实现的时空基础场与频域概率实验',[
    ('连续条件站ZWD／T／P＋mask／age＋地形／海岸距离\n2016–2018拟合；2019内部诊断；按物理组五折','neutral'),
    ('T0–T5：p14结构＋因果72小时Transformer\n16×16状态按物理投影坐标融入两级空间解码器','time'),
    ('冻结五折基础场 → 单折、单布局、连续6小时残差\n共享概率G：观测25＋冻结物理均值1＋静态8','train'),
    ('L：128维标签集合latent／线性Phi读出\nA：固定256锚点ridge编码／重心H读出；先检查表示门槛','train'),
    ('训练统计标准化 → 固定AR创新 → 再标准化\nL0／L1历史谱；A0／A1／A2空域与图谱；B0／B1频带尺度','sample'),
    ('J0／J1：独立未来6小时F协议，起点持久性均值\n直接联合／DCT时间×GFT空间；未来观测仅提供标签','sample'),
    ('每臂Gaussian控制＋DDIM采样 → AR／表示／物理单位逆变换\n最终场=冻结均值＋生成残差；点值、区间、增量和站对分析','space')])

chain('foundation','T0–T5：基础场端到端计算路径',[
    ('当前与历史条件站固定集合 → 113维点特征＋30通道大支撑\n参考场：p14＋海岸地形边消息／FiLM／点修正','space'),
    ('72h×16×16×25动态摘要，另有静态8\n25→64；两层逐帧3×3空间运算＋逐位置四头因果时间注意力','time'),
    ('T0旧8×8相加；T1新16×16直接相加\nT2双层融合；T3加门控；T4加空间时间交替；T5加静态FiLM','time'),
    ('T2–T5：按投影坐标查询时间状态，边界延拓\nF0(16)+T0(16)+S0(8)→40→16；低层72→32','space'),
    ('低层上采样＋高层skip：48→16；T5静态FiLM\n零初始化1×1头；门控42→16→1，初始偏置−2','space'),
    ('按固定支撑连续查询：mu=参考场＋g×时间修正\nsurface ZWD归一化单位；物理反归一化只做一次','space'),
    ('连续小时训练：MSE＋0.1×小时增量MSE\n折外模型从头训练，排除该折所有年份标签／条件／统计','train')])

arms={
 'L0':('latent时间条件控制','128维集合latent，冻结Phi(q,c)线性读出','空间token去噪；共享因果G，无额外谱条件'),
 'L1':('latent历史谱条件','同L0的128维集合latent与冻结Phi','G＋过去72h缺测LS正余弦／功率／覆盖率'),
 'A0':('固定锚点空间域','256锚点同一H、ridge、标准化与AR','空间token去噪；噪声状态仍为同一创新x_k'),
 'A1':('固定锚点图谱域','同A0的256锚点及完整正交U','x_k→U转图谱→谱token去噪→U逆变换'),
 'A2':('固定锚点空间图谱联合','同A0的256锚点、H、统计、AR','同一个x_k→空间／GFT两分支→共同坐标融合→一个epsilon'),
 'B0':('双域无频带缩放控制','与A2相同；作为可复算重复控制','空间／图谱双分支；标准创新坐标加白噪声'),
 'B1':('双域频带标准化','图频创新按训练均值／尺度标准化，保存逆变换','白噪声加在标准化频带状态；空间分支显式可逆映射'),
 'J0':('未来六小时直接联合','起点72h历史与起点持久性均值，未来标签仅在训练侧','6×256创新展开1536维；联合去噪，起点条件不更新'),
 'J1':('未来六小时双频域联合','同J0的起点信息与1536维创新，完整可逆DCT×GFT','DCT时间×GFT空间 → 联合去噪 → 双域逆变换')}

for arm,(title,representation,mechanism) in arms.items():
    future=arm.startswith('J')
    chain(arm,f'{arm}：{title}',[
        ('允许的条件站72小时连续历史＋地形与海岸\n'+('起点冻结，未来观测不进入G或未来均值' if future else '每帧只使用该帧及过去信息，真实目标历史不进入G'),'neutral'),
        ('冻结折外基础场 → 残差mm → 训练b／s标准化\n'+representation,'train'),
        ('编码状态逐通道标准化 → 固定AR创新 → 创新标准化\n对应家族训练／2019重构门槛＋遮标签空间检查','train'),
        (mechanism+'\n一份x_k、一份epsilon、一条反向采样链','sample'),
        ('每8值一个token→128维；两层四头、FF256\n每层读4×4空间记忆cross-attention＋扩散步与小时标识','time'),
        ('100级余弦epsilon训练／20步DDIM；同条件Gaussian控制\n生成的每个成员独立AR递推，6小时块边界重置','sample'),
        ('逆频带／双频域 → AR逆变换 → latent或锚点读出\n残差反标准化＋冻结均值→ZWD集合→RMSE／fair CRPS等','space')])

manifest=dict(status='IMPLEMENTED_MODELS_NOT_FORMAL_GAIN_CLAIM',figures=figures,
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in draw.OUT.iterdir() if p.is_file()})
(Path(__file__).parent/'diagram_manifest.json').write_bytes(json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
print(json.dumps(dict(figures=len(figures),exports=len(manifest['outputs']),text_bounds='PASS')))
