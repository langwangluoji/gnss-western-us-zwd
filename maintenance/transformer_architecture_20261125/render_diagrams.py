"""Render proposed architecture diagrams; does not build or train a model."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/figures/transformer_fusion_20261125'
FONT = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'savefig.facecolor': '#ffffff'})
COLORS = {'space': ('#e8f2ed', '#38745a'), 'time': ('#e8f0fb', '#33669a'),
          'train': ('#fff1df', '#a8661c'), 'sample': ('#f1ebfa', '#7653a0'),
          'neutral': ('#f3f5f7', '#526372')}


def node(key, x, y, w, h, text, role='neutral'):
    return dict(id=key, x=x, y=y, width=w, height=h, text=text, role=role)


def edge(points, dashed=False):
    return dict(points=points, dashed=dashed)


def render(name, title, subtitle, nodes, edges, footer):
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.subplots_adjust(left=.025, right=.975, top=.97, bottom=.035)
    ax.set(xlim=(0, 100), ylim=(0, 105)); ax.axis('off')
    ax.text(50, 102, title, ha='center', va='top', fontsize=19, weight='bold', color='#213547')
    ax.text(50, 97.5, subtitle, ha='center', va='top', fontsize=10.5, color='#526372')
    for item in edges:
        pts = item['points']
        style = '--' if item['dashed'] else '-'
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#66788a', lw=1.4,
                    ls=style, zorder=1)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>',
                     mutation_scale=15, lw=1.4, color='#66788a', ls=style, zorder=1))
    texts = []
    boxes = []
    for n in nodes:
        fill, stroke = COLORS[n['role']]
        x, y, w, h = n['x'], n['y'], n['width'], n['height']
        patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle='round,pad=0.25,rounding_size=1.0',
                 linewidth=1.5, facecolor=fill, edgecolor=stroke, zorder=2)
        ax.add_patch(patch)
        texts.append(ax.text(x, y, n['text'], ha='center', va='center',
                     fontsize=11.5, linespacing=1.65, color='#213547', zorder=3))
        boxes.append(patch)
    ax.text(50, 2.0, footer, ha='center', va='bottom', fontsize=10, color='#526372', linespacing=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, b in zip(texts, boxes):
        tb, bb = t.get_window_extent(renderer), b.get_window_extent(renderer)
        assert bb.x0 <= tb.x0 and tb.x1 <= bb.x1, t.get_text()
        assert bb.y0 <= tb.y0 and tb.y1 <= bb.y1, t.get_text()
    for suffix in ('png', 'svg'):
        path = OUT / f'{name}.{suffix}'
        fig.savefig(path, dpi=180)
        if suffix == 'svg':
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_bytes(('\n'.join(line.rstrip() for line in lines)+'\n').encode('utf-8'))
    plt.close(fig)
    return dict(id=name, title=title, nodes=nodes, edges=edges,
                status='PROPOSED_ARCHITECTURE_NOT_TRAINING_RESULT', text_bounds='PASS')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.append(render('overview', '因果Transformer融合基础场与标准化残差生成',
        '设计方案 · 当前时刻空间重建R · 基础场训练完成后再训练残差模型', [
        node('input',50,90,90,7,'条件站连续历史ZWD/T/P＋mask/age\n地形、海岸距离与真实小时索引'),
        node('p14',25,78,42,9,'p14＋海岸／地形边消息\n113维点特征｜144×96支撑','space'),
        node('history',75,78,42,9,'逐小时16×16空间状态\n72小时 × 256位置 × 25通道','time'),
        node('spatial',25,65,42,9,'当前多尺度空间特征\nF0：16×144×96；F1：32×72×48','space'),
        node('temporal',75,65,42,9,'因果Transformer时间状态H\n2层、4头、宽度64；局地空间交互','time'),
        node('fuse',50,51,90,9,'两级残差空间解码器\n时间特征投影＋空间skip＋地形FiLM＋条件门控','time'),
        node('mean',50,38,90,8,'确定性基础场 μ(q,t) = μ_ref(q,t) + readout(g·δT,q)\n任意坐标读出；训练完成后冻结','space'),
        node('oof',25,25,42,9,'训练标签：空间5折交叉拟合\nr = y − μ_OOF → u = (r − b) / s','train'),
        node('sample',75,25,42,9,'采样：标准高斯潜噪声\n历史条件 → 因果残差diffusion','sample'),
        node('latent',25,11,42,9,'稀疏残差编码E／线性坐标解码D\n潜变量标准化 → AR创新标准化','train'),
        node('final',75,11,42,9,'AR递推＋潜变量／物理残差逆变换\n最终ZWD = μ + b + s·D(z_hat)','sample')], [
        edge([(25,86.5),(25,82.5)]), edge([(75,86.5),(75,82.5)]),
        edge([(25,73.5),(25,69.5)]), edge([(75,73.5),(75,69.5)]),
        edge([(25,60.5),(25,55.5)]), edge([(75,60.5),(75,55.5)]),
        edge([(50,46.5),(50,42)]), edge([(25,34),(25,29.5)]),
        edge([(75,34),(75,29.5)]), edge([(25,20.5),(25,15.5)]),
        edge([(75,20.5),(75,15.5)]), edge([(46.3,11),(50,11),(50,25),(53.7,25)], True),
        edge([(95.3,38),(98,38),(98,11),(96.3,11)])],
        '橙色分支只用于训练标签；采样仅使用允许条件、冻结参数和训练统计。'))

    figures.append(render('fusion_detail','两级空间解码器融合Transformer时间状态',
        '保留参考场 μ_ref = p14＋已有地形海岸修正；新增δT头零初始化', [
        node('f0',18,87,28,10,'p14高分辨率特征F0\n16×144×96','space'),
        node('ht',50,87,28,10,'Transformer状态H\n64×16×16','time'),
        node('f1',82,87,28,10,'p14低分辨率特征F1\n32×72×48','space'),
        node('t0',35,73,27,8,'时间投影T0\n16×144×96','time'),
        node('t1',65,73,27,8,'时间投影T1\n32×72×48','time'),
        node('a0',25,57,42,11,'高分辨率融合A0\nconcat[F0(16),T0(16),S0(8)]\n40→16通道；地形FiLM','space'),
        node('a1',75,57,42,11,'低分辨率融合A1\nconcat[F1(32),T1(32),S1(8)]\n72→32通道；地形FiLM','space'),
        node('decode',50,41,72,9,'空间skip融合：concat[A0, upsample(A1)]\n48→16通道的局地残差解码器','time'),
        node('delta',25,27,42,9,'零初始化1×1残差头\nδT：1×144×96','time'),
        node('gate',75,27,42,9,'条件门控g = sigmoid(Gate)\n当前支撑＋静态条件＋日历','neutral'),
        node('readout',50,12,90,9,'融合修正场 g·δT → 连续坐标读出\nμ(q,t) = μ_ref(q,t) + readout(g·δT,q)','space')], [
        edge([(46,82),(35,77)]), edge([(54,82),(65,77)]),
        edge([(35,69),(35,62.5)]), edge([(65,69),(65,62.5)]),
        edge([(18,82),(18,62.5)]), edge([(82,82),(82,62.5)]),
        edge([(25,51.5),(25,45.5)]), edge([(75,51.5),(75,45.5)]),
        edge([(25,36.5),(25,31.5)]), edge([(25,22.5),(25,16.5)]),
        edge([(75,22.5),(75,16.5)])],
        'S0/S1为对齐的8通道地形与海岸条件；所有尺度共享投影、像元中心和查询约定。'))

    figures.append(render('residual_pipeline','标准化残差场：训练标签路径与采样路径',
        '左侧构造完整扩散目标；右侧从高斯噪声生成，不读取目标真值', [
        node('r',25,84,42,11,'训练：冻结各空间折基础场\n隐藏站残差 r = y − μ_OOF\n记录fold／布局／权重哈希','train'),
        node('c',75,84,42,11,'采样：共享概率条件编码器G\n原始允许历史／支撑摘要、\n基础场物理输出、地形与海岸','space'),
        node('u',25,68,42,10,'物理残差标准化（mm）\nu = (r − b) / s\nb/s仅由允许训练残差拟合','train'),
        node('noise',75,68,42,10,'每帧创新的独立初始噪声\nv_K ~ N(0,I)\n128维；同一成员延续AR状态','sample'),
        node('ae',25,52,42,10,'逐帧编码z=E(u)，w=(z−bz)/sz\nAR创新 e=w−rho·w_prev\n创新标准化 v=(e−be)/se','train'),
        node('denoise',75,52,42,10,'逐帧创新Transformer去噪\n因果条件memory＋start标记\n扩散步k独立编码；DDIM20','sample'),
        node('eps',25,36,42,10,'对完整创新标签v加噪\nv_k = sqrt(abar)·v + sqrt(1-abar)·ε\nε ~ N(0,I)','train'),
        node('inverse',75,36,42,10,'AR递推 w=rho·w_prev+be+se·v_hat\nz=bz+sz·w；u=Φ(q,c)·z\n连续残差 r_hat=b+s·u','sample'),
        node('loss',25,20,42,10,'训练ε预测器：MSE(eps_hat, ε)\n双种子、固定3000步\n过去标签仅参与创新标签变换','train'),
        node('zwd',75,20,42,10,'最终场与集合评价\nZWD_hat = 冻结μ + r_hat\nK64均值／区间／时空联合评分','sample')], [
        edge([(25,78.5),(25,73)]), edge([(25,63),(25,57)]),
        edge([(25,47),(25,41)]), edge([(25,31),(25,25)]),
        edge([(96.3,84),(98,84),(98,52),(96.3,52)]), edge([(75,63),(75,57)]),
        edge([(75,47),(75,41)]), edge([(75,31),(75,25)]),
        edge([(46.3,20),(50,20),(50,52),(53.7,52)], True)],
        '采样条件从右侧首节点传入去噪器和Φ；rho、be/se、b/s、bz/sz与E/D/G均冻结。\n首轮6小时块独立且start重置；随机时间依赖来自AR，帧内创新由diffusion学习。'))
    manifest = dict(status='DESIGN_DIAGRAMS_NOT_EXPERIMENT_RESULTS', figures=figures,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(OUT.iterdir()) if p.suffix in ('.png','.svg')})
    (OUT/'figure_manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'figures':len(figures),'exports':len(manifest['outputs']),
                      'text_bounds':'PASS','out':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
