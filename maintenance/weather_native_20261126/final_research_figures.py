"""Render existing verified results only; no fitting or new evaluation dates."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results'
OUT = ROOT/'docs/figures/final_research_20261126'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=170, facecolor='white')
    svg=OUT/f'{name}.svg'
    fig.savefig(svg, facecolor='white')
    # Matplotlib leaves spaces at SVG path line ends; retain geometry verbatim.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.set(xlim=(0, 17), ylim=(0, 12.5)); ax.axis('off')
    ax.text(.35, 12.05, '当前最佳架构与已验证分支', fontsize=23, weight='bold', color='#142b46')
    ax.text(.35, 11.58, '2016–2019连续训练 · 2020两组抽样开发日历 · 当前重建 / 六小时预报 / 概率预测分别选择', fontsize=12, color='#46576b')

    def box(x, y, w, h, title, content, color='#eaf1f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',facecolor=color,edgecolor='#9aabba',linewidth=1))
        ax.text(x+.18,y+h-.30,title,fontsize=13,weight='bold',va='top',color='#142b46')
        ax.text(x+.18,y+h-.75,content,fontsize=11,va='top',linespacing=1.65,color='#27384c')

    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,color='#526a80',linewidth=1.5))

    box(.35,8.55,4.6,2.35,'① 原始信息与冻结背景',
        'GNSS ZWD / T / P；mask与age\n地形、高程、海岸距离、坐标\np14点特征MLP + 小UNet16\n原地形/海岸分支 + GRU72')
    box(5.55,8.55,5.2,2.35,'② I1：历史观测创新校正',
        '条件站观测 − 同位置背景预测\n0 / 1 / 3h × 50 / 150 / 300km软核\n3km高差输送；9维岭回归\n修正限幅 ±5mm', '#e5f3ed')
    box(11.4,8.55,5.15,2.35,'当前场推荐：I1',
        '主 / 补 RMSE：9.817 / 10.740mm\n合并72抽样日：10.320mm\njoint主当前9.806mm，合并10.327mm\n合并当前差异区间跨零', '#e5f3ed')
    arrow(5.05,9.72,5.42,9.72); arrow(10.85,9.72,11.27,9.72)

    box(.35,4.55,4.6,3.2,'③ 保留逐站72h状态',
        '动态15 + 静态9 + ID嵌入8 = 32维\n含原始观测、两类创新及日/年时钟\n32→64；因果Transformer 2层\n4头；FF128；保留每个历史key\n72h状态依赖146h原始数值')
    box(5.55,4.55,5.2,3.2,'④ 地形/海岸条件有向时滞传递',
        '查询111→64；四尺度空间注意力\n历史key：0 / 1 / 3 / 6 / 12 / 24h\n地理8 + lag + lead → MLP10→32→4\n创新、ΔZWD、ΔT、ΔP独立mask归一\n物理创新读出 + 历史增量辅助监督\n新路径零读出；当前/未来限幅3/6mm')
    box(11.4,4.55,5.15,3.2,'六小时预报推荐：joint v2',
        'μh = μI1(t) + δH(q,h) + δdirected(q,h)\n主 / 补 F6：13.852 / 16.078mm\n合并F6：15.054mm\n相对H_delta合并减少0.126mm\n新校正载体117757参数\n不含冻结背景与I1来源参数', '#e5f3ed')
    arrow(2.65,8.42,2.65,7.88); arrow(5.05,6.1,5.42,6.1); arrow(10.85,6.1,11.27,6.1)
    ax.text(8.5,4.03,'以下概率分支来源于原TF3000；早期含S175概率中心校准，均未接入新joint v2',ha='center',fontsize=12,color='#a15025',weight='bold')

    box(.35,.8,4.6,2.65,'⑤ 概率来源与残差',
        'TF3000点场：R9.818 / F6 14.127\n早期S175校准中心；后期严格固定均值\n训练尺度标准化；strict_oof=false\nS175尺度 + G265天气相关GP\n查询坐标解码，保留完整剩余过程', '#f1edf8')
    box(5.55,.8,5.2,2.65,'⑥ 已验证概率 / 扩散对照',
        'GP预条件v扩散：Heun / DDIM\n完整GFT×DCT与频率token对照\n共享诱导创新：Affine / odd非线性\n末轮：天气尺度G与同协方差Student-t', '#f1edf8')
    box(11.4,.8,5.15,2.65,'概率选择按指标区分',
        '末轮Affine time：F6 CRPS 7.810mm\n同协方差t相对天气G改善0.01129mm\n联合NLL/d：2.98435 → 2.97195\n天气t相对常数t的额外收益未成立', '#f1edf8')
    arrow(5.05,2.1,5.42,2.1); arrow(10.85,2.1,11.27,2.1)
    ax.text(.35,.22,'内部固定支撑图保留区域上下文；任意坐标查询不受输出格网绑定。上述为已实测分支，不表示已实现无损编码或统一最优生成模型。',fontsize=11,color='#46576b')
    fig.subplots_adjust(left=.015,right=.99,bottom=.015,top=.99)
    save(fig,'best_architecture')


def performance(point, tail):
    fig, axes = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    labels={'I1_persistence':'I1 / persistence','H_delta':'H_delta','factorized':'factorized','joint':'joint v2'}
    colors=['#697c91','#e49b56','#6c85bc','#148b79']
    for r,c in zip(point['rows'],colors):
        axes[0].plot(range(7),r['macro_rmse_mm'],'o-',label=labels[r['arm']],color=c)
    axes[0].set(title='合并72抽样日：预报时效',xlabel='Lead (h)',ylabel='站宏RMSE (mm)',xticks=range(7));axes[0].legend(fontsize=9)
    rows=point['rows']; x=np.arange(4)
    axes[1].bar(x-.18,[r['main_R_F6'][0] for r in rows],.36,label='主日历8–10日',color='#697c91')
    axes[1].bar(x+.18,[r['supplement_R_F6'][0] for r in rows],.36,label='补充15–17日',color='#58a89b')
    axes[1].set(title='当前重建：两组日期分别核验',ylabel='站宏RMSE (mm)',xticks=x,xticklabels=['I1','H_delta','factorized','joint'],ylim=(9.5,11));axes[1].legend(fontsize=9)
    variants=['fixed','G_weather','G_extra','T_constant','T_weather']
    vals=lambda key:[np.mean([r['records'][key]['value'] for r in tail['rows'] if r['name']==n]) for n in variants]
    axes[2].barh(range(5),vals('crps_h6'),color=['#148b79','#a0afc1','#a0afc1','#ae92c6','#ae92c6'])
    axes[2].set(title='末轮六臂中的F6概率评分',xlabel='fair CRPS (mm), lower is better',yticks=range(5),yticklabels=['Affine time','天气G','额外预算G','常数t','天气t'],xlim=(7.78,7.86));axes[2].invert_yaxis()
    for i,v in enumerate(vals('crps_h6')):axes[2].text(v+.0008,i,f'{v:.4f}',va='center',fontsize=9)
    for ax in axes:ax.grid(alpha=.18,axis='y');ax.set_axisbelow(True)
    fig.suptitle('当前精度总览｜全部为开发暴露的2020抽样日期；两种子等权；概率分支沿用原TF3000',fontsize=14)
    save(fig,'performance_overview')


def curves():
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,seed in zip(axes,(42,17)):
        for arm,c in zip(('G_weather','G_extra','T_weather'),('#697c91','#dd9c53','#148b79')):
            rows=read(RESULTS/f'20261126_weather_scale_tail/{arm}_s{seed}/curve.json')
            ax.plot([r['step'] for r in rows],[r['nll_per_dimension'] for r in rows],'.-',label=arm,color=c,alpha=.85)
        ax.set(title=f'seed{seed}：连续块训练记录',xlabel='Consumed blocks',ylabel='Block NLL / dimension');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('最终天气尺度 / 尾部训练｜各臂3000连续块；天气G冻结后分别拟合额外G与t；不同块天气导致波动',fontsize=12)
    save(fig,'final_training_curves')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'path'})
    point=read(RESULTS/'20261126_final_research_summary/point_metrics.json')
    tail=read(RESULTS/'20261126_weather_scale_tail/paired/summary.json')
    architecture();performance(point,tail);curves()
    files=[RESULTS/'20261126_final_research_summary/point_metrics.json',RESULTS/'20261126_weather_scale_tail/paired/summary.json',Path(__file__)]
    manifest={'purpose':'render existing locked results; no new fitting or dates','sources':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    manifest['figures']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.png','.svg')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered 3 PNG/SVG pairs and source manifest')


if __name__=='__main__':main()
