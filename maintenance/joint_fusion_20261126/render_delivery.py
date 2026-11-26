"""Render the measured joint/diffusion fusion architecture, curves and report."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_best_base_diffusion_fusion'
FIG=ROOT/'docs/figures/best_base_diffusion_fusion_20261126'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=170,facecolor='white')
    p=FIG/f'{name}.svg';fig.savefig(p,facecolor='white')
    p.write_text('\n'.join(l.rstrip() for l in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8');plt.close(fig)


def architecture(summary):
    fig,ax=plt.subplots(figsize=(17,10.6));ax.set(xlim=(0,17),ylim=(0,10.6));ax.axis('off')
    ax.text(.35,10.1,'最佳基础场＋天气条件残差扩散',fontsize=24,weight='bold',color='#142b46')
    ax.text(.35,9.62,'冻结joint v2 · 新拟合尺度/相关性/扩散 · 逐站72h历史 · 联合生成当前至未来6h',fontsize=12,color='#4d6277')
    def box(x,y,w,h,title,body,c='#eaf1f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',facecolor=c,edgecolor='#9aabba'))
        ax.text(x+.15,y+h-.27,title,fontsize=13,weight='bold',va='top',color='#142b46')
        ax.text(x+.15,y+h-.72,body,fontsize=11,va='top',linespacing=1.65,color='#27384c')
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#526a80',linewidth=1.5))
    base=np.mean([r['point_R_F6_mm'] for r in summary['rows'] if r['day']==8],0)
    box(.35,6.65,4.75,2.3,'① 已验证空间与历史背景','p14点特征 + UNet16 / GRU72\n地形、高程、海岸距离\nI1历史创新校正\nGNSS / T / P、mask / age、站点身份')
    box(5.65,6.65,5.0,2.3,'② 冻结joint v2均值载体','2层4头逐站因果Transformer\n地形/海岸查询与有向时滞传递\nlag 0/1/3/6/12/24h，lead 0–6h\n重新导出条件64维与七小时均值')
    box(11.25,6.65,5.15,2.3,'独立确定性点输出',f'μjoint(q,h) 保持原权重\n主当前 {base[0]:.3f}mm / F6 {base[1]:.3f}mm\n所有概率候选逐值保持该点输出\n不以有限成员均值替换基础场','#e5f3ed')
    arrow((5.2,7.8),(5.5,7.8));arrow((10.75,7.8),(11.1,7.8))
    box(.35,3.45,4.75,2.35,'③ 新残差与Gaussian边际','训练残差 e=(y−μjoint−b)/s\nb、s仅用2016–2019训练统计\nc175：条件64 + 原查询111\nS175新拟合位置mG与尺度σG')
    box(5.65,3.45,5.0,2.35,'④ G265天气相关过程','c265 = c175 + 90维历史天气摘要\n新训练copula与完整正定相关核R\n完整场初始噪声 uGP=LR ε\n每查询7×7主块Aq白化/反变换')
    box(11.25,3.45,5.15,2.35,'⑤ 残差轨迹扩散与强对照','Local非线性v扩散 / Affine对照\n同起点、条件、噪声流与3000块\nHeun0：20步、39次网络计算\n联合生成7h，不逐小时自回归')
    arrow((7.95,6.5),(2.7,5.95));arrow((5.2,4.6),(5.5,4.6));arrow((10.75,4.6),(11.1,4.6))
    box(.35,.45,7.65,2.1,'⑥ 原始生成分布：K32 IID','Z = μjoint + b + s[mG + σG·udiff]\nGaussian残差中心、扩散均值漂移与点输出分别报告\n评估单站/站对/时间增量CRPS、energy、事件和覆盖','#f1edf8')
    box(8.65,.45,7.75,2.1,'⑦ 均值保持融合：K16独立成员对','Zpair = μjoint + (Z2k − Z2k+1)/√2\n总体均值=μjoint，协方差保持；偏态被对称化\n不重叠配对；Gaussian/Affine/Local同规则对照','#f1edf8')
    arrow((13.7,3.3),(4.2,2.7));arrow((13.7,3.3),(12.4,2.7))
    fig.subplots_adjust(left=.015,right=.99,bottom=.015,top=.99);save(fig,'architecture')


def curves():
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for ax,mode in zip(axes.flat,('S175','G265','affine','local')):
        for seed,c in ((42,'#168878'),(17,'#697eb8')):
            r=read(RESULT/f's{seed}/{mode}/curve.json');key='loss'
            ax.plot([v['step'] for v in r],[v[key] for v in r],'.-',label=f'seed{seed}',color=c)
        ax.set(title=mode,xlabel='Consumed continuous blocks',ylabel='Recorded block objective');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('各臂3000块完整训练曲线｜不同连续天气块的训练目标，非固定开发集收敛曲线')
    save(fig,'training_curves')


def performance(summary):
    rows=summary['rows'];get=lambda day,name,key:np.mean([r['records'][key]['value'] for r in rows if r['day']==day and r['name']==name])
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained');names=('G265','affine','local')
    for ax,(key,title) in zip(axes,[('crps_h6','F6单站CRPS / mm'),('pair_0-50_h6','近邻F6差值CRPS / mm'),('increment','小时增量CRPS / mm')]):
        x=np.arange(3)
        for offset,day,c in ((-.18,8,'#697eb8'),(.18,15,'#168878')):
            ax.bar(x+offset,[get(day,n,key) for n in names],.36,label=f'{day}日历',color=c)
        ax.set(title=title,xticks=x,xticklabels=names);ax.legend();ax.grid(alpha=.2,axis='y');ax.set_axisbelow(True)
    fig.suptitle('最佳joint基础场上的新概率模型｜K32原始生成，越低越好；两日历均为开发')
    save(fig,'performance')


def report(summary):
    path=ROOT/'docs/research_notes/2026-11-26-best-base-diffusion-fusion.md'
    lines=['# 最佳基础场与扩散融合：实测架构与结果','',
        '本实验把已验证的joint v2/last载体冻结，重新拟合其训练残差上的S175、G265及Affine/Local扩散。两种子各四臂3000连续块完成；点输出与原joint v2数组逐值一致。原始K32生成与独立双场差分K16均完整评价两组2020开发日期。','',
        '![完整融合架构](../figures/best_base_diffusion_fusion_20261126/architecture.png)','',
        '## 点预测与生成场怎样连接','',
        '确定性均值使用p14/I1背景、逐站因果Transformer和地形/海岸条件有向时滞传递。新概率分支接收其64维查询状态、111维原始查询条件和90维固定可见站历史天气摘要。72h状态依赖146h原始数值，输入截至预测起点。','',
        '设冻结点场为μjoint，训练残差均值/尺度为b、s，先标准化e=(y−μjoint−b)/s。S175学习条件残差位置mG与尺度σG；G265学习完整时空相关矩阵R。用同一Gaussian场作为Affine与Local的初值，逐查询7×7轨迹块白化后执行v扩散/仿射流，再恢复物理mm。','',
        '原始输出为Z=μjoint+b+s[mG+σG·udiff]。μjoint仍作为独立点输出，概率中心可以因S175及扩散而变化。中心保持输出用两个独立完整场之差：Zpair=μjoint+(Z−Z′)/√2，其总体均值严格等于μjoint、协方差等于原Z；偏态被对称化。它不等于将同一成员集合减样本均值，实际K16成员均值仍有采样方差。','',
        '## 完整评分与额外收益','']
    for day in (8,15):
        for edge in ('local_vs_affine','paired_local_vs_paired_affine'):
            r=summary['comparisons'][f'day{day}/{edge}']['crps_h6'];ci=r['delta_ci95']
            lines.append(f"- 日历{day}，{edge}：F6 CRPS差值{r['delta']:.6f}mm，95%共同月份区间[{ci[0]:.6f},{ci[1]:.6f}]mm；负值表示Local改善。")
    rel='../../research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results/20261126_best_base_diffusion_fusion/paired/'
    lines+=['',f'[完整数值、点预测与采样均值表]({rel}REPORT.md) · [全部区间与分层JSON]({rel}summary.json) · [覆盖和重点站]({rel}DIAGNOSTICS.md)','',
        '![概率性能](../figures/best_base_diffusion_fusion_20261126/performance.png)','',
        '![配对收益与区间]('+rel+'fusion_effects.png)','',
        '## 训练与核验','',
        '![完整训练曲线](../figures/best_base_diffusion_fusion_20261126/training_curves.png)','',
        '训练曲线展示每100块记录的当前块目标；连续天气块分布变化引起波动，不能把最后一个点与第一个点直接解释为固定总体的收敛。均值保持、来源、真实监督时钟及K16/K32秩覆盖测试均有RED→GREEN记录。每个正式模型核对最终checkpoint与last逐张量一致，优化器/scheduler时钟由原始标签掩码重算。','',
        '## 数据范围与复现','',
        '训练2016–2019、410站训练池；连续3000×12h块，末91块顺序重放，历史不跨回绕拼接。2020每月8–10日及15–17日、三布局、137注册查询（R/F6实际135站有标签）均为已暴露开发，未作多重探索校正。严格折外残差尚未构建：strict_oof=false。历史PWV制包筛选、原Q128未来可用性选站限制保留，确认138及原C/D未读。','',
        '新模型不使用PWV/IWV/TCWV/Tm条件或教师。原始数据、权重、大数组留本机；Git同步代码、协议、指标、图和哈希。本实验的概率成绩来自重新拟合的新joint残差，不与旧TF3000概率成绩混用。','',
        '[冻结科学合同](../superpowers/specs/2026-11-26-best-base-diffusion-fusion.md) · [实施计划](../superpowers/plans/2026-11-26-best-base-diffusion-fusion.md) · [逐步执行与纠错记录](../../maintenance/joint_fusion_20261126/progress.md)']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    FIG.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'path'})
    summary=read(RESULT/'paired/summary.json');read(RESULT/'paired/diagnostics.json')
    architecture(summary);curves();performance(summary);report(summary)
    effect=RESULT/'paired/fusion_effects.svg'
    effect.write_text('\n'.join(l.rstrip() for l in effect.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    sources=[RESULT/'paired/summary.json',RESULT/'paired/diagnostics.json',Path(__file__),
        *[RESULT/f's{seed}/{mode}/curve.json' for seed in (42,17) for mode in ('S175','G265','affine','local')]]
    manifest=dict(sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},figures={p.name:sha(p) for p in FIG.iterdir() if p.suffix in ('.png','.svg')})
    (FIG/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Rendered fusion architecture, curves, performance and report',flush=True)


if __name__=='__main__':main()
