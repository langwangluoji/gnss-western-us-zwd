"""Scientific figures and a Chinese report from frozen numerical artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .data import HERE,CACHE,write_json,sha

BLUE='#176B9B';ORANGE='#D77636';INK='#213548';GREEN='#26856A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','savefig.facecolor':'white'})
FIG=HERE/'figures'


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def coast(ax):
    payload=json.loads((CACHE/'ne_50m_coastline.geojson').read_text(encoding='utf-8'))
    for feature in payload['features']:
        geo=feature['geometry'];lines=[geo['coordinates']] if geo['type']=='LineString' else geo['coordinates']
        for line in lines:
            p=np.asarray(line)
            if ((p[:,0]>-20)&(p[:,0]<40)&(p[:,1]>25)&(p[:,1]<75)).any():ax.plot(p[:,0],p[:,1],color='#C7CDD3',lw=.45,zorder=0)
    ax.set(xlim=(-17,37),ylim=(29,72),xlabel='Longitude (degrees)',ylabel='Latitude (degrees)')
    ax.grid(alpha=.12)


def station_map():
    d=pd.read_csv(HERE/'stations.csv');fig,ax=plt.subplots(figsize=(9,7));coast(ax)
    for role,color,marker in [('train',BLUE,'o'),('dev',ORANGE,'s'),('confirm','#AB3B4C','^'),('new2024',GREEN,'D')]:
        a=d[d.role==role];ax.scatter(a.Longitude,a.Latitude,s=20 if role=='train' else 35,c=color,marker=marker,edgecolors='white',linewidths=.35,label=f'{role}: {len(a)}',zorder=2)
    ax.legend(loc='upper left',framealpha=.95);ax.set_title('Europe transfer | frozen station-site split\n1 km connected site groups stay in one role',loc='left',pad=12)
    save(fig,'01_station_split')


def architecture():
    fig,ax=plt.subplots(figsize=(14,7.8));ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,title,detail,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.12',facecolor=color,edgecolor='none'))
        ax.text(x+.15,y+h-.28,title,weight='bold',fontsize=11,color=INK,va='top')
        ax.text(x+.15,y+h-.76,detail,fontsize=9.5,color=INK,va='top',linespacing=1.55)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#597187'))
    box(.1,6,13.6,1.45,'Shared data and frozen evaluation','Train pool: 2020-2023 | 273 training stations | 64 dev / 63 confirm / 11 new station IDs\n6 h UTC records; genuine gaps; 16-state history spans 90 h | final comparison: common 2024 station-UTC keys','#ECF0F3')
    box(.2,2.7,2.6,2.65,'Our allowed conditions','Reference GNSS ZWD / T / P\nDry-air pressure correction\nMask / age / station identity\nCoordinates, height, terrain\nNatural Earth coast distance','#E3F0F5')
    box(3.25,3.4,3.1,1.95,'p14 + GRU carrier','113-D point summaries\n30-channel fixed support U-Net16\n8 x 8 multiscale history + GRU\nContinuous-coordinate readout','#D9EAF1')
    box(6.85,3.4,2.65,1.95,'I1 innovation head','Reference model errors\n3 spatial scales x 3 lags\n9-D ridge correction\nEuropean residual fitting','#DCEEE6')
    box(9.95,2.7,3.55,2.65,'Native joint candidate','Per-station causal Transformer\n2 layers / 4 heads / width 64\nTerrain + coast query attention\nDirected station x lag transport\n7 outputs: 0, 6, ..., 36 h','#DEE9F6')
    arrow((2.9,4.3),(3.15,4.3));arrow((6.45,4.3),(6.75,4.3));arrow((9.6,4.3),(9.85,4.3))
    box(.2,.2,6.15,1.85,'Original Model4 system','Same training station IDs and 2020-2023 pool\nOriginal FFT-SCHA + CNN-BiLSTM-Attention + XGBoost\nOwn humidity / meteorology / target-station past inputs','#F7EBDD')
    box(6.85,.2,6.65,1.85,'Freeze on 2023 dev sites; evaluate once on 2024','Compare p14 / I1 / joint on spatially held-out dev sites\nBoth systems: seeds42 and17; final single-seed and ensemble RMSE\nDiffusion is outside this point-RMSE transfer experiment','#EDF1F5')
    arrow((11.7,2.6),(11.7,2.15));arrow((6.45,1.1),(6.75,1.1))
    ax.text(.1,7.85,'EUROPE | architecture migration and system comparison',fontsize=17,weight='bold')
    save(fig,'02_architecture')


def learning_curves():
    curves={};fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    for seed,color in [(42,BLUE),(17,ORANGE)]:
        for kind in ('base_v2','joint_v2'):
            path=HERE/f'runs/{kind}_s{seed}/curve.jsonl';rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()];curves[f'{kind}_s{seed}']=rows
            if kind=='base_v2':
                selected=[r for r in rows if 'dev_station_macro_rmse_mm' in r]
                axes[0].plot([r['step'] for r in selected],[r['dev_station_macro_rmse_mm'] for r in selected],'-o',color=color,label=f'seed{seed}')
            else:
                selected=[r for r in rows if 'dev_current_rmse_mm' in r]
                axes[1].plot([r['step'] for r in selected],[r['dev_current_rmse_mm'] for r in selected],'-o',color=color,label=f'current, seed{seed}')
                axes[1].plot([r['step'] for r in selected],[r['dev_f6_rmse_mm'] for r in selected],'--',color=color,label=f'+6 h, seed{seed}')
        rows=json.loads((HERE/f'runs/original_s{seed}/learning_curve.json').read_text(encoding='utf-8'));curves[f'original_s{seed}']=rows
        axes[2].plot([r['epoch'] for r in rows],[np.sqrt(r['val_loss'])*1000 for r in rows],color=color,label=f'seed{seed}')
    for ax,title in zip(axes,['p14 + GRU | 2023 dev','Native joint | 2023 dev','Original CNN-LSTM | 2023 monitor']):
        ax.set_title(title,loc='left');ax.grid(alpha=.18);ax.legend(fontsize=8)
    for ax in axes[:2]:ax.set(xlabel='Consecutive-block optimizer updates',ylabel='Station macro RMSE (mm)')
    axes[2].set(xlabel='Epoch',ylabel='sqrt(weighted validation loss) (mm)')
    fig.tight_layout();save(fig,'03_training_curves');write_json(HERE/'curves.json',curves)


def results():
    summary=json.loads((HERE/'comparison.json').read_text(encoding='utf-8'));main=summary['cohorts']['confirm'];models=main['models'];chosen=summary['selection']
    fig,ax=plt.subplots(figsize=(9,4.7));x=np.arange(3);old=[models[k]['micro_rmse_mm'] for k in ['original_s42','original_s17','original_ensemble']];new=[models[k]['micro_rmse_mm'] for k in [f'{chosen}_s42',f'{chosen}_s17','ours_ensemble']]
    ax.bar(x-.18,old,.34,color=ORANGE,label='Original CNN-LSTM-XGBoost');ax.bar(x+.18,new,.34,color=BLUE,label=f'Our {chosen}')
    for positions,values in [(x-.18,old),(x+.18,new)]:
        for p,v in zip(positions,values):ax.text(p,v+.15,f'{v:.3f}',ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=['Seed42','Seed17','Two-seed ensemble'],ylabel='Common-key RMSE (mm)',ylim=(0,max(old+new)*1.22));ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.15)
    ax.set_title(f'2024 | held-out confirm sites\n{main["stations"]} active stations; {main["records"]:,} shared station-UTC records',loc='left',pad=12)
    fig.tight_layout();save(fig,'04_rmse_comparison')
    monthly=pd.read_csv(HERE/'monthly_rmse.csv');fig,ax=plt.subplots(figsize=(10,4.1))
    for name,color,label in [('original_ensemble',ORANGE,'Original ensemble'),('ours_ensemble',BLUE,f'Our {chosen} ensemble')]:
        d=monthly[monthly.model==name];ax.plot(d.month,d.micro_rmse_mm,'-o',color=color,label=label)
    ax.set(xticks=np.arange(1,13),xlabel='Month in2024',ylabel='Common-key RMSE (mm)',title='2024 confirm sites | seasonal error');ax.legend();ax.grid(alpha=.2);fig.tight_layout();save(fig,'05_monthly_rmse')
    stations=pd.read_csv(HERE/'station_rmse.csv');fig,axes=plt.subplots(1,3,figsize=(15,5.6),sharex=True,sharey=True)
    maximum=float(np.quantile(np.r_[stations.original_ensemble,stations.ours_ensemble],.95));m=None
    for ax,name,title in zip(axes[:2],['original_ensemble','ours_ensemble'],['Original system','Our selected field']):
        coast(ax);m=ax.scatter(stations.Longitude,stations.Latitude,c=stations[name],s=23,cmap='viridis',vmin=0,vmax=maximum,edgecolors='white',linewidths=.2);ax.set_title(title,loc='left')
    fig.colorbar(m,ax=axes[:2],fraction=.02,pad=.02,label='Station RMSE (mm)')
    coast(axes[2]);delta=stations.ours_ensemble-stations.original_ensemble;lim=max(float(np.quantile(np.abs(delta),.95)),1)
    m=axes[2].scatter(stations.Longitude,stations.Latitude,c=delta,s=23,cmap='RdBu_r',vmin=-lim,vmax=lim,edgecolors='white',linewidths=.2);axes[2].set_title('Our - original (negative = better)',loc='left');fig.colorbar(m,ax=axes[2],fraction=.04,pad=.02,label='RMSE difference (mm)')
    fig.suptitle('2024 spatially held-out sites | identical valid UTC records',x=.1,ha='left',weight='bold');save(fig,'06_station_error_maps')
    return summary


def report(summary):
    selected=summary['selection'];cohorts=summary['cohorts'];rows=[]
    for key,title in [('confirm','最终留站'),('dev','开发站年度回看'),('new_unseen_site','新站且站址未见'),('all_spatial_holdout','全部空间留出')]:
        c=cohorts[key];m=c['models'];old=m['original_ensemble']['micro_rmse_mm'];new=m['ours_ensemble']['micro_rmse_mm'];rows.append(f'| {title} | {c["stations"]} | {c["records"]:,} | {old:.4f} | {new:.4f} | {new-old:+.4f} |')
    seeds=[]
    for seed in (42,17):
        m=cohorts['confirm']['models'];seeds.append(f'| {seed} | {m[f"original_s{seed}"]["micro_rmse_mm"]:.4f} | {m[f"{selected}_s{seed}"]["micro_rmse_mm"]:.4f} |')
    selection=json.loads((HERE/'selection_2023_v2.json').read_text(encoding='utf-8'))
    scores='、'.join(f'{k}={v:.4f}mm' for k,v in selection['mean_scores_mm'].items())
    text=f'''# 欧洲数据上的架构迁移与最终RMSE

正式v2在Model4欧洲数据上完成2020–2023训练池迁移，并在2024完整年度的共同有效站–UTC记录上比较原CNN-LSTM-XGBoost。2023开发站选择的点输出架构为 **{selected}**；两系统均报告seed42、seed17及两种子等权平均。最终留站ensemble RMSE差值（本模型−原模型）为{cohorts['confirm']['models']['ours_ensemble']['micro_rmse_mm']-cohorts['confirm']['models']['original_ensemble']['micro_rmse_mm']:+.4f}mm。

![最终RMSE](figures/04_rmse_comparison.png)

## 数据与站点划分

400个训练期站和385个2024站合并为411站，374站跨年共享。按1km连通物理站址分组，再按坐标空间分层划分273训练、64开发、63最终留出站；另有11个2024新站。训练、开发和最终留站之间没有1km物理组交叉；新站如果与训练站同址，从“新站未见站址”指标中剔除。站点清单见[stations.csv](stations.csv)，源文件SHA和单位审计见[data_audit.json](data_audit.json)。

![划分地图](figures/01_station_split.png)

输入是实际6h UTC记录，保留缺测，不造小时值。16记录历史跨度90h，7个输出对应0/6/12/18/24/30/36h。本报告主指标为当前ZWD的RMSE，单位mm。2024已在历史研究中暴露，本轮属于年度回看。共同键规则同时要求两系统、两个种子和标签有效；每站断测引起的旧模型连续序列缺失在共同覆盖率中如实体现。

## 网络和特征

本模型复用实际p14固定载体U-Net16、113维多尺度点读出、原8×8 GRU历史分支及海岸/地形边分支；随后拟合9维I1历史创新岭头，并训练实际NativeDirectedLagV2因果Transformer候选。逐站15维状态包含GNSS/T/P、mask、age、模型误差和时钟，身份嵌入独立保留，查询注意力用方向、高差、地形和海岸关系传递信息；新增校正头零初始化。

![全流程架构](figures/02_architecture.png)

新模型只使用参考站GNSS ZWD、T、P及静态几何/地形/海岸和时间条件。P由ERA5原始气压、T、高差进行干空气高程修正；不读原流程WVP、TCWV、FFT-SH及VMF3特征。欧洲地形采用站点ERA5静态地形的32×32粗载体插值，梯度和粗糙度从该粗场计算；不是美国SRTM地形的直接复制。海岸距离来自Natural Earth1:50m海岸线EPSG3035投影线段距离。

原模型按相同273训练站重新拟合原FFT-SCHA、CNN-BiLSTM-Attention和XGBoost，保留其原始湿度、气象和目标站历史特征。FFT-SCHA使用2020–2023；CNN-LSTM按原流程2020–2022拟合、2023早停；XGBoost在2023年前9个月拟合/Q4早停后按全年重拟合。本模型正式v2基础场、I1和joint使用2020–2023训练站池，2023监控是独立站址开发监控。两者训练站和总体年份一致，各自特征及训练阶段不同，因此结果是系统效果比较。

空间场在留出目标的整个站址历史上均不读GNSS；原模型允许读该目标站严格早于当前的历史及其气象湿度。两者最终预测的是同一份Model4 ZWD标签，标签和行筛选继承原制包谱系。以上信息条件应与RMSE同时解释。

## 训练与冻结选择

正式v2 p14+GRU、joint各两个种子、每种子3000连续四记录块，时间按顺序遍历，仅条件站动态轮换。训练标签来自训练站，开发/确认站标签不用于梯度和I1拟合。2023固定24个四记录开发块的两种子平均站宏RMSE：{scores}。根据此结果在读取2024评价指标前冻结{selected}，见[selection_2023_v2.json](selection_2023_v2.json)。三年神经阶段v1保留为先行诊断，不与正式四年版混排。

![训练曲线](figures/03_training_curves.png)

原CNN-LSTM保留50epoch上限、8epoch早停，XGBoost保留3000树上限和原默认参数；本模型使用固定3000块研究预算。计算预算不是相同更新数或相同样本数。曲线中的原模型损失为sigma加权损失，其平方根与最终无权RMSE分别标注。

## 最终RMSE

主表采用两种子等权平均，micro RMSE为所有共同有效记录的误差平方平均后开方；站宏RMSE及所有中间模型数值另见[comparison.json](comparison.json)。

| 评价人口 | 有效站 | 共同记录 | 原模型RMSE/mm | 本模型RMSE/mm | 本模型−原模型/mm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

最终留站单种子结果：

| 种子 | 原模型RMSE/mm | 本模型RMSE/mm |
|---|---:|---:|
{chr(10).join(seeds)}

![月份RMSE](figures/05_monthly_rmse.png)

![逐站误差](figures/06_station_error_maps.png)

逐站及月份原始指标分别见[station_rmse.csv](station_rmse.csv)、[monthly_rmse.csv](monthly_rmse.csv)。当前评价输出为确定性点场；美国概率扩散分支的权重未混入欧洲模型。本轮按用户指定的最终RMSE对比完成，概率分支可在欧洲新残差上独立拟合与评价。

## 复现和产物

Anaconda环境：`D:/anaconda3-2025.12-2/envs/pytorch/python.exe`。实际命令：

```powershell
python -m maintenance.europe_transfer_20261126.data
python -m maintenance.europe_transfer_20261126.train_base --seed 42 --steps 3000 --version 2
python -m maintenance.europe_transfer_20261126.train_base --seed 17 --steps 3000 --version 2
python -m maintenance.europe_transfer_20261126.native --seed 42 --prepare --version 2
python -m maintenance.europe_transfer_20261126.native --seed 17 --prepare --version 2
python -m maintenance.europe_transfer_20261126.native --seed 42 --steps 3000 --version 2
python -m maintenance.europe_transfer_20261126.native --seed 17 --steps 3000 --version 2
python -m maintenance.europe_transfer_20261126.legacy --prepare-only
python -m maintenance.europe_transfer_20261126.legacy --seed 42
python -m maintenance.europe_transfer_20261126.legacy --seed 17
python -m maintenance.europe_transfer_20261126.evaluate select
python -m maintenance.europe_transfer_20261126.evaluate predict --seed 42
python -m maintenance.europe_transfer_20261126.evaluate predict --seed 17
python -m maintenance.europe_transfer_20261126.evaluate compare
python -m maintenance.europe_transfer_20261126.figures_report
```

首次制包已有同址修正提案历史，当前data.npz不可原地覆盖；重新执行请使用新的版本目录。数据、权重和预测数组位于本机`cache/`与`runs/*/*.pt`，代码、轻量指标、源哈希和图件同步GitHub，原Model4文件保留。
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')


def run():
    FIG.mkdir(exist_ok=True);station_map();architecture();learning_curves();summary=results();report(summary)
    write_json(HERE/'figure_manifest.json',{p.name:sha(p) for p in sorted(FIG.iterdir())})


if __name__=='__main__':run()
