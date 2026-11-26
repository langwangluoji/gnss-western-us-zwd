"""Rendered scientific figures and the VMF3-conditioned diffusion report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json
from maintenance.europe_transfer_20261126.figures_report import coast

FIG=HERE/'figures';BLUE='#176B9B';ORANGE='#D77636';GREEN='#25816C';INK='#213548'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');path=FIG/(name+'.svg');fig.savefig(path,bbox_inches='tight');plt.close(fig)
    path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


def architecture():
    fig,ax=plt.subplots(figsize=(15,10));ax.set(xlim=(0,15),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,title,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.1',facecolor=color,edgecolor='none'))
        ax.text(x+.13,y+h-.2,title,va='top',weight='bold',color=INK,fontsize=11)
        ax.text(x+.13,y+h-.68,text,va='top',color=INK,fontsize=9.5,linespacing=1.45)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#5E778B'))
    ax.text(0,9.75,'VMF3-FC | European foundation field + space-time residual diffusion',weight='bold',fontsize=17,color=INK)
    box(.1,7.85,4.35,1.35,'Frozen GNSS foundation','Reference GNSS / T / P; terrain / coast\np14 U-Net + GRU + I1; query sites hidden','#E1EDF4')
    box(5,7.85,4.5,1.35,'VMF3-FC: current + 16-state history','1 degree forecast archive; vertical then horizontal\nNative / aligned / anomaly + mask; GRU32','#DCEEE6')
    box(10.05,7.85,4.7,1.35,'New frozen mean at 7 leads','VMF3 current correction + native forecast increments\n0 / 6 / 12 / 18 / 24 / 30 / 36 h','#DDE8F4')
    arrow((4.55,8.52),(4.9,8.52));arrow((9.6,8.52),(9.95,8.52))
    box(.1,5.3,6.8,1.85,'Explicit probability condition: 218 + 90 channels','218 = native h64 + query111 + final mean7 + VMF3 current4 + VMF3 GRU32\n90 = fixed-reference weather mean/std over last1/4/16 records\nMatched ablation: zero only the explicit VMF3 current4 + GRU32','#EAF0F4')
    box(7.5,5.3,7.25,1.85,'Rebuild and standardize European residuals','e = (GNSS ZWD - new mean - training bias) / training scale\nTrain2020-2023 only; scale about11 mm now to33 mm at36 h\n2024 retrospective; same59 active confirm sites / 63,494 anchors','#F5EBDC')
    arrow((12.4,7.72),(12.4,7.23))
    box(.1,2.6,4.3,1.85,'Conditional Gaussian','218-D condition -> residual location / scale\nMatched base / explicit-VMF3 conditions\nDeterministic point mean remains frozen','#E0EBF3')
    box(5,2.6,4.5,1.85,'Weather projective GP','Learned space-time correlation, unit diagonal\nPSD kernel; no pooling over query sites\nGP-correlated IID field noise','#DDECE7')
    box(10.1,2.6,4.65,1.85,'Affine / Local v diffusion','Whiten each7-lead trajectory\n308-D condition; Heun0 / 20 steps\nSame frozen GP paths for comparisons','#E5E5F3')
    arrow((3.4,5.18),(2.2,4.55));arrow((11,5.18),(12.4,4.55));arrow((4.5,3.5),(4.9,3.5));arrow((9.6,3.5),(10,3.5))
    box(.1,.2,14.65,1.45,'Outputs and evidence','Point output: new deterministic field | Probability output: K32 coherent7-lead fields, restored to mm\n14 formal models, two seeds, 3000 consecutive blocks each; Gaussian / GP / Affine / Local attribution\nFixed existing CNN reference; no continued CNN training after user clarification','#EEF1F4')
    arrow((12.4,2.48),(12.4,1.78));save(fig,'01_architecture')


def point_figures(point):
    c=point['cohorts']['confirm'];keys=['old_I1','control','vmf_native','vmf_aligned','vmf_field'];labels=['Old I1','Matched head\nno VMF3','VMF3 native','VMF3 aligned\nreference P','New VMF3\nfoundation'];values=[c[k]['micro_rmse_mm'] for k in keys]
    fig,ax=plt.subplots(figsize=(10,4.6));ax.bar(np.arange(5),values,color=['#9EB4C3','#7E9DB2',ORANGE,'#DBA07B',BLUE])
    for i,v in enumerate(values):ax.text(i,v+.15,f'{v:.3f}',ha='center')
    ax.set(xticks=np.arange(5),xticklabels=labels,ylabel='RMSE (mm)',ylim=(0,max(values)*1.18),title='2024 confirm sites | common63,494 records, two-seed prediction ensemble');ax.grid(axis='y',alpha=.15);fig.tight_layout();save(fig,'02_point_rmse')
    arrays=np.load(CACHE/'point_delivery.npz');a=np.load(OLD_CACHE/'data.npz');role=a['role']=='confirm';months=pd.to_datetime(arrays['hours'].astype('datetime64[h]')).month.to_numpy();fig,ax=plt.subplots(figsize=(10,4))
    for key,color,label in [('old_I1',ORANGE,'Old I1'),('vmf_aligned','#7F9B79','VMF3 aligned'),('vmf_field',BLUE,'New VMF3 field')]:
        vals=[]
        for month in range(1,13):
            mask=arrays['valid']&(months==month)[:,None]&role[None];vals.append(np.sqrt(np.mean((arrays[key][mask]-arrays['target'][mask])**2)))
        ax.plot(np.arange(1,13),vals,'-o',color=color,label=label)
    ax.set(xticks=np.arange(1,13),xlabel='Month in2024',ylabel='RMSE (mm)',title='Seasonal current-field accuracy');ax.grid(alpha=.18);ax.legend();fig.tight_layout();save(fig,'03_monthly_point')


def probability_figures(prob):
    models=prob['models'];colors={'base_local':'#8497A5','vmf_gaussian':'#D0A883','vmf_gp':ORANGE,'vmf_affine':GREEN,'vmf_local':BLUE};fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for key,color in colors.items():
        d=models[key];label=key.replace('_',' ');axes[0].plot(np.arange(0,37,6),d['fair_crps_mm_by_lead'],'-o',color=color,label=label)
        axes[1].plot(np.arange(0,37,6),np.array(d['rank3_30_coverage'])*100,'-o',color=color,label=label)
    axes[0].set(xlabel='Lead time (hours)',ylabel='Fair CRPS (mm)',title='Marginal forecast distribution');axes[1].set(xlabel='Lead time (hours)',ylabel='Rank3-30 interval coverage (%)',title='K32 finite-member calibration');axes[1].axhline(100*27/33,color=INK,ls='--',lw=1,label='Exchangeable reference81.82%')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'04_probability_performance')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,key,title in zip(axes,['vmf_local_minus_base_local','vmf_local_minus_vmf_affine'],['Extra explicit VMF3 probability condition','Local nonlinear versus matched affine']):
        v=prob['contrasts'][key];y=np.array(v['delta_fair_crps_mm_by_lead']);lo=np.array(v['month_block95_low']);hi=np.array(v['month_block95_high'])
        ax.errorbar(np.arange(0,37,6),y,yerr=np.stack([y-lo,hi-y]),fmt='o-',color=BLUE,capsize=3);ax.axhline(0,color=INK,lw=1);ax.set(xlabel='Lead time (hours)',ylabel='CRPS difference (mm); negative = better',title=title);ax.grid(alpha=.18)
    fig.tight_layout();save(fig,'05_attribution_intervals')


def curves():
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    for seed,color in [(42,BLUE),(17,ORANGE)]:
        rows=[json.loads(s) for s in (HERE/f'runs/vmf3_s{seed}/curve.jsonl').read_text(encoding='utf-8').splitlines() if s];selected=[r for r in rows if 'dev_station_macro_rmse_mm' in r]
        axes[0].plot([r['step'] for r in selected],[r['dev_station_macro_rmse_mm'] for r in selected],'-o',color=color,label=f'VMF3 mean, seed{seed}')
        for ax,kind in zip(axes[1:],['gaussian','local']):
            for view,style in [('base','--'),('vmf','-')]:
                r=json.loads((HERE/f'probability/s{seed}/{view}/{kind}/curve.json').read_text(encoding='utf-8'));ax.plot([v['step'] for v in r],[v['loss'] for v in r],style,color=color,label=f'{view}, seed{seed}')
    axes[0].set(title='New mean | 2023 spatial-dev',ylabel='Station macro RMSE (mm)');axes[1].set(title='Conditional Gaussian | varying blocks',ylabel='Standardized marginal NLL');axes[2].set(title='Local v diffusion | varying blocks',ylabel='Velocity MSE')
    for ax in axes:ax.set_xlabel('Consecutive-block updates');ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'06_training_curves')


def station_maps():
    points=pd.read_csv(HERE/'point_station_rmse.csv');d=points[points.role=='confirm'];fig,axes=plt.subplots(1,3,figsize=(15,6))
    vmax=float(np.ceil(max(d.old_I1.max(),d.vmf_field.max())/5)*5);maximum=max(1,float(np.ceil(np.max(abs(d.vmf_field-d.old_I1))/5)*5))
    for ax,key,title in zip(axes[:2],['old_I1','vmf_field'],['Old GNSS foundation','New VMF3 foundation']):
        coast(ax);absolute=ax.scatter(d.Longitude,d.Latitude,c=d[key],s=35,cmap='viridis',vmin=0,vmax=vmax,edgecolors='white',linewidths=.3);ax.set_title(title,loc='left')
    coast(axes[2]);delta=axes[2].scatter(d.Longitude,d.Latitude,c=d.vmf_field-d.old_I1,s=35,cmap='RdBu_r',vmin=-maximum,vmax=maximum,edgecolors='white',linewidths=.3);axes[2].set_title('New - old (negative = better)',loc='left')
    for ax in axes[1:]:ax.set_ylabel('')
    fig.subplots_adjust(bottom=.26,wspace=.2);fig.colorbar(absolute,cax=fig.add_axes([.17,.11,.4,.025]),orientation='horizontal',label='Station RMSE (mm), full range');fig.colorbar(delta,cax=fig.add_axes([.71,.11,.19,.025]),orientation='horizontal',label='RMSE difference (mm)')
    save(fig,'07_station_maps')


def report(point,prob):
    current=point['cohorts']['confirm'];old=current['old_I1']['micro_rmse_mm'];new=current['vmf_field']['micro_rmse_mm'];aligned=current['vmf_aligned']['micro_rmse_mm'];models=prob['models'];local=models['vmf_local'];gp=models['vmf_gp'];affine=models['vmf_affine']
    names=dict(old_cnn='既有CNN固定参考',old_I1='原p14＋GRU＋I1',control='新增校正器，无VMF3输入',vmf_field='新VMF3基础场',vmf_native='VMF3 native直接背景',vmf_aligned='VMF3 aligned直接背景（参考站P）')
    point_rows='\n'.join(f'| {names[name]} | {v["micro_rmse_mm"]:.4f} | {v["station_macro_rmse_mm"]:.4f} |' for name,v in current.items())
    probability_rows='\n'.join(f'| {key} | {v["fair_crps_mm_by_lead"][0]:.5f} | {v["fair_crps_mm_by_lead"][1]:.5f} | {v["fair_crps_mm_by_lead"][6]:.5f} | {v["six_hour_increment_fair_crps_mm"]:.5f} | {v["paired_energy_mm"]:.5f} |' for key,v in models.items())
    contrasts=prob['contrasts'];ci=contrasts['vmf_local_minus_vmf_affine'];extra=contrasts['vmf_local_minus_base_local'];priority=pd.read_csv(HERE/'priority_station_comparison.csv')
    priority_rows='\n'.join(f'| {r.station_phys} | {r.old_I1:.3f} | {r.vmf_field:.3f} | {r.vmf_aligned:.3f} |' for r in priority.itertuples())
    text=f'''# VMF3-FC条件的欧洲基础场＋时空扩散：完整结果

VMF3-FC已同时进入新基础场和残差扩散条件。2024完整年度回看、相同59有效确认站/63494共同起点上，双种子预测集成的当前场RMSE由{old:.4f}降至{new:.4f}mm，降低{(old-new)/old*100:.2f}%；相对VMF3 aligned直接背景的{aligned:.4f}mm降低{(aligned-new)/aligned*100:.2f}%。固定基础场上，Local相对同条件天气GP的+6h CRPS改善{(gp['fair_crps_mm_by_lead'][1]-local['fair_crps_mm_by_lead'][1])/gp['fair_crps_mm_by_lead'][1]*100:.2f}%。非Gaussian相对Affine的额外收益按下方区间单独解释。

按用户修订，已有CNN保留为固定参考，后续研究只训练新融合模型；澄清前已完成的CNN seed42尝试归档，seed17已停止，均不进入本轮主比较。

![新模型架构](figures/01_architecture.png)

## VMF3怎样进入模型

VMF3-FC1°预报归档按原CNN的算法处理：四格点分别进行Kouba高程修正，再双线性到查询点。三个背景特征是native ZWD、与本模型干湿口径对齐的aligned ZWD、aligned减原I1场的异常。aligned中的P由原有参考站T/P按干空气高程传递和空间加权得到；目标站GNSS/T/P、WVP、TCWV和原FFT-SCHA均不作为新条件。缺2023-12-18 00UTC的原始背景用mask保留，不从未来有效时刻填充。

当前均值采用冻结p14＋GRU＋I1、111维查询条件与16步VMF3三特征/mask的GRU32校正。该分支两个种子各3000连续四记录块，2023留站监控选择；另外完成同结构、VMF3输入置零的control。未来7时效保留原欧洲因果Transformer的预测增量，以新当前均值作公共偏移，公式为μnew(t,h)=μjoint(t,h)−μjoint(t,0)+μVMF(t,0)。本轮未来部分是这个明确的锚定规则，尚未使用未来VMF3有效时刻作为输入。

概率查询条件为218维：原查询状态64＋原条件111＋最终均值7＋VMF3当前三特征/mask4＋VMF3 GRU状态32。另有固定参考站15维状态在1/4/16记录上的均值/标准差90维，全局天气摘要不依赖查询集合。16状态跨度90h；7输出是0/6/12/18/24/30/36h。

训练残差按e=(y−μnew−b)/s标准化。b、s由2020–2023训练站重新拟合，s约11mm（当前）至33mm（36h）。条件Gaussian学习残差位置/尺度，WeatherProjectiveGP学习PSD、单位对角的空间—时间相关；Local/Affine将每查询7时效轨迹白化，进行v速度传输，再着色并恢复mm。实际复用原NativeGaussian、WeatherProjectiveGP核、LocalTrajectoryVelocity、AffineTrajectoryVelocity与Heun0；欧洲GP适配218＋90条件，不伪造72小时状态。

物理输出为Z=μnew+b+s[mG+σG·T(η)]，η来自耦合GP场。确定性点输出始终为冻结μnew，随机成员均值与概率中心另作诊断；其采样方差不计为基础场精度。

## 确定性场的收益

表格均为两个训练种子的预测先等权平均，再计算共同键RMSE。native/aligned是直接背景，不含新神经训练；control是相同新校正器、无VMF3输入。CNN来自上轮固定273站模型，使用其原气象湿度和同站历史特征，未启用VMF3，仅保留为固定系统参考。主要归因比较原I1/control与直接VMF3背景。

| 点模型 | micro RMSE/mm | 站宏RMSE/mm |
|---|---:|---:|
{point_rows}

![点值对比](figures/02_point_rmse.png)

![月份变化](figures/03_monthly_point.png)

VMF3背景加入了新的外部物理信息，收益包含条件信息扩充；control显示新增校正器本身的变化较小。native与aligned的差异也说明干湿分离口径对齐的重要性，融合场在对齐背景之上继续改善。

## 扩散条件的对照与概率结果

base与vmf两臂共享同一个新VMF3基础场及最终7时效均值。base仅把额外显式VMF3当前4维和GRU32维置零，所以它表示“无额外显式VMF3概率条件”，并非整个系统无VMF3。两臂各训练Gaussian、GP、Local；vmf另有同预算Affine。共14模型，每个3000连续块；全部Adam/scheduler真实监督时钟3000。

下面分数为两个固定训练种子的平均，未拼接为64成员分布。每模型K32独立成员；Local/Affine采用Heun0 20步，Gaussian/GP直接采样。GP/Local/Affine共享同源GP噪声路径以作配对比较。当前/未来共用上轮固定起点，未来mask仅用于监督和评分。Gaussian与GP具有相同解析边际，其采样CRPS微差来自有限成员；GP主要检验联合相关结构。

| 概率模型 | 当前CRPS/mm | +6h CRPS/mm | +36h CRPS/mm | 内部6h增量CRPS/mm | 配对场能量/mm |
|---|---:|---:|---:|---:|---:|
{probability_rows}

![概率表现](figures/04_probability_performance.png)

Local相对同VMF条件天气GP的+6h CRPS降低{gp['fair_crps_mm_by_lead'][1]-local['fair_crps_mm_by_lead'][1]:.5f}mm。Local相对Affine的+6h差值为{ci['delta_fair_crps_mm_by_lead'][1]:+.5f}mm，95%共同月份重采样区间[{ci['month_block95_low'][1]:+.5f},{ci['month_block95_high'][1]:+.5f}]，区间跨零；当前Local比Affine高{ci['delta_fair_crps_mm_by_lead'][0]:.5f}mm。当前证据支持强Gaussian基础上的传输收益，未建立非Gaussian模块的独立优势。

显式VMF条件对Local的额外当前CRPS差值为{extra['delta_fair_crps_mm_by_lead'][0]:+.5f}mm、36h为{extra['delta_fair_crps_mm_by_lead'][6]:+.5f}mm；+6h为{extra['delta_fair_crps_mm_by_lead'][1]:+.5f}mm，区间[{extra['month_block95_low'][1]:+.5f},{extra['month_block95_high'][1]:+.5f}]。各时效分别判断，不能概括为全时效统一改善。

![归因与月份区间](figures/05_attribution_intervals.png)

K32的第3/30秩区间在交换性条件下参考覆盖率为27/33=81.82%，不是精确80%。Local VMF的当前/+6h秩覆盖为{local['rank3_30_coverage'][0]*100:.2f}%/{local['rank3_30_coverage'][1]*100:.2f}%；解析Gaussian80%覆盖另存JSON。月份区间固定本次两个训练种子和采样流，表示该回看年份内的过程变化。场能量采用独立配对估计并除以观测维数平方根；各模型使用相同观测维度。近邻站对的几何选择、数量和分数见[spatial_contrasts.json](spatial_contrasts.json)。

## 区域与下一步

![逐站点场](figures/07_station_maps.png)

| 上轮重点站 | 原I1/mm | 新VMF场/mm | aligned背景/mm |
|---|---:|---:|---:|
{priority_rows}

VMF3明显改善了上轮较难的沿海/岛站；MERS仍有较高误差，部分站的直接aligned背景优于融合场。下一步优先检验按参考站距离/覆盖/地形调节的背景融合权重，并核实预报发布时间后再接入可在起点获取的多时效VMF3预报。概率侧先解释Gaussian/Affine已获得的校准收益，再针对有证据的非Gaussian残差设计扩散改进。

![训练曲线](figures/06_training_curves.png)

## 数据范围与核验

继承273训练/64开发/63注册确认/11新站的1km物理组隔离；2020–2023拟合、2023开发为留站监控，2024为历史已暴露年度回看。确认63站中59有本轮共同有效记录。原Model4标签和湿度参与制包筛选的谱系保留；strict_oof=false。原始NWP归档仅验证有效时刻映射，未逐文件追溯发布时间，不将本轮称为业务实时预报验证。滚动起点间独立采样，每起点内生成相关7时效场。

[点值完整指标](point_results.json) · [概率完整指标和区间](probability_results.json) · [训练时钟核验](training_verification.json) · [导出来源seed42](diffusion_source_s42.json) · [seed17](diffusion_source_s17.json) · [标准化统计seed42](residual_statistics_s42.json) · [seed17](residual_statistics_s17.json) · [固定采样协议](probability_lock.json) · [原欧洲架构与站址划分](../europe_transfer_20261126/REPORT.md) · [计划](../../docs/superpowers/plans/2026-11-26-europe-vmf3-diffusion.md)

原始网格、缓存、权重和大样本数组留本地；代码、轻量协议、指标、图件和来源哈希同步GitHub。
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')


def run():
    FIG.mkdir(exist_ok=True);point=json.loads((HERE/'point_results.json').read_text(encoding='utf-8'));prob=json.loads((HERE/'probability_results.json').read_text(encoding='utf-8'))
    for seed in (42,17):(HERE/f'residual_statistics_s{seed}.json').write_bytes((CACHE/f'diffusion_s{seed}/statistics.json').read_bytes())
    architecture();point_figures(point);probability_figures(prob);curves();station_maps();report(point,prob)
    write_json(HERE/'figure_manifest.json',{f.name:sha(f) for f in sorted(FIG.iterdir())})


if __name__=='__main__':run()
