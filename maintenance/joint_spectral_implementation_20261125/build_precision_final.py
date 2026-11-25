"""Build the measured outcome, spatial figures and next encoder decisions."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

from precision_analysis import ROOT,ENGINE,statistics,bootstrap_macro
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.coast_features import load_coast,CRS


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def average(records,case,key):return float(np.mean([r[key] for r in records if r['case']==case]))


def main():
    results=ENGINE/'results';out=results/'20261125_warm_joint_precision/combined_precision'
    c=read(out/'comparison.json');g=read(out/'geographic_and_pair_metrics.json')
    extra=results/'20261125_innovation_calendar15';additional=read(extra/'results.json')
    ag=read(extra/'geographic_and_pair_metrics.json');records=c['records']
    selected=read(results/'20261125_observed_innovation_precision/frozen_selection.json')
    if selected['primary_candidate']!='I1_history':raise ValueError('report expects the frozen I1 selection')
    data=results/'20261125_temporal_continuous_data';geo=dict(np.load(data/'geometry.npz'))
    pooled=[];pooled_draws={};month_indices=np.random.default_rng(252511).integers(0,12,(1000,12))
    paired_indices=np.concatenate([month_indices,month_indices+12],axis=1)
    for seed in (42,17):
        refdraw=None;reference=None
        for name in ['reference','I0_current','I1_history','I2_seasonal']:
            primary_path=results/f'20261125_p14_continuous_fusion/gru_s{seed}/evaluation.npz' if name=='reference' else results/f'20261125_observed_innovation_precision/{name}_s{seed}/evaluation.npz'
            a=np.load(primary_path);b=np.load(extra/f'{name}_s{seed}/evaluation.npz')
            prediction=np.concatenate([a['prediction'],b['prediction']],axis=1)
            truth=np.concatenate([a['truth'],b['truth']],axis=0)
            metrics=statistics(prediction,truth)
            draws=bootstrap_macro(prediction.astype('float64')-truth[None],paired_indices)
            if name=='reference':reference=metrics;refdraw=draws
            pooled_draws[name,seed]=draws-refdraw
            pooled.append(dict(case=name,seed=seed,**metrics,
                delta_macro_rmse_mm=metrics['macro_rmse_mm']-reference['macro_rmse_mm']))
    pooled_means=[]
    for name in ['reference','I0_current','I1_history','I2_seasonal']:
        rows=[r for r in pooled if r['case']==name]
        pooled_means.append(dict(case=name,macro_rmse_mm=average(pooled,name,'macro_rmse_mm'),
            increment_macro_rmse_mm=average(pooled,name,'increment_macro_rmse_mm'),
            paired95_mm=np.quantile((pooled_draws[name,42]+pooled_draws[name,17])/2,[.025,.975]).tolist()))
    write_json(out/'pooled_calendar_metrics.json',dict(records=pooled,means=pooled_means,
        grouping='24 separate72h blocks; resample the two calendars jointly by month; no cross-gap increments',
        no_new_fitting=True))
    panels=[('nearest_current_condition_km',['<50','50-100','100-200','>=200'],'Condition distance (km)'),
            ('coast_km',['<50','50-150','150-300','>=300'],'Coast distance (km)'),
            ('height_m',['<500','500-1500','>=1500'],'Height (m)'),
            ('terrain_roughness_m',['<100','100-300','>=300'],'Terrain roughness (m)'),
            ('seasons',['winter','spring','summer','autumn'],'Season'),
            ('station_pairs',['0-50','50-150','150-300'],'Station-pair distance (km)')]
    summaries={};fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,(key,labels,title) in zip(axes.flat,panels):
        field='micro_rmse_mm' if key=='station_pairs' else 'macro_rmse_mm';table=[]
        for i,label in enumerate(labels):
            index=label if key=='seasons' else i
            def score(source,case):return float(np.mean([source[f'{case}_s{s}'][key][index][field] for s in (42,17)]))
            old,new=score(g,'reference'),score(g,'I1_history')
            aold,anew=score(ag,'reference'),score(ag,'I1_history')
            table.append(dict(group=label,reference=old,I1=new,delta=new-old,
                              calendar15_reference=aold,calendar15_I1=anew,calendar15_delta=anew-aold))
        summaries[key]=table;x=np.arange(len(labels))
        ax.bar(x-.18,[r['reference'] for r in table],.36,color='#8b9da7',label='Original GRU+p14')
        ax.bar(x+.18,[r['I1'] for r in table],.36,color='#35836c',label='History innovation I1')
        ax.set_xticks(x,labels,rotation=15);ax.set_title(title)
        ax.set_ylabel('Difference micro RMSE (mm)' if key=='station_pairs' else 'Macro RMSE (mm)')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Primary calendar: matched two-seed regional and temporal evaluation')
    regional=out/'selected_regional_analysis.png';fig.savefig(regional,dpi=170);plt.close(fig)
    write_json(out/'selected_regional_analysis.json',summaries)
    station=[]
    for seed in (42,17):
        old=np.load(results/f'20261125_p14_continuous_fusion/gru_s{seed}/evaluation.npz')
        new=np.load(results/f'20261125_observed_innovation_precision/I1_history_s{seed}/evaluation.npz')
        def station_rmse(pred):
            error=pred.astype('float64')-old['truth'][None];n=np.isfinite(error).sum((1,2))
            return np.sqrt(np.divide(np.nansum(error**2,(1,2)),n,out=np.full(n.shape,np.nan),where=n>0))
        station.append((station_rmse(old['prediction']),station_rmse(new['prediction'])))
    def safe_mean(a):
        count=np.isfinite(a).sum(0)
        return np.divide(np.nansum(a,0),count,out=np.full(count.shape,np.nan),where=count>0)
    baseline=safe_mean(np.concatenate([r[0] for r in station]));candidate=safe_mean(np.concatenate([r[1] for r in station]))
    change=candidate-baseline;good=np.isfinite(change)
    expected=average(records,'I1_history','macro_rmse_mm')-average(records,'reference','macro_rmse_mm')
    identity=abs(float(np.mean(change[good]))-expected)
    if identity>1e-6:raise ValueError('station map does not reproduce overall paired macro change')
    inverse=Transformer.from_crs(CRS,4326,always_xy=True)
    lon,lat=inverse.transform(geo['dev_xy'][:,0]*1000,geo['dev_xy'][:,1]*1000)
    ids=geo['dev_ids'].astype(str)
    csv_path=out/'station_changes.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['station','longitude','latitude','height_m','coast_km','roughness_m','reference_macro_rmse_mm','I1_macro_rmse_mm','change_mm'])
        for i in np.flatnonzero(good):writer.writerow([ids[i],lon[i],lat[i],geo['dev_height'][i],geo['dev_coast'][i],geo['dev_static'][i,5]*500,baseline[i],candidate[i],change[i]])
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained');xy=geo['dev_xy']
    coast=data/'coast_source/ne_50m_coastline.geojson'
    if coast.exists():
        lines,_=load_coast(coast.parent)
        for ax in axes:
            for line in lines:ax.plot(line[:,0],line[:,1],color='#b0b8bb',lw=.6,zorder=0)
    limit=max(.1,float(np.max(np.abs(change[good]))))
    first=axes[0].scatter(xy[good,0],xy[good,1],c=candidate[good],cmap='YlOrRd',s=32,edgecolors='white',linewidths=.3)
    second=axes[1].scatter(xy[good,0],xy[good,1],c=change[good],cmap='RdYlGn_r',vmin=-limit,vmax=limit,s=32,edgecolors='white',linewidths=.3)
    fig.colorbar(first,ax=axes[0],label='I1 station macro RMSE (mm)',shrink=.65)
    fig.colorbar(second,ax=axes[1],label='I1 - reference (mm); negative improves',shrink=.65)
    for ax,title in zip(axes,['Error remaining at held-out stations','Change relative to the same frozen source']):
        ax.set(xlim=(xy[:,0].min()-100,xy[:,0].max()+100),ylim=(xy[:,1].min()-100,xy[:,1].max()+100),xlabel='LCC easting (km)',ylabel='LCC northing (km)',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15)
    worst=np.flatnonzero(good)[np.argsort(candidate[good])[-5:][::-1]]
    for i in worst[:3]:axes[0].annotate(ids[i],xy[i],xytext=(5,4),textcoords='offset points',fontsize=8)
    spatial=out/'selected_station_map.png';fig.savefig(spatial,dpi=170);plt.close(fig)
    write_json(out/'selected_station_verification.json',dict(overall_macro_identity_difference_mm=identity,
        station_count=int(good.sum()),coastline_sha256=sha(coast) if coast.exists() else None,
        worst_stations=[dict(station=ids[i],longitude=float(lon[i]),latitude=float(lat[i]),I1_rmse_mm=float(candidate[i]),change_mm=float(change[i])) for i in worst]))
    rel='../../'+out.relative_to(ROOT).as_posix();erel='../../'+extra.relative_to(ROOT).as_posix()
    lines=['# 约10 mm基础场优化：结果、架构与下一步编码','',
        '在相同主队列上，历史观测创新校正I1将双种子宏RMSE从9.929151降至9.817064 mm，改善0.112086 mm（约1.13%），月份配对95%区间为[−0.193456,−0.042281] mm。冻结同一模型后，补充日历从10.869282降至10.740343 mm，两种子也均改善。','',
        '研究任务为当前时刻空间重建R：2016–2019训练，2020条件站截至当前时刻的观测用于重建隐藏位置；两个日历均为开发评价。主日历每月8–10日，补充日历每月15–17日，各12×72小时×3布局、137登记站/135有效站。主队列327330条有效记录，补充326937条；它们共享站点、月份、年份和布局，分别报告。','',
        '## 模型选择与关键证据','',
        '| 模型 | 主日历宏RMSE/mm | 主日历小时增量/mm | 补充日历宏RMSE/mm | 补充日历小时增量/mm |',
        '|---|---:|---:|---:|---:|']
    for name in ['reference','I0_current','I1_history','I2_seasonal']:
        a=next(r for r in additional['means'] if r['case']==name)
        lines.append(f"| {name} | {average(records,name,'macro_rmse_mm'):.6f} | {average(records,name,'increment_macro_rmse_mm'):.6f} | {a['macro_rmse_mm']:.6f} | {a['increment_macro_rmse_mm']:.6f} |")
    lines+=['','I0只用当前创新，点精度改善而小时增量增加；I1加入1h/3h创新后同时保留点精度与动态表现。I1相对I0的宏RMSE差为−0.032182 mm，配对区间[−0.049616,−0.017093] mm。I2相对I1的额外差为−0.002878 mm，区间[−0.008967,0.002346] mm，故在补充日期读取前冻结9维I1；补充日历中I1也略优于I2。','',
        '合并两个日历为72个抽样日、1728个不重复小时，每个72h段分别处理；先累计每站平方误差及有效数，再开方，未直接平均两张表的RMSE。月份重采样同时带上该月两段日期，增量不跨日期空档。','',
        '| 72个抽样日合并 | 宏RMSE/mm | 小时增量/mm | 相对参考月份配对95%区间/mm |','|---|---:|---:|---|']
    for row in pooled_means:lines.append(f"| {row['case']} | {row['macro_rmse_mm']:.6f} | {row['increment_macro_rmse_mm']:.6f} | {row['paired95_mm']} |")
    lines+=['','该合并分数代表更广的开发日期样本，仍非完整年度；模型选择已在补充日历读取前冻结。','',
        '神经分支T2/T5、查询融合Q、原始站点Transformer S、IDW空间异常A和距离加权D均已完成两种子3000更新，共14条成功轨迹、42000次梯度更新。D对S的EMA改善约0.008096 mm且两种子同方向，但相对原GRU参考的seed17仍略增。I系列另累计每种子3000个连续训练块，以解析岭回归拟合三组头，不计作新增神经梯度更新。','',
        f'[完整30组记录（28候选＋2参考）、指标及配对门槛]({rel}/REPORT.md)｜[补充日历逐项结果]({erel}/REPORT.md)','',
        '## 当前架构','',
        '冻结原GRU+p14及地形/海岸分支，在每个可见条件站计算观测减去同位置模型预测的误差。将误差按3 km高程尺度输送，并在50/150/300 km软核上查询；当前、1h、3h组成9维创新编码。训练目标是隐藏位置的基础场残差，采用D距离权重、0.1小时增量与岭项；最终修正限制±5 mm。每次背景输入72h，历史创新使总原始观测跨度达到75h，同一条件站集合贯穿全部历史。','',
        '![已实现的基础场与创新编码](../figures/observed_innovation_20261125/architecture.png)','',
        '该提升来自历史观测创新校正。新增Transformer分支尚未对原参考形成稳定双种子优势，新diffusion也尚未在该基础场上重训。','',
        '## 六类误差与重点区域','',
        '时间留出使用2020年，空间留出使用137个登记开发站；训练监督只来自410站池中不属于条件物理组的目标。小时增量在每个72小时连续段内部评价，包含12小时计算批的边界。站对差值采用micro RMSE，其他分层采用站宏RMSE，所有表中数值为双种子均值。','',
        f'![六类分层比较]({rel}/selected_regional_analysis.png)','',
        '| 分层 | 主参考/mm | 主I1/mm | 主变化/mm | 补充参考/mm | 补充I1/mm | 补充变化/mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,labels,title in panels:
        for row in summaries[key]:lines.append(f"| {title} {row['group']} | {row['reference']:.3f} | {row['I1']:.3f} | {row['delta']:+.3f} | {row['calendar15_reference']:.3f} | {row['calendar15_I1']:.3f} | {row['calendar15_delta']:+.3f} |")
    lines+=['','主日历海岸150–300 km档增加0.104 mm，触发重点审阅；补充日历同档改善0.110 mm。距条件站100–200 km档也从主日历增加0.051 mm转为补充改善0.514 mm，说明固定地理分档内仍有明显天气过程差异。夏季主日历基本持平，补充改善0.164 mm；远条件站200 km以上仍约14–16 mm。后续应同时利用天气状态与地形约束创新传递，而不是按主日历的局部误差直接设置固定地理开关。','',
        f'![逐站误差与变化]({rel}/selected_station_map.png)','',
        f'[逐站经纬度、高程、海岸距离、粗糙度与误差表]({rel}/station_changes.csv)。主日历最大剩余误差站依次为：'+', '.join(f"{ids[i]} {candidate[i]:.2f} mm" for i in worst)+'。这些站点用于下一轮天气过程和空间支撑诊断。','',
        '其中PLTX/P475/PTEX位于美墨边境及下加利福尼亚一带，改善后仍约17–18 mm；研究域东北侧P049约16.27 mm且增加0.134 mm，北加州近海P786约15.97 mm。后续须分别核对这些站点的天气过程、条件站方向/距离及高差，而不把单一日历的分档变化解释为固定地理规律。','',
        '## 曲线、验证与来源','',
        f'![创新头累计训练目标]({rel}/innovation_training_objective.png)','',
        'I曲线是岭解在逐渐扩大的连续训练前缀上的水平＋0.1增量数据目标，既不是随机批点RMSE，也不是固定总体的收敛曲线。原始数据、预测大数组、训练统计张量和系数留本机；Git同步代码、协议、指标、哈希与图件。六组系数均由最终训练统计独立重解，逐元素完全一致；主候选及系数哈希在读取补充日期前锁定。','',
        '## 下一步模型与实验','',
        '以I1为性能下限，下一版因果Transformer应编码逐站模型误差、原始GNSS/T/P、mask/age及地形/海岸关系。保留原始站点身份与方向，经过时间编码再用地形/海岸条件查询注意力传递；新增均值头零初始化。优先比较GRU、LSTM、Transformer在同等有效历史范围下的表现，并检验创新表示与包含同等信息的原始值＋模型预测表示。','',
        '误差历史会扩大所需原始观测跨度：若每次源背景72h、校正滞后3h，再读取L小时的校正后站点误差，总跨度为72＋3＋L−1；L=12/24/72/168分别需要86/98/146/242小时。条件站集合须覆盖整个跨度保持一致；缓存键应包含条件站集合、时间范围、权重和数据合同哈希，不直接拼接不同条件布局的上一个训练块。','',
        '两个2020日历现均已开发暴露。后续残差训练重建严格空间折外来源，重新拟合基础场及I1系数，再用训练折统计标准化残差；Gaussian与diffusion共享条件编码、表示和采样预算，检验CRPS、区间校准、时间增量及站对联合误差。频域/图谱分支继续作为同条件表示对照。当前结果未给出新diffusion相对该9.817 mm基础场的收益数值。']
    note=ROOT/'docs/research_notes/2026-11-25-precision-final.md'
    note.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    write_json(out/'final_figure_manifest.json',dict(script_sha256=sha(Path(__file__)),
        primary_comparison_sha256=sha(out/'comparison.json'),additional_results_sha256=sha(extra/'results.json'),
        figures={p.name:sha(p) for p in [regional,spatial]},report_sha256=sha(note),station_csv_sha256=sha(csv_path)))
    print(json.dumps(dict(report=str(note),station_identity_difference_mm=identity,worst_station_ids=ids[worst].tolist())))


if __name__=='__main__':main()
