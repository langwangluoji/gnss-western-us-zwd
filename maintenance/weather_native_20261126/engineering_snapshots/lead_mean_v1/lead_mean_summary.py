"""Separate mean-center effects from fixed Gaussian/paired residual structure."""
import json
from pathlib import Path
import numpy as np
from campaign import RESULTS,ROOT,read
from weather_controls_summary import paired_record
from paired_mean_summary import paired_interval
from lead_mean_audit import mean_centers,checked_source
from tools.terrain_diffusion.data import sha,write_json


def main():
    root=RESULTS/'20261126_lead_mean_audit';lockpath=Path(__file__).with_name('lead_mean_audit_lock.json');lock=read(lockpath)
    code=Path(__file__).with_name('lead_mean_audit.py');engine=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
    rows=[];records={};sources={}
    for phase in ['gaussian','paired']:
        for seed in lock['seeds']:
            for day in lock['gaussian_calendars'] if phase=='gaussian' else [lock['paired_calendar']]:
                for model in ['reference'] if phase=='gaussian' else lock['paired_models']:
                    baseline=None
                    for center in lock['centers']:
                        p=root/phase/f's{seed}/calendar{day}'/model/center/'metrics.json';r=read(p)
                        if r['phase']!=phase or r['seed']!=seed or r['calendar']!=day or r['model']!=model or r['center']!=center:
                            raise ValueError('intervention identity differs')
                        if r['lock_sha256']!=sha(lockpath) or r['code_sha256']!=sha(code):raise ValueError('intervention implementation changed')
                        for name,value in r['dependency_sha256'].items():
                            path=engine/'tools/terrain_diffusion'/name if name=='native_weather_evaluation.py' else Path(__file__).with_name(name)
                            if sha(path)!=value:raise ValueError('intervention scoring dependency changed')
                        if r['source']!=lock['sources'][f'calendar{day}/s{seed}/{model}'] or r['center_source']!=lock['sources'][f'calendar{day}/s{seed}/reference']:
                            raise ValueError('intervention source differs')
                        if r['members']!=(32 if phase=='gaussian' else 16) or not r['analytic_center_known'] or not r['exact_unchanged_residual_verified']:
                            raise ValueError('intervention math recipe differs')
                        if baseline is None:baseline=r['base_point_metrics']
                        elif baseline!=r['base_point_metrics']:raise ValueError('base point scores differ by center')
                        key=(phase,day,model,center);records[seed,*key]=r['records'];rows.append(r);sources[str(p)]=sha(p)
    comparisons={};point_comparisons={}
    for phase in ['gaussian','paired']:
        for day in lock['gaussian_calendars'] if phase=='gaussian' else [8]:
            models=['reference'] if phase=='gaussian' else lock['paired_models']
            edges=[((m,a),(m,b)) for m in models for a,b in [('gaussian','base'),('hybrid','base'),('hybrid','gaussian')]]
            if phase=='paired':
                for center in lock['centers']:
                    for solver in ['heun0','ddim']:
                        edges += [(('local_'+solver,center),('affine_'+solver,center)),(('local_'+solver,center),('reference',center)),(('affine_'+solver,center),('reference',center))]
            for (a,ac),(b,bc) in edges:
                name=f'{phase}/calendar{day}/{a}/{ac}_vs_{b}/{bc}'
                keys=set(records[42,phase,day,a,ac])&set(records[17,phase,day,a,ac])&set(records[42,phase,day,b,bc])&set(records[17,phase,day,b,bc])
                comparisons[name]={k:paired_record([(records[s,phase,day,a,ac][k],records[s,phase,day,b,bc][k]) for s in lock['seeds']]) for k in sorted(keys)}
    for day in lock['gaussian_calendars']:
        errors={center:[] for center in lock['centers']}
        for seed in lock['seeds']:
            parent,_=checked_source(seed,day,'reference',lock)
            with np.load(parent/'evaluation.npz') as data:
                centers=mean_centers(data['prediction'],data['gaussian_distribution_mean']);truth=data['truth'].astype('float64')
                parent_metrics=read(parent/'metrics.json')
                row=next(r for r in rows if r['phase']=='gaussian' and r['calendar']==day and r['seed']==seed and r['center']=='gaussian')
                np.testing.assert_allclose([row['records'][f'analytic_crps_h{h}']['value'] for h in range(7)],parent_metrics['analytic_gaussian_crps_mm_by_horizon'],atol=2e-7,rtol=0)
                np.testing.assert_allclose(row['analytic_center_point_metrics']['horizon_macro_rmse_mm'],parent_metrics['gaussian_distribution_mean']['horizon_macro_rmse_mm'],atol=1e-10,rtol=0)
                for center,mu in centers.items():errors[center].append(mu-truth)
        for a,b in [('gaussian','base'),('hybrid','base'),('hybrid','gaussian')]:
            point_comparisons[f'calendar{day}/{a}_vs_{b}']=dict(macro_RMSE_delta_ci95_mm_by_horizon=paired_interval(np.stack(errors[a]),np.stack(errors[b])))
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,comparisons=comparisons,analytic_center_point_comparisons=point_comparisons,sources=sources,
        code_sha256=sha(__file__),lock_sha256=sha(lockpath),intervention_code_sha256=sha(code),
        point_interval_code_sha256=sha(Path(__file__).with_name('paired_mean_summary.py')),
        interval='2000 shared-month draws across two fixed seeds; exposed2020 exploratory; no multiplicity correction',
        strict_oof=False,confirmation_C_D_read=False))
    render(out,rows,comparisons,point_comparisons)


def render(out,rows,comparisons,point_comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def average(phase,day,model,center,key):
        rr=[r for r in rows if r['phase']==phase and r['calendar']==day and r['model']==model and r['center']==center]
        return np.mean([r['records'][key]['value'] for r in rr])
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    colors={'base':'#6d7c8d','gaussian':'#ba8548','hybrid':'#34847b'}
    for center in ['base','gaussian','hybrid']:
        axes[0].plot(range(7),[average('gaussian',8,'reference',center,f'analytic_crps_h{h}') for h in range(7)],'o-',label=center,color=colors[center])
        rr=[r for r in rows if r['phase']=='gaussian' and r['calendar']==8 and r['center']==center]
        axes[1].plot(range(7),np.mean([r['analytic_center_point_metrics']['horizon_macro_rmse_mm'] for r in rr],axis=0),'o-',label=center,color=colors[center])
    keys=['crps_h0','crps_h6','increment','pair_0-50_h6','energy']
    for j,center in enumerate(['base','gaussian','hybrid']):
        table=comparisons[f'paired/calendar8/local_heun0/{center}_vs_affine_heun0/{center}'];v=np.array([table[k]['delta'] for k in keys]);ci=np.array([table[k]['delta_ci95'] for k in keys])
        axes[2].errorbar(np.arange(len(keys))+(j-1)*.14,v,yerr=np.maximum(np.stack([v-ci[:,0],ci[:,1]-v]),0),fmt='o',capsize=3,label=center,color=colors[center])
    axes[0].set(xlabel='Lead (h)',ylabel='Analytic Gaussian CRPS (mm)',title='Unchanged sigma and correlation')
    axes[1].set(xlabel='Lead (h)',ylabel='Analytic center macro RMSE (mm)',title='Same frozen center sources')
    axes[2].axhline(0,color='gray',ls='--');axes[2].set(xticks=range(len(keys)),xticklabels=['R','F6','increment','near pair','energy'],ylabel='Paired Local minus Affine score (mm)',title='Fixed shape; K16 for all centers')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.savefig(out/'lead_center_comparison.png',dpi=170);plt.close(fig)
    lines=['# 按时效中心消融：均值与概率结构分开','',
        'base为全时效TF3000中心；gaussian为全时效S175解析中心；hybrid为h0=TF3000、h1–h6=S175。规则及源清单在新成绩前冻结，不拟合开发门控或混合系数。A固定G265的K32随机残差，B固定独立配对后的K16随机残差；新操作均不训练网络。','',
        '| 阶段/日历 | 模型 | 中心 | 解析当前宏RMSE/mm | 解析F6宏RMSE/mm | 当前解析CRPS/mm | F6解析CRPS/mm | 增量fair CRPS/mm | 近邻F6差值CRPS/mm | energy/mm |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase,day in [('gaussian',8),('gaussian',15),('paired',8)]:
        for model in ['reference'] if phase=='gaussian' else ['reference','local_heun0','affine_heun0']:
            for center in ['base','gaussian','hybrid']:
                rr=[r for r in rows if r['phase']==phase and r['calendar']==day and r['model']==model and r['center']==center]
                rmse=np.mean([r['analytic_center_point_metrics']['horizon_macro_rmse_mm'] for r in rr],axis=0)
                def value(key):return f'{average(phase,day,model,center,key):.6f}' if key in rr[0]['records'] else 'NA'
                lines.append(f"| {phase}/{day} | {model} | {center} | {rmse[0]:.6f} | {rmse[6]:.6f} | {value('analytic_crps_h0')} | {value('analytic_crps_h6')} | {value('increment')} | {value('pair_0-50_h6')} | {value('energy')} |")
    lines+=['','| 比较 | 指标 | 改善 | 差值95%区间 |','|---|---|---:|---|']
    chosen=[f'gaussian/calendar{day}/reference/hybrid_vs_reference/{reference}' for day in [8,15] for reference in ['base','gaussian']]
    chosen += ['paired/calendar8/local_heun0/hybrid_vs_affine_heun0/hybrid','paired/calendar8/local_heun0/hybrid_vs_local_heun0/base']
    for name in chosen:
        table=comparisons[name]
        for key in ['analytic_crps_h0','analytic_crps_h6','crps_h0','crps_h6','increment','pair_0-50_h6','energy','event20_increase_brier','event20_decrease_brier']:
            if key in table:
                v=table[key];lines.append(f"| {name} | {key} | {v['gain_percent']:.3f}% | {v['delta_ci95']} |")
    lines+=['','## 解释','',
        '同一模型、同一阶段中，每个成员的随机残差逐值不变，因此中心比较不再混入形状或噪声变化。解析中心由可见条件的冻结模型决定，truth只参与评分。不同中心的尺度和总体相关结构不变，站对均值和跨lead增量均值允许改变。',
        '配对Local/Affine共享同一解析中心，其点值RMSE相同；不同分布只比较概率评分。K16/K32样本均值的有限成员误差另存JSON，不能当解析均值排名。解析CRPS仅适用于Gaussian阶段，配对非Gaussian阶段记NA。',
        '主2020每月8–10日，补充15–17日均已开发暴露，三布局137查询。缺标签按相同mask计分；确认及原C/D标签未读取。主事件20mm，10/20/30mm及两方向全部见JSON。',
        '点值宏RMSE与概率分数使用各自匹配统计量；点值月份区间保留站/布局结构。所有区间条件于固定两种子，未经多重比较校正。','',
        '![中心与概率结构的分离](lead_center_comparison.png)','','完整源哈希、三个中心所有时效、DDIM复核、事件、覆盖和区间见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(selected={name:comparisons[name] for name in chosen},point=point_comparisons)),flush=True)


if __name__=='__main__':main()
