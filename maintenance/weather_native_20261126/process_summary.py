"""Independent fair scores and shared-month intervals for frozen GP-process trials."""
import argparse
import json
import numpy as np
from campaign import ROOT,RESULTS,read
from context_summary import score_interval
import sys
from pathlib import Path

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_evaluation import fair_crps


def scalar_interval(sums,counts,reference_sums,reference_counts,draws=2000):
    rng=np.random.default_rng(261126);values=[]
    for _ in range(draws):
        index=rng.integers(len(sums),size=len(sums))
        values.append(sums[index].sum()/counts[index].sum()-reference_sums[index].sum()/reference_counts[index].sum())
    return np.quantile(values,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();root=RESULTS/'20261126_gp_process'/f's{args.seed}';out=root/'summary';out.mkdir(parents=True,exist_ok=True)
    folders=[('reference',root/'reference')]
    folders += [(f'{mode}_{solver}',root/mode/f'eval_{solver}') for mode in ('full','local','affine') for solver in ('heun0','ddim')]
    if all((root/'graph_affine'/f'eval_{s}'/'metrics.json').exists() for s in ('heun0','ddim')):
        folders += [(f'graph_affine_{s}',root/'graph_affine'/f'eval_{s}') for s in ('heun0','ddim')]
    rows=[];scores={};increments={};source=None;query_audits={}
    for label,folder in folders:
        lineage=read(folder/'lineage.json');metrics=read(folder/'metrics.json')
        if source is None:source=lineage['source']
        if source!=lineage['source']:raise ValueError('probability references differ')
        if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('probability array changed')
        with np.load(folder/'evaluation.npz') as data:
            scores[label]=data['crps'].astype('float64');truth=data['truth'];samples=data['samples']
        np.testing.assert_allclose(np.nanmean(scores[label],axis=(0,1,2,3)),metrics['fair_crps_mm_by_horizon'],atol=1e-7,rtol=0)
        if label!='reference':np.testing.assert_array_equal(np.isfinite(scores[label]),np.isfinite(scores['reference']))
        increment=fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1));axes=(0,2,3,4)
        increments[label]=(np.nansum(increment,axes),np.isfinite(increment).sum(axes))
        np.testing.assert_allclose(np.nanmean(increment),metrics['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
        del samples,truth,increment
        row=dict(name=label,fair_crps_mm=metrics['fair_crps_mm_by_horizon'],
            increment_fair_crps_mm=metrics['forecast_increment_fair_crps_mm'],
            sample_mean_R_F6_macro_rmse_mm=metrics['sample_mean']['horizon_macro_rmse_mm'][::6],
            coverage80=metrics['coverage80_by_horizon'],rank3_30_coverage=metrics['rank3_30_coverage'],
            NFE=lineage['neural_evaluations_per_sample'],members=lineage['members'],
            R_delta_ci95_mm=score_interval(scores[label],scores['reference'],0),
            F6_delta_ci95_mm=score_interval(scores[label],scores['reference'],6),
            increment_delta_ci95_mm=scalar_interval(*increments[label],*increments['reference']),
            source_prediction_sha256=lineage['prediction_sha256'])
        if 'analytic_gaussian_crps_mm_by_horizon' in metrics:
            row['analytic_gaussian_crps_mm']=metrics['analytic_gaussian_crps_mm_by_horizon']
            row['analytic_gaussian_80_coverage']=metrics['analytic_gaussian_80_coverage']
            row['analytic_gaussian_mean_R_F6_rmse_mm']=metrics['gaussian_distribution_mean']['horizon_macro_rmse_mm'][::6]
        mode='graph_affine' if label.startswith('graph_affine_') else lineage['mode']
        if mode!='reference':
            protocol=read(root/mode/'protocol.json');row['parameters']=protocol['parameters']
            row['training_seconds']=read(root/mode/'status.json')['seconds']
            if (root/mode/'query_audit.json').exists():query_audits[mode]=read(root/mode/'query_audit.json')
        else:row['parameters']=0
        rows.append(row)
    comparisons={}
    for label in ('local_heun0','full_heun0'):
        comparisons[label+'_vs_affine_heun0']=dict(R_delta_ci95_mm=score_interval(scores[label],scores['affine_heun0'],0),
            F6_delta_ci95_mm=score_interval(scores[label],scores['affine_heun0'],6),
            increment_delta_ci95_mm=scalar_interval(*increments[label],*increments['affine_heun0']))
    if 'graph_affine_heun0' in scores:
        comparisons['full_heun0_vs_graph_affine_heun0']=dict(
            R_delta_ci95_mm=score_interval(scores['full_heun0'],scores['graph_affine_heun0'],0),
            F6_delta_ci95_mm=score_interval(scores['full_heun0'],scores['graph_affine_heun0'],6),
            increment_delta_ci95_mm=scalar_interval(*increments['full_heun0'],*increments['graph_affine_heun0']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    primary=[r for r in rows if r['name']=='reference' or r['name'].endswith('heun0')]
    x=np.arange(len(primary));ref=rows[0]
    for h,offset,color in [(0,-.15,'#407b76'),(6,.15,'#b97a4d')]:
        gain=[100*(ref['fair_crps_mm'][h]-r['fair_crps_mm'][h])/ref['fair_crps_mm'][h] for r in primary]
        axes[0].bar(x+offset,gain,width=.28,label=f'h={h}',color=color)
    axes[0].axhline(0,color='#666',lw=.8);axes[0].axhline(5,color='#999',ls='--',label='5% target')
    axes[0].set(xticks=x,xticklabels=[r['name'].removesuffix('_heun0') for r in primary],ylabel='Fair CRPS improvement over GP reference (%)');axes[0].legend(fontsize=8)
    for row in primary:axes[1].plot(range(7),row['rank3_30_coverage'],'o-',label=row['name'])
    axes[1].axhline(27/33,color='#888',ls='--',label='K32 IID rank reference:81.82%')
    axes[1].set(xlabel='Lead (h)',ylabel='Rank3/30 interval coverage');axes[1].legend(fontsize=7)
    for mode,audit in query_audits.items():
        y=[max(r['path_rms_difference_mm'] for r in audit['rows'] if r['query_count']==n) for n in (1,64,128,137)]
        axes[2].plot([1,64,128,137],np.maximum(y,1e-8),'o-',label=mode)
    axes[2].set(xlabel='Query population',ylabel='Common GP path change RMS (mm)',yscale='log',title='Fixed visible information; coupled-path diagnostic')
    if query_audits:axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    fig.suptitle(f'Seed{args.seed}: same frozen mean, S175 condition and Gaussian-process reference')
    fig.savefig(out/'gp_process_comparison.png',dpi=170);plt.close(fig)
    payload=dict(seed=args.seed,rows=rows,nonlinear_vs_affine=comparisons,query_audits=query_audits,
        source=source,interval='2000 shared-month draws; exposed development, exploratory comparison',
        strict_oof=False,confirmation_C_D_read=False,code_sha256=sha(Path(__file__)),figure_sha256=sha(out/'gp_process_comparison.png'))
    write_json(out/'process_summary.json',payload)
    lines=['# GP参考轨迹扩散：机制与概率结果','',f'seed{args.seed}；同冻结TF3000均值、S175条件、训练Gaussian/copula。每新模型3000连续块。2020为已暴露开发日历，非完整年度或严格OOF。','',
        '| 模型 | R fair CRPS/mm | F6 fair CRPS/mm | 预报内部增量CRPS/mm | NFE | 新增参数 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['name']} | {r['fair_crps_mm'][0]:.6f} | {r['fair_crps_mm'][6]:.6f} | {r['increment_fair_crps_mm']:.6f} | {r['NFE']} | {r['parameters']} |")
    lines+=['','R/F6为K32 IID成员fair CRPS，预报内部增量按同起点0–6h轨迹计算。Gaussian/Affine解析均值和解析CRPS在JSON单列；K32样本均值含有限成员误差。第3/30顺序统计量覆盖的连续校准IID参考为81.818%，不同于解析80%区间。','',
        '| 比较 | R差值95%区间/mm | F6差值95%区间/mm | 增量差值95%区间/mm |','|---|---|---|---|']
    for name,c in comparisons.items():lines.append(f"| {name} | {c['R_delta_ci95_mm']} | {c['F6_delta_ci95_mm']} | {c['increment_delta_ci95_mm']} |")
    lines+=['','Local/Affine逐查询运算，空间相关来自GP初值，表示GP噪声耦合的逐站轨迹过程。Full读取查询图；共同GP路径随查询集合变化的差异本身不等同于边际分布差异。','',
        '![概率性能、覆盖及查询集合诊断](gp_process_comparison.png)','','完整区间、来源哈希、解析Gaussian控制与查询路径结果见[process_summary.json](process_summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(seed=args.seed,rows=[{k:r[k] for k in ('name','fair_crps_mm','increment_fair_crps_mm')} for r in rows],comparisons=comparisons)))


if __name__=='__main__':main()
