"""Matched precision report; only completed final3000-step fits are ranked."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.p14_fusion_experiment import reconstruction_scores
from tools.terrain_diffusion.data import sha,write_json


def macro_from_error(e):
    count=np.isfinite(e).sum((1,2));good=count>0
    values=np.sqrt(np.divide(np.nansum(e*e,(1,2)),count,out=np.full(count.shape,np.nan),where=good))
    return float(np.nanmean(values))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'));paired=[];checks=[];rng=np.random.default_rng(252511)
    base={};old={};traces={}
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    for row in summary:
        folder=run/f"{row['case']}_s{row['seed']}";a=dict(np.load(folder/'evaluation.npz'))
        actual=reconstruction_scores(a['prediction'],a['truth'])
        error=max(abs(actual[k]-row['metrics'][k]) for k in ('macro_rmse_mm','micro_rmse_mm','increment_macro_rmse_mm'))
        assert error<1e-5
        checks.append(dict(case=row['case'],seed=row['seed'],max_metric_difference_mm=error,sha256=sha(folder/'evaluation.npz')))
        if row['case']=='reference':
            base[row['seed']]=a
            source=Path(protocol['config']['source'])/f"{protocol['config']['source_kind']}_s{row['seed']}"/'evaluation.npz'
            b=dict(np.load(source));assert np.allclose(a['truth'],b['truth'],equal_nan=True)
            old[row['seed']]=float(np.max(np.abs(a['prediction']-b['prediction'])))
    for row in summary:
        if row['case']=='reference':continue
        seed=row['seed'];folder=run/f"{row['case']}_s{seed}";a=dict(np.load(folder/'evaluation.npz'));b=base[seed]
        assert np.allclose(a['truth'],b['truth'],equal_nan=True)
        e=a['prediction']-a['truth'][None];r=b['prediction']-b['truth'][None]
        difference=row['metrics']['macro_rmse_mm']-next(x['metrics']['macro_rmse_mm'] for x in summary if x['case']=='reference' and x['seed']==seed)
        draws=[]
        for _ in range(1000):
            months=rng.integers(0,12,12);draws.append(macro_from_error(e[:,months])-macro_from_error(r[:,months]))
        paired.append(dict(case=row['case'],seed=seed,delta_macro_rmse_mm=difference,
            monthly_paired_bootstrap95_mm=np.quantile(draws,[.025,.975]).tolist(),
            delta_micro_rmse_mm=row['metrics']['micro_rmse_mm']-next(x['metrics']['micro_rmse_mm'] for x in summary if x['case']=='reference' and x['seed']==seed),
            cohort='same2020monthly72h x3layouts; development-exposed',resampling='12 month blocks paired over layouts/hours/stations;1000replicates'))
        fit=json.loads((folder/'fit.json').read_text(encoding='utf-8'));traces[f"{row['case']}_s{seed}"]=fit['trace']
    means=[]
    for case in protocol['config']['cases']:
        rows=[r for r in paired if r['case']==case]
        if len(rows)==len(protocol['config']['seeds']):
            means.append(dict(case=case,seed_mean_delta_macro_rmse_mm=float(np.mean([r['delta_macro_rmse_mm'] for r in rows])),
                improves_each_seed=all(r['delta_macro_rmse_mm']<0 for r in rows)))
    write_json(run/'paired_analysis.json',dict(paired=paired,seed_means=means,reference_replication_max_mm=old))
    write_json(run/'verification.json',dict(checks=checks,all_final_metrics_recomputed=True,source_replication=old))
    if traces:
        fig,ax=plt.subplots(figsize=(10,4))
        for label,trace in traces.items():ax.plot([r['step'] for r in trace],[r['batch_rmse_mm'] for r in trace],label=label)
        ax.set(xlabel='Updates (consecutive12h blocks)',ylabel='Varying training-block RMSE (mm)',title='Matched ~10mm warm refinement; training weather varies');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(run/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# 既有约10mm模型上的正式配对精度优化','',
        '参考保留原GRU+p14全部分支与归一化。所有候选训练3000步，2016–2019连续12小时批、仅条件站抽样；原2020每月72小时×3布局×137登记开发站评价。该开发队列已暴露。','',
        '| 配置 | 种子 | 步数 | 站宏RMSE/mm | 微RMSE/mm | 增量宏RMSE/mm |','|---|---:|---:|---:|---:|---:|']
    for row in summary:
        m=row['metrics'];lines.append(f"| {row['case']} | {row['seed']} | {row['steps']} | {m['macro_rmse_mm']:.6f} | {m['micro_rmse_mm']:.6f} | {m['increment_macro_rmse_mm']:.6f} |")
    lines+=['','仅列完成的最终3000步结果。尚在运行的配置不参与排名。','',
        '## 与同种子参考的配对差值','', '| 配置 | 种子 | 宏RMSE变化/mm | 配对月份bootstrap95%区间/mm |','|---|---:|---:|---|']
    for r in paired:lines.append(f"| {r['case']} | {r['seed']} | {r['delta_macro_rmse_mm']:.6f} | [{r['monthly_paired_bootstrap95_mm'][0]:.6f}, {r['monthly_paired_bootstrap95_mm'][1]:.6f}] |")
    lines+=['','差值为新模型减参考，负数代表改善。月份重采样保留同一天、布局和站点的配对，12个时间块给出的区间反映此开发样本的不确定性，不替代新年份评价。','',
        '[训练与结构合同](../../../../../../../docs/superpowers/plans/2026-11-25-warm-joint-precision.md)',
        '[实际架构图](../../../../../../../docs/figures/joint_spectral_implementation_20261125/warm_10mm.png)']
    if traces:lines+=['','![训练曲线](learning_curves.png)']
    (run/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(completed=len(summary),comparisons=len(paired),means=means)))


if __name__=='__main__':main()
