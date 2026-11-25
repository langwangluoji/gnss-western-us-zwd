"""Matched information/capacity controls with calendar-paired fair-score intervals."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def score_interval(a,b,horizon,draws=2000):
    aa=a[...,horizon];bb=b[...,horizon];axes=(0,2,3)
    asum=np.nansum(aa,axes);ac=np.isfinite(aa).sum(axes)
    bsum=np.nansum(bb,axes);bc=np.isfinite(bb).sum(axes)
    rng=np.random.default_rng(261126);delta=[]
    for _ in range(draws):
        index=rng.integers(len(asum),size=len(asum))
        delta.append(asum[index].sum()/ac[index].sum()-bsum[index].sum()/bc[index].sum())
    return np.quantile(delta,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(ENGINE))
    from tools.terrain_diffusion.native_paths import file_path
    from tools.terrain_diffusion.data import sha,write_json
    results=file_path(ENGINE/'results');root=results/'20261126_probability_context'
    source=[('L64',results/f'20261126_native_weather_probability/transformer72_physics_aux_last_s{args.seed}/gaussian')]
    source += [(name,root/f'{name}_s{args.seed}') for name in ('R111','S175','Lwide')]
    rows=[];scores={};common=None
    for name,folder in source:
        evaluation=folder/'eval_ddim';metrics=json.loads((evaluation/'metrics.json').read_text())
        calibration=json.loads((evaluation/'calibration_audit.json').read_text())
        protocol=json.loads((folder/'protocol.json').read_text());lineage=json.loads((evaluation/'lineage.json').read_text())
        if common is None:common=lineage['cache_manifest_sha256']
        if common!=lineage['cache_manifest_sha256']:raise ValueError('control calendars differ')
        if sha(evaluation/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('control artifact changed')
        with np.load(evaluation/'evaluation.npz') as data:scores[name]=data['crps'].astype('float64')
        np.testing.assert_allclose(np.nanmean(scores[name],axis=(0,1,2,3)),metrics['fair_crps_mm_by_horizon'],rtol=0,atol=1e-7)
        if name!='L64':np.testing.assert_array_equal(np.isfinite(scores[name]),np.isfinite(scores['L64']))
        row=dict(name=name,parameters=protocol['parameters'],condition_view=protocol['config'].get('condition_view','latent'),
            width=protocol['config'].get('width',64),analytic_crps_mm=metrics['analytic_gaussian_crps_mm_by_horizon'],
            fair_crps_mm=metrics['fair_crps_mm_by_horizon'],distribution_mean_rmse_mm=metrics['gaussian_distribution_mean']['horizon_macro_rmse_mm'],
            analytic_coverage80=calibration['analytic_gaussian_80']['coverage_by_horizon'],distance_calibration=calibration['distance_calibration'],
            paired_R_fair_delta_ci95_mm=score_interval(scores[name],scores['L64'],0),
            paired_F6_fair_delta_ci95_mm=score_interval(scores[name],scores['L64'],6),
            weights_sha256=lineage['probability_weights_sha256'],evaluation_sha256=lineage['prediction_sha256'])
        rows.append(row)
    skip_wide=dict(R_delta_ci95_mm=score_interval(scores['S175'],scores['Lwide'],0),
        F6_delta_ci95_mm=score_interval(scores['S175'],scores['Lwide'],6))
    args.out=file_path(args.out);args.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    x=np.arange(len(rows))
    for offset,h,color in [(-.12,0,'#35836c'),(.12,6,'#c4875d')]:
        value=np.array([r['fair_crps_mm'][h]-rows[0]['fair_crps_mm'][h] for r in rows])
        cis=np.array([r['paired_R_fair_delta_ci95_mm'] if h==0 else r['paired_F6_fair_delta_ci95_mm'] for r in rows])
        axes[0].errorbar(x+offset,value,yerr=np.maximum(np.stack([value-cis[:,0],cis[:,1]-value]),0),fmt='o',capsize=3,color=color,label=f'h={h}')
        axes[1].plot(x,[r['analytic_coverage80'][h] for r in rows],'o-',color=color,label=f'h={h}')
    axes[0].axhline(0,color='#777',ls='--');axes[0].set(xticks=x,xticklabels=[r['name'] for r in rows],ylabel='K32 fair CRPS change vs L64 (mm)',title='Paired 12-month exploratory intervals')
    axes[1].axhline(.8,color='#777',ls='--');axes[1].set(xticks=x,xticklabels=[r['name'] for r in rows],ylabel='Analytic Gaussian coverage',title='Population 80% intervals')
    for row in rows:axes[2].plot(range(5),[r['h0']['forecast_std_rms_mm'] for r in row['distance_calibration']],'o-',label=row['name'])
    axes[2].plot(range(5),[r['h0']['error_rmse_mm'] for r in rows[0]['distance_calibration']],'k--',label='L64 observed error RMS')
    axes[2].set(xticks=range(5),xticklabels=['<50','50-100','100-150','150-300','>=300'],xlabel='Nearest visible distance (km)',ylabel='Current sigma / error RMS (mm)',title='Geometry sensitivity')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.suptitle(f'Seed {args.seed}: information vs matched capacity; same frozen causal mean and residuals')
    fig.savefig(args.out/'context_controls.png',dpi=170);plt.close(fig)
    payload=dict(seed=args.seed,rows=rows,skip_vs_matched_capacity=skip_wide,
        inference='Shared month bootstrap,2000 draws; exposed development, exploratory selection, no multiplicity correction. Analytic descriptive scores and K32 fair-score intervals reported separately.',
        strict_oof=False,confirmation_C_D_read=False,code_sha256=sha(Path(__file__)),figure_sha256=sha(args.out/'context_controls.png'))
    write_json(args.out/'context_summary.json',payload)
    text=['# 概率条件的信息直连与容量对照','',f'seed{args.seed}，各3000连续块；L64复用原Gaussian。均值源相同，概率头另有条件均值修正。2020主日历为已暴露开发评价。','',
        '| 条件 | 参数 | R解析CRPS/mm | F6解析CRPS/mm | R分布均值RMSE/mm | F6分布均值RMSE/mm | R解析80%覆盖 | F6解析80%覆盖 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        r=row
        text.append(f"| {r['name']} | {r['parameters']} | {r['analytic_crps_mm'][0]:.6f} | {r['analytic_crps_mm'][6]:.6f} | {r['distribution_mean_rmse_mm'][0]:.6f} | {r['distribution_mean_rmse_mm'][6]:.6f} | {r['analytic_coverage80'][0]:.2%} | {r['analytic_coverage80'][6]:.2%} |")
    text+=['','S175与Lwide分别为12174/12180参数。其R的K32 fair CRPS差值95%月份配对区间为'+str(skip_wide['R_delta_ci95_mm'])+' mm，F6为'+str(skip_wide['F6_delta_ci95_mm'])+' mm。区间保留每个月实际有效标签数；它衡量本日历的开发比较，不代表完整年度或盲测。','',
        '直接信息通道主要改善当前场概率校准；全局CRPS收益尚未达到预先约定的5%门槛，未来预报分开判断。150–300km仍有解析覆盖缺口，不能把条件向量增加视为瓶颈已经解决。','',
        '![信息和容量对照](context_controls.png)','','全部时效、来源/权重哈希、距离尺度与配对区间见[context_summary.json](context_summary.json)。']
    (args.out/'REPORT.md').write_bytes(('\n'.join(text)+'\n').encode('utf-8'))
    print(json.dumps(dict(seed=args.seed,skip_vs_matched_capacity=skip_wide)))


if __name__=='__main__':main()
