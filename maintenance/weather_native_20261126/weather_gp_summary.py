"""Analytic matched-marginal and covariance evidence with shared-month intervals."""
import json
import sys
from pathlib import Path
import numpy as np
from campaign import RESULTS,ROOT,read

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json


def interval(pairs):
    rng=np.random.default_rng(261126);values=[]
    for _ in range(2000):
        months=rng.integers(12,size=12);delta=[]
        for a,b in pairs:
            def value(r):
                s=np.asarray(r['monthly_score_sum']);n=np.asarray(r['monthly_valid_count'])
                return s[months].sum()/n[months].sum()
            delta.append(value(a)-value(b))
        values.append(np.mean(delta))
    return np.quantile(values,[.025,.975]).tolist()


def main():
    root=RESULTS/'20261126_weather_projective_gp';out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    records={};rows=[];source_hashes={}
    for seed in (42,17):
        baseline=None;basemetrics=None
        for mode in ('reference','c175','c265'):
            folder=root/f's{seed}'/mode
            if mode!='reference':folder=folder/'eval'
            metrics=read(folder/'metrics.json');joint=read(folder/'joint_metrics.json');lineage=read(folder/'lineage.json')
            if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('saved Gaussian arrays changed')
            with np.load(folder/'evaluation.npz') as data:
                arrays={k:data[k] for k in ('truth','prediction','gaussian_distribution_mean','gaussian_distribution_std')}
            if baseline is None:baseline=arrays;basemetrics=metrics
            else:
                for name in arrays:np.testing.assert_array_equal(arrays[name],baseline[name])
                np.testing.assert_allclose(metrics['analytic_gaussian_crps_mm_by_horizon'],basemetrics['analytic_gaussian_crps_mm_by_horizon'],atol=0,rtol=0)
            record=dict(seed=seed,mode=mode,analytic_pairs=metrics['analytic_station_pair_crps'],
                analytic_increment_crps_mm=metrics['analytic_forecast_increment_crps_mm'],
                analytic_marginal_crps_mm=metrics['analytic_gaussian_crps_mm_by_horizon'],
                deterministic_R_F6_rmse_mm=metrics['deterministic_mean']['horizon_macro_rmse_mm'][::6],
                gaussian_mean_R_F6_rmse_mm=metrics['gaussian_distribution_mean']['horizon_macro_rmse_mm'][::6],
                energy=joint['energy'],sample_crps_mm=metrics['fair_crps_mm_by_horizon'],
                analytic_80coverage=metrics['analytic_gaussian_80_coverage'],source=lineage['source'])
            if mode!='reference':
                protocol=read(folder.parent/'protocol.json');status=read(folder.parent/'status.json')
                if status['status']!='COMPLETE_WEATHER_GP' or status['steps']!=3000:raise ValueError('incomplete GP')
                record.update(parameters=protocol['parameters'],training_seconds=status['seconds'])
            records[seed,mode]=record;rows.append(record)
            source_hashes[str(folder)]=dict(lineage_sha256=sha(folder/'lineage.json'),metrics_sha256=sha(folder/'metrics.json'),joint_sha256=sha(folder/'joint_metrics.json'))
    comparisons={}
    for candidate,reference in (('c175','reference'),('c265','reference'),('c265','c175')):
        distances={}
        for distance in ('0-50','50-150','150-300','300-inf'):
            distances[distance]={}
            for lead in ('h0','h6'):
                pairs=[(records[s,candidate]['analytic_pairs'][distance][lead],records[s,reference]['analytic_pairs'][distance][lead]) for s in (42,17)]
                a=float(np.mean([p[0]['analytic_crps_mm'] for p in pairs]));b=float(np.mean([p[1]['analytic_crps_mm'] for p in pairs]))
                distances[distance][lead]=dict(candidate_mm=a,reference_mm=b,gain_percent=100*(b-a)/b,delta_ci95_mm=interval(pairs))
        energy=np.mean([records[s,candidate]['energy']['mean_mm']-records[s,reference]['energy']['mean_mm'] for s in (42,17)])
        rng=np.random.default_rng(261126);monthly=np.mean([np.asarray(records[s,candidate]['energy']['monthly_mean_mm'])-records[s,reference]['energy']['monthly_mean_mm'] for s in (42,17)],axis=0)
        energy_ci=np.quantile(monthly[rng.integers(12,size=(2000,12))].mean(-1),[.025,.975]).tolist()
        comparisons[candidate+'_vs_'+reference]=dict(distances=distances,energy_delta_mm=float(energy),energy_delta_ci95_mm=energy_ci)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained');colors={'reference':'#8b939b','c175':'#c38947','c265':'#367f83'}
    distance_labels=['0-50','50-150','150-300','300-inf'];x=np.arange(4)
    for k,mode in enumerate(('reference','c175','c265')):
        pair=[np.mean([records[s,mode]['analytic_pairs'][d]['h6']['analytic_crps_mm'] for s in (42,17)]) for d in distance_labels]
        axes[0].bar(x+(k-1)*.25,pair,.23,label=mode,color=colors[mode])
        axes[1].plot(range(7),np.mean([records[s,mode]['analytic_marginal_crps_mm'] for s in (42,17)],0),'o-',label=mode,color=colors[mode],alpha=.8)
        for seed,style in ((42,'-'),(17,'--')):
            if mode=='reference':continue
            curve=read(root/f's{seed}'/mode/'curve.json');axes[2].plot([v['step'] for v in curve],[v['loss'] for v in curve],style,label=f'{mode},s{seed}',color=colors[mode],alpha=.8)
    axes[0].set(xticks=x,xticklabels=distance_labels,xlabel='Station-pair distance (km)',ylabel='Analytic F6 difference CRPS (mm)');axes[0].legend()
    axes[1].set(xlabel='Forecast lead (h)',ylabel='Analytic marginal CRPS (mm)',title='Frozen marginals: curves coincide');axes[1].legend()
    axes[2].set(xlabel='Consecutive training block',ylabel='Per-coordinate Gaussian NLL (constant omitted)');axes[2].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.savefig(out/'weather_gp_comparison.png',dpi=170);plt.close(fig)
    payload=dict(rows=rows,comparisons=comparisons,source_hashes=source_hashes,code_sha256=sha(__file__),
        interval='2000 shared-month draws, shared across both seeds; exposed2020 exploratory intervals, no multiplicity correction',
        exact_frozen_marginal_arrays_verified=True,strict_oof=False,confirmation_C_D_read=False)
    write_json(out/'weather_gp_summary.json',payload)
    lines=['# 天气条件时空Gaussian核：双种子结果','','2016–2019各3000连续12h块，仅抽站身份；2020每月8–10日、三布局、137查询，已暴露开发。冻结TF3000均值、S175边际尺度。G175与G265分别43893/44027参数。','',
        '| 模型 | 0–50km F6解析差值CRPS/mm | 内部增量解析CRPS/mm | K32 fair energy/mm |', '|---|---:|---:|---:|']
    for mode in ('reference','c175','c265'):
        v=[records[s,mode] for s in (42,17)]
        lines.append(f"| {mode} | {np.mean([r['analytic_pairs']['0-50']['h6']['analytic_crps_mm'] for r in v]):.6f} | {np.mean([r['analytic_increment_crps_mm'] for r in v]):.6f} | {np.mean([r['energy']['mean_mm'] for r in v]):.6f} |")
    lines+=['','| 比较 | 距离/km | F6差值CRPS改善 | 差值95%区间/mm |','|---|---|---:|---|']
    for label,c in comparisons.items():
        for d,r in c['distances'].items():
            h=r['h6'];lines.append(f"| {label} | {d} | {h['gain_percent']:.2f}% | {h['delta_ci95_mm']} |")
    lines+=['','边际均值、标准差、truth与确定性预测逐值相同，解析边际CRPS精确相同；K32边际采样分数的差异来自有限成员。上表站对与增量使用Gaussian解析分布，energy使用32个IID成员。共同月份区间为探索性开发比较。','',
        '![双种子Gaussian相关核结果与训练曲线](weather_gp_comparison.png)','','完整来源、能量评分区间和h0结果见[weather_gp_summary.json](weather_gp_summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(comparisons))


if __name__=='__main__':main()
