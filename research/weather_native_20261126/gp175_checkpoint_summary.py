"""Interim two-seed G175 result, without waiting or selecting the G265 arm."""
import json
import sys
from pathlib import Path
import numpy as np
from campaign import ROOT,RESULTS,read
from weather_gp_summary import interval

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
from tools.terrain_diffusion.native_weather_gp_experiment import codes


def main():
    root=RESULTS/'20261126_weather_projective_gp';out=root/'g175_checkpoint';out.mkdir(parents=True,exist_ok=True)
    records={};sources={};energies={}
    for seed in (42,17):
        folder=root/f's{seed}/c175';seal=validate_model_seal(folder,'COMPLETE_WEATHER_GP',codes())
        evaluation_marker(folder/'eval',seal)
        ref=root/f's{seed}/reference';new=folder/'eval';full=RESULTS/'20261126_gp_process'/f's{seed}/full/eval_heun0'
        metrics=read(new/'metrics.json');reference=read(ref/'metrics.json');joint=read(full/'joint_metrics.json')
        for p in (ref,new,full):
            lineage=read(p/'lineage.json')
            if lineage['source']!=seal['source'] or sha(p/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('interim sources differ')
            sources[f's{seed}/{p.parent.name}/{p.name}']=dict(lineage_sha256=sha(p/'lineage.json'),array_sha256=lineage['prediction_sha256'])
        with np.load(ref/'evaluation.npz') as a,np.load(new/'evaluation.npz') as b:
            for k in ('truth','prediction','gaussian_distribution_mean','gaussian_distribution_std'):np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_allclose(metrics['analytic_gaussian_crps_mm_by_horizon'],reference['analytic_gaussian_crps_mm_by_horizon'],atol=0,rtol=0)
        records[seed]=dict(candidate=metrics,reference=reference,full=joint,status=read(folder/'status.json'),seal=seal)
        for label,p in [('reference',ref),('g175',new)]:
            if (p/'joint_metrics.json').exists():energies[f's{seed}/{label}']=read(p/'joint_metrics.json')['energy']['mean_mm']
    contrasts={}
    for comparison in ('g175_vs_reference','g175_vs_full'):
        contrasts[comparison]={}
        for distance in ('0-50','50-150','150-300','300-inf'):
            contrasts[comparison][distance]={}
            for lead in ('h0','h6'):
                pairs=[];x=[];y=[]
                for seed in (42,17):
                    a=records[seed]['candidate']['analytic_station_pair_crps'][distance][lead]
                    b=(records[seed]['reference']['analytic_station_pair_crps'][distance][lead] if comparison=='g175_vs_reference' else
                        records[seed]['full']['spatial_pair_crps'][distance][lead])
                    pairs.append((a,b));x.append(a['analytic_crps_mm']);y.append(b['analytic_crps_mm'] if comparison=='g175_vs_reference' else b['fair_crps_mm'])
                candidate=float(np.mean(x));reference=float(np.mean(y))
                contrasts[comparison][distance][lead]=dict(candidate_mm=candidate,reference_mm=reference,gain_percent=100*(reference-candidate)/reference,delta_ci95_mm=interval(pairs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');distances=list(contrasts['g175_vs_reference']);x=np.arange(4)
    for index,comparison in enumerate(contrasts):
        for lead,offset,color in [('h0',-.16,'#b9874c'),('h6',.16,'#367f83')]:
            axes[index].bar(x+offset,[contrasts[comparison][d][lead]['gain_percent'] for d in distances],width=.3,label=lead,color=color)
        axes[index].set(xticks=x,xticklabels=distances,xlabel='Station-pair distance (km)',ylabel='Difference CRPS improvement (%)',title=comparison)
        axes[index].axhline(0,color='#555',lw=.8);axes[index].spines[['top','right']].set_visible(False);axes[index].grid(alpha=.2);axes[index].legend()
    fig.suptitle('G175: two seeds, fixed mean and standard deviation; G265 still in progress')
    fig.savefig(out/'g175_spatial_gain.png',dpi=170);plt.close(fig)
    payload=dict(status='INTERIM_G175_COMPLETE_G265_PENDING',records=records,contrasts=contrasts,energies_if_complete=energies,sources=sources,
        exact_reference_mean_std_and_prediction_verified=True,code_sha256=sha(__file__),strict_oof=False,confirmation_C_D_read=False,
        full_comparison='analytic Gaussian CRPS vs IID K32 fair CRPS; same target and source, finite-member variance retained for Full',
        interval='2000 common-month draws shared across seeds; exposed2020 exploratory, no multiplicity correction')
    write_json(out/'g175_checkpoint.json',payload)
    lines=['# G175条件相关核：双种子阶段结果','','G175各3000连续块已经完成；G265及其预注册Local/Affine连接仍在运行，本报告不替代完整四臂结果。2020每月8–10日×三布局×137查询为已暴露开发。','',
        '## 核心结果','','逐值核验两个种子的基础场预测、Gaussian解析均值和标准差与各自参考完全相同；解析边际CRPS也完全相同。下表仅反映空间依赖变化。','',
        '| 距离/km | 当前差值CRPS改善 | F6差值CRPS改善 | F6差值95%区间/mm |','|---|---:|---:|---|']
    for distance,row in contrasts['g175_vs_reference'].items():lines.append(f"| {distance} | {row['h0']['gain_percent']:.2f}% | {row['h6']['gain_percent']:.2f}% | {row['h6']['delta_ci95_mm']} |")
    lines+=['','seed42/17近邻F6解析差值CRPS分别4.139470/4.115608mm，相对各自参考改善21.13%/19.42%。当前近邻也改善，≥300km基本持平。G175无需改动已验证均值即可学习这项联合收益；它是条件Gaussian核，不是新增扩散收益。','',
        '## 相对原Full的同来源比较','','| 距离/km | F6改善 | F6差值95%区间/mm |','|---|---:|---|']
    for distance,row in contrasts['g175_vs_full'].items():lines.append(f"| {distance} | {row['h6']['gain_percent']:.2f}% | {row['h6']['delta_ci95_mm']} |")
    lines+=['','此项使用Gaussian解析CRPS与Full的K32 IID fair CRPS比较，二者估计同一分布评分，Full含有限成员波动。它比较整套方法，不将所有差异单独归因为一个参数。全场energy来源完整时在JSON列出；尚未完成的项不补造。','',
        '![空间联合收益](g175_spatial_gain.png)','','完整来源和各lead结果见[g175_checkpoint.json](g175_checkpoint.json)。G175/G265额外日期政策在读新增日期成绩前另行锁定，属于冻结补充开发验证。','',
        'G175仍对959维完整时空矩阵分解，本报告不据网络调用次数推断端到端推断更快。固定G265的预注册扩散连接政策保持原样，G175的阶段成绩不回改旧实验锁。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'));print(json.dumps(contrasts))


if __name__=='__main__':main()
