"""Summarize locked K16 raw/disjoint-pair fields without selecting member pairs."""
import json
from pathlib import Path
import numpy as np
from campaign import RESULTS,read
from weather_controls_summary import paired_record
from tools.terrain_diffusion.data import sha,write_json


def main():
    root=RESULTS/'20261126_independent_pair_centering';lockpath=Path(__file__).with_name('independent_pair_centering_lock.json');lock=read(lockpath)
    records={};rows=[];sources={};code=Path(__file__).with_name('independent_pair_centering.py')
    for seed in lock['seeds']:
        base=None
        for model in lock['models']:
            for variant in lock['families']:
                path=root/f's{seed}'/model/variant/'metrics.json';row=read(path)
                if row['lock_sha256']!=sha(lockpath) or row['code_sha256']!=sha(code):raise ValueError('pair diagnostic source changed')
                for name,value in row['dependency_sha256'].items():
                    if value!=sha(Path(__file__).with_name(name)):raise ValueError('pair diagnostic helper changed')
                if row['source']!=lock['sources'][f's{seed}/{model}'] or row['members']!=16 or row['seed']!=seed:
                    raise ValueError('pair diagnostic recipe changed')
                if row['model']!=model or row['variant']!=variant or row['analytic_mean_is_base']!=(variant!='raw_first16'):
                    raise ValueError('pair diagnostic mode changed')
                if base is None:base=row['base_point_metrics']
                elif base!=row['base_point_metrics']:raise ValueError('base prediction population differs')
                rows.append(row);records[seed,model,variant]=row['records'];sources[f's{seed}/{model}/{variant}']=sha(path)
    comparisons={};edges=[]
    raw='raw_first16';centered='independent_pair_centered16'
    for variant in lock['families']:
        for solver in ['heun0','ddim']:
            edges += [(('local_'+solver,variant),('affine_'+solver,variant)),
                (('local_'+solver,variant),('reference',variant)),(('affine_'+solver,variant),('reference',variant))]
    edges += [((model,centered),(model,raw)) for model in lock['models']]
    for (a,av),(b,bv) in edges:
        keys=set(records[42,a,av])&set(records[17,a,av])&set(records[42,b,bv])&set(records[17,b,bv])
        name=f'{a}/{av}_vs_{b}/{bv}'
        comparisons[name]={key:paired_record([(records[s,a,av][key],records[s,b,bv][key]) for s in lock['seeds']]) for key in sorted(keys)}
    mean_gaps={f'{m}/{v}':{f'h{h}':paired_record([(records[s,m,v][f'fair_mean_mse_h{h}'],records[s,m,v][f'base_mse_h{h}']) for s in lock['seeds']]) for h in [0,6]} for m in lock['models'] for v in lock['families']}
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,comparisons=comparisons,fair_mean_MSE_gap_mm2=mean_gaps,sources=sources,
        code_sha256=sha(__file__),lock_sha256=sha(lockpath),transform_sha256=sha(code),
        pairing='(k,k+16), no overlapping pairs, fixed before score reads',
        strict_oof=False,confirmation_C_D_read=False,
        interval='2000 shared-month draws, two fixed seeds, fixed K16 noise; exposed2020 exploratory, no multiplicity correction'))
    render(out,rows,comparisons,mean_gaps)


def render(out,rows,comparisons,mean_gaps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw='raw_first16';centered='independent_pair_centered16';primary=['reference','local_heun0','affine_heun0']
    def value(model,variant,key):return np.mean([r['records'][key]['value'] for r in rows if r['model']==model and r['variant']==variant])
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for model in primary:
        for variant,style in [(raw,'--'),(centered,'-')]:
            axes[0].plot(range(7),[value(model,variant,f'crps_h{h}') for h in range(7)],style,label=model+('/raw' if variant==raw else '/center'))
        axes[1].plot(range(4),[value(model,centered,f'pair_{d}_h6') for d in ['0-50','50-150','150-300','300-inf']],'o-',label=model)
    for i,variant in enumerate([raw,centered]):
        label=f'local_heun0/{variant}_vs_affine_heun0/{variant}';table=comparisons[label]
        keys=['crps_h6','increment','pair_0-50_h6','energy'];v=np.array([table[k]['delta'] for k in keys]);ci=np.array([table[k]['delta_ci95'] for k in keys])
        axes[2].errorbar(np.arange(4)+(i-.5)*.13,v,yerr=np.maximum(np.stack([v-ci[:,0],ci[:,1]-v]),0),fmt='o',capsize=3,label='raw' if i==0 else 'pair-centered')
    axes[0].set(xlabel='Lead (h)',ylabel='K16 fair CRPS (mm)');axes[0].legend(fontsize=6)
    axes[1].set(xticks=range(4),xticklabels=['0-50','50-150','150-300','300+'],xlabel='Pair distance (km)',ylabel='Centered F6 pair CRPS (mm)');axes[1].legend(fontsize=7)
    axes[2].axhline(0,color='gray',ls='--');axes[2].set(xticks=range(4),xticklabels=['F6','increment','near pair','energy'],ylabel='Local minus Affine score (mm)');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'pair_centering.png',dpi=170);plt.close(fig)
    lines=['# 独立双场差分：严格均值保持诊断','',
        '冻结原32个IID成员，按事先锁定(k,k+16)配成16个互不重叠的独立输出成员。原模型前16成员为匹配K16描述对照。所有模型均执行相同变换；原模型和权重未改写。',
        'Y=mu_base+(X-X\')/sqrt(2)具有解析均值mu_base并保持原条件总体协方差；有限样本协方差不要求逐值相等。分布被对称化；四阶矩存在时，四阶累积量减半，不能认为原高阶分布保持不变。','',
        '| 模型 | 输出 | 当前CRPS/mm | F6 CRPS/mm | 增量CRPS/mm | 近邻F6差值CRPS/mm | energy/mm |','|---|---|---:|---:|---:|---:|---:|']
    for model in ['reference','local_heun0','affine_heun0','local_ddim','affine_ddim']:
        for variant in [raw,centered]:
            values=[value(model,variant,k) for k in ['crps_h0','crps_h6','increment','pair_0-50_h6','energy']]
            lines.append('| '+model+' | '+('raw16' if variant==raw else 'pair-center16')+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','| 比较 | 指标 | 改善 | 差值95%区间 |','|---|---|---:|---|']
    chosen=[f'local_heun0/{centered}_vs_affine_heun0/{centered}',f'local_heun0/{centered}_vs_reference/{centered}',
        f'local_ddim/{centered}_vs_affine_ddim/{centered}',f'local_heun0/{centered}_vs_local_heun0/{raw}']
    for name in chosen:
        for key in ['crps_h0','crps_h6','increment','pair_0-50_h6','energy','event20_increase_brier','event20_decrease_brier']:
            v=comparisons[name][key];lines.append(f"| {name} | {key} | {v['gain_percent']:.3f}% | {v['delta_ci95']} |")
    lines+=['','## 解释边界','',
        '配对变换的解析点预测仍是约9.818mm的mu_base；K16样本均值有有限成员误差，不用它替换确定性点值。原始与变换的差异同时包含中心和高阶形状变化；只有同变换Local/Affine/G265比较才用于判断保留的非Gaussian价值。',
        '所有fair评分使用实际K16。第2/15顺序统计量覆盖的连续IID参考为13/17，不能与K32第3/30参考混用。事件阈值沿用10/20/30mm，主20mm；全部有效起点评分。',
        '2020主日历已暴露开发；没有新增日期验证、训练或读取确认/C/D标签。原32成员生成成本已付出，若线上获得32个新配对成员需64个原始场，成本应另计。两种子共享月份重采样，区间不包含训练种子总体不确定性。','',
        '![均值保持与匹配对照](pair_centering.png)','','完整所有距离/时效、事件、覆盖、fair均值MSE及哈希见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps({key:comparisons[key] for key in chosen}),flush=True)


if __name__=='__main__':main()
