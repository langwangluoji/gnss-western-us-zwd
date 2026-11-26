"""Shared-block transfer evidence for the frozen full-field tail diagnostic."""
from pathlib import Path
import numpy as np
from scipy.stats import chi2,f
from campaign import RESULTS,read
from weather_controls_summary import paired_record
from tools.terrain_diffusion.data import sha,write_json


def record(rows,name,blocks):
    sums=[];counts=[]
    for block in blocks:
        values=[r['nll_per_dimension'][name] for r in rows if r['block_index0']==block and r['dimension']>0]
        sums.append(sum(values));counts.append(len(values))
    return dict(value=sum(sums)/sum(counts),monthly_score_sum=sums,monthly_valid_count=counts)


def main():
    root=RESULTS/'20261126_full_field_tail_probe';summaries={};records={};sources={}
    generated=read(root/'generated_check.json')
    if not generated['complete']:raise ValueError('generated Gaussian covariance check incomplete')
    for seed in (42,17):
        folder=root/f's{seed}';summary=read(folder/'summary.json');protocol=read(folder/'protocol.json');data=read(folder/'records.json')
        if not summary['complete'] or summary['protocol_sha256']!=sha(folder/'protocol.json') or data['source_protocol_sha256']!=sha(folder/'protocol.json'):
            raise ValueError('radial source differs')
        if protocol['blocks']!=np.linspace(0,2908,48,dtype=int).tolist():raise ValueError('registered blocks changed')
        if len(data['rows'])!=48*12:raise ValueError('registered origin population changed')
        selected=summary['selected_tail_family']
        valid=[r for r in data['rows'] if r['dimension']>0];fit=[r for r in valid if r['year']<2019]
        names=['gaussian',*[f't{v}' for v in protocol['nu_grid']]]
        if selected!=min(names,key=lambda n:np.mean([r['nll_per_dimension'][n] for r in fit])):
            raise ValueError('tail family was not selected using2016-2018 only')
        if any(r['year'] not in (2016,2017,2018,2019) for r in valid):raise ValueError('nontraining labels read')
        for role,isfit in [('fit2016_2018',True),('transfer2019',False)]:
            rows=[r for r in valid if (r['year']<2019)==isfit];blocks=sorted({r['block_index0'] for r in rows})
            for label,name in [('gaussian','gaussian'),('scaled_gaussian','scaled_gaussian'),('selected_tail',selected)]:
                records[seed,role,label]=record(rows,name,blocks)
                expected=summary['scores'][name][role]
                np.testing.assert_allclose(records[seed,role,label]['value'],expected,atol=1e-12,rtol=0)
        summary['radial_calibration']={}
        for role,isfit in [('fit2016_2018',True),('transfer2019',False)]:
            subset=[r for r in valid if (r['year']<2019)==isfit]
            q=np.array([r['radius'] for r in subset]);d=np.array([r['dimension'] for r in subset])
            g=chi2.cdf(q,d);sg=chi2.cdf(q/summary['gaussian_scale2_fit'],d)
            nu=None if selected=='gaussian' else float(selected[1:])
            t=g if nu is None else f.cdf(q*nu/((nu-2)*d),d,nu)
            summary['radial_calibration'][role]=dict(gaussian_above95=float((g>.95).mean()),
                scaled_gaussian_above95=float((sg>.95).mean()),selected_tail_above95=float((t>.95).mean()))
        summaries[str(seed)]=summary
        sources[str(seed)]={n:sha(folder/n) for n in ('protocol.json','records.json','summary.json')}
    comparisons={}
    for role in ('fit2016_2018','transfer2019'):
        for a,b in [('scaled_gaussian','gaussian'),('selected_tail','gaussian'),('selected_tail','scaled_gaussian')]:
            comparisons[f'{role}/{a}_vs_{b}']=paired_record([(records[s,role,a],records[s,role,b]) for s in (42,17)])
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(summaries=summaries,comparisons=comparisons,sources=sources,
        generated_check_sha256=sha(root/'generated_check.json'),code_sha256=sha(__file__),
        interval='2000 shared selected-block resamples, equal fixed seeds; paired_record generic slots are blocks, not extra independent hours',
        dependency_sha256=sha(Path(__file__).with_name('weather_controls_summary.py')),
        interpretation='probe transfer2019; base/covariance already fitted2019; log-score diagnostic, not2020 forecast gain',
        strict_oof=False,confirmation_C_D_read=False))
    render(out,summaries,comparisons,root)


def render(out,summaries,comparisons,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for seed in (42,17):
        rows=read(root/f's{seed}/records.json')['rows'];pit=[r['gaussian_radial_pit'] for r in rows if r['year']==2019 and r['dimension']>0]
        axes[0].hist(pit,bins=np.linspace(0,1,11),density=True,histtype='step',lw=1.6,label=f's{seed},2019')
        s=summaries[str(seed)];grid=[3,4,6,8,12,20,40,80,160,1000]
        for role,style in [('fit2016_2018','-'),('transfer2019','--')]:
            axes[1].semilogx(grid,[s['scores'][f't{v}'][role]-s['scores']['scaled_gaussian'][role] for v in grid],style,label=f's{seed},{role}')
    axes[0].axhline(1,color='#64748b',ls=':');axes[0].set(xlabel='Gaussian radial PIT',ylabel='Density',title='Observed joint-radius calibration');axes[0].legend(fontsize=7)
    axes[1].axhline(0,color='#64748b',lw=1);axes[1].set(xlabel='Student degrees of freedom',ylabel='NLL / dimension difference',title='Fixed covariance t minus calibrated Gaussian');axes[1].legend(fontsize=6)
    edges=['scaled_gaussian_vs_gaussian','selected_tail_vs_gaussian','selected_tail_vs_scaled_gaussian']
    for i,edge in enumerate(edges):
        r=comparisons['transfer2019/'+edge];v=r['delta'];lo,hi=r['delta_ci95']
        axes[2].errorbar(v,2-i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4)
    axes[2].axvline(0,color='#64748b',lw=1);axes[2].set(yticks=[2,1,0],yticklabels=['Scale G - G','Selected t - G','Selected t - Scale G'],xlabel='NLL / dimension difference',title='2019 transfer; shared-block 95% interval')
    axes[2].tick_params(axis='y',labelsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'tail_probe.png',dpi=170);plt.close(fig)
    lines=['# 完整残差场重尾诊断：Gaussian尺度校准后仍有联合对数评分信号','',
        '每种子48个预先固定训练块，2016–2018拟合常数探针、2019检验探针迁移。原TF3000/G265/Affine来源已经拟合2019，因此不是整模型独立留年验证。没有读取2020或确认/C/D标签。','',
        '| 种子 | 训练内选择 | Gaussian方差校准倍数 | 2019 Gaussian NLL/d | 2019尺度Gaussian | 2019所选t |','|---|---|---:|---:|---:|---:|']
    for seed in (42,17):
        s=summaries[str(seed)];choice=s['selected_tail_family'];score=s['scores']
        lines.append(f"| {seed} | {choice} | {s['gaussian_scale2_fit']:.5f} | {score['gaussian']['transfer2019']:.6f} | {score['scaled_gaussian']['transfer2019']:.6f} | {score[choice]['transfer2019']:.6f} |")
    lines+=['','## 配对块区间','']
    for edge in edges:
        r=comparisons['transfer2019/'+edge];lines.append(f"- {edge}：NLL/d差值{r['delta']:.6f}，95%共同块区间{r['delta_ci95']}。")
    lines+=['','## 径向校准','', '| 种子 | 2019 Gaussian PIT>95% | 尺度Gaussian | 所选t | 校准参考 |','|---|---:|---:|---:|---:|']
    for seed in (42,17):
        r=summaries[str(seed)]['radial_calibration']['transfer2019']
        lines.append(f"| {seed} | {r['gaussian_above95']:.4f} | {r['scaled_gaussian_above95']:.4f} | {r['selected_tail_above95']:.4f} | 0.05 |")
    lines+=['','固定协方差Student-t具有相同均值和协方差；尺度Gaussian的常数只由2016–2018拟合。对数评分改善不能直接换算成CRPS、RMSE或事件收益，也不能排除遗漏天气依赖尺度/相关误配。下一步应加入同天气输入的条件Gaussian尺度控制。',
        '生成自校验的q/d均值约1.000，Gaussian径向PIT>95%比例约4.8%–5.3%；真实2019残差对应比例约56%–58%。这支持继续研究联合残差分布，并非协方差实现错误的直接迹象。','',
        '![径向与对数评分诊断](tail_probe.png)','','完整半径、有效维数、代码/权重/协议哈希见[summary.json](summary.json)。48块诊断不代表神经3000更新或2020正式预测收益。旧strict_oof=false、目标未来可用性筛选和历史QC来源限制保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(comparisons['transfer2019/selected_tail_vs_scaled_gaussian'],flush=True)


if __name__=='__main__':main()
