"""Report paired short-screen results without promoting local diagnostics to final claims."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnss_field.artifacts import read_json,write_json,sha256
from gnss_field.evaluation import point_metrics

ROOT=Path(__file__).resolve().parent
ARMS=['Gaussian-original','Gaussian-calibrated','G0','G-pair','G-noise','D0','D-attn','D-MP','N-point','N-set','N-GP-point','N-GP-set']
BASE={'G-pair':'G0','G-noise':'G0','D-attn':'D0','D-MP':'D0','N-set':'N-point','N-GP-point':'N-point','N-GP-set':'N-GP-point'}


def paired_rmse(path,base):
    a=dict(np.load(path));b=dict(np.load(base));assert np.array_equal(a['hours'],b['hours']) and np.array_equal(a['population'],b['population'])
    rng=np.random.default_rng(42);values=[]
    for _ in range(2000):
        ix=rng.integers(0,len(a['hours']),len(a['hours']));mask=a['population'][ix]
        ra=point_metrics(a['truth'][ix],a['mean'][ix],mask)['station_macro_rmse_mm']
        rb=point_metrics(b['truth'][ix],b['mean'][ix],mask)['station_macro_rmse_mm'];values.append(ra-rb)
    return dict(delta_RMSE_95_interval_mm=np.quantile(values,[.025,.975]).tolist(),
        resampling='paired twelve sampled calendar dates,2000 draws; exploratory not 7-day block bootstrap')


def main():
    rows=[];intervals={}
    for fold in (0,1):
        for arm in ARMS:
            directory=ROOT/f'fold_{fold}'/arm
            if not (directory/'result.json').exists():continue
            row=read_json(directory/'result.json');rows.append(row)
            if arm in BASE:
                intervals[f'{fold}/{arm}']=paired_rmse(directory/'outer_predictions.npz',ROOT/f'fold_{fold}'/BASE[arm]/'outer_predictions.npz')
    write_json(ROOT/'paired_intervals.json',intervals)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for fold in (0,1):
        for arm in ARMS[2:]:
            path=ROOT/f'fold_{fold}'/arm/'training.jsonl'
            if not path.exists():continue
            history=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            ax=axes[fold,1 if arm.startswith('N-') else 0]
            ax.plot([r['step'] for r in history],[r.get('DSM',r.get('ES_mm')) for r in history],label=arm,alpha=.85)
        for col in (0,1):
            ax=axes[fold,col];ax.set_title(f'Fold{fold}: '+('weighted observation DSM' if col else 'training ES (mm)'))
            ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,key,title in zip(axes,['group_ES_mm','CRPS_mm','station_macro_rmse_mm'],['Joint ES, K32 (mm)','Marginal CRPS, K32 (mm)','Point RMSE, K32 raw estimator (mm)']):
        for fold,color in [(0,'#286d97'),(1,'#d07e29')]:
            vals=[next((r['outer'][key] for r in rows if r['arm']==a and r['fold']==fold),np.nan) for a in ARMS]
            ax.plot(range(len(ARMS)),vals,'o-',label=f'fold{fold}',color=color)
        ax.set_xticks(range(len(ARMS)),ARMS,rotation=70,ha='right');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'metric_comparison.png',dpi=160);plt.close(fig)
    if (ROOT/'sampling_stability.json').exists():
        stability=read_json(ROOT/'sampling_stability.json');fig,axes=plt.subplots(1,2,figsize=(13,5))
        for fold,ax in enumerate(axes):
            for arm,color in zip(['N-point','N-set','N-GP-point','N-GP-set'],['#286d97','#d07e29','#379a6a','#9d5ba0']):
                m=stability[f'{fold}/{arm}']['metrics'];ks=[32,128,512]
                ax.plot(ks,[m[str(k)]['station_macro_rmse_mm'] for k in ks],'o--',color=color,alpha=.6,label=arm+' raw MC')
                ax.plot(ks,[m[str(k)]['point_control_variate']['station_macro_rmse_mm'] for k in ks],'o-',color=color,label=arm+' control variate')
            ax.axhline(stability[f'{fold}/Gaussian-calibrated']['metrics']['512']['station_macro_rmse_mm'],color='black',linestyle=':',label='Frozen mean')
            ax.set_xscale('log',base=2);ax.set_xticks(ks,[str(k) for k in ks]);ax.set_xlabel('Ensemble members, fixed checkpoint')
            ax.set_ylabel('Station-macro RMSE (mm)');ax.set_title(f'Fold{fold}: estimator precision, same model');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(ROOT/'sampling_estimator_comparison.png',dpi=160);plt.close(fig)
    grid_arms=['G0','G-pair','G-noise','D0','D-attn','D-MP']
    fig,axes=plt.subplots(2,6,figsize=(18,7),squeeze=False)
    for j,arm in enumerate(grid_arms):
        path=ROOT/'fold_0'/arm/'field_case.npz'
        if not path.exists():continue
        a=np.load(path);mean=a['mean_Z0'].squeeze();sd=a['std_Z0'].squeeze()
        for i,value,label,limits in [(0,mean,'Z0 mean (mm)',(0,250)),(1,sd,'latent field SD (mm)',(0,20))]:
            im=axes[i,j].imshow(value,origin='lower',cmap='viridis',vmin=limits[0],vmax=limits[1]);axes[i,j].set_title(arm+' '+label,fontsize=9)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        if j==5:
            fig.colorbar(axes[0,j].images[0],ax=axes[0,:].tolist(),shrink=.7)
            fig.colorbar(axes[1,j].images[0],ax=axes[1,:].tolist(),shrink=.7)
    fig.suptitle('First predeclared outer hour, fold0;25km output spacing; uncertainty excludes observation tau')
    fig.savefig(ROOT/'field_examples.png',dpi=150,bbox_inches='tight');plt.close(fig)
    text=['# 本机生成架构短程试验结果','',
        f'状态：已完成{len(rows)}/24个模型及对照结果。'+('本批已完成。' if len(rows)==24 else '这是训练中的部分结果，不作为本批结论。'),'',
        '本批真实训练四项GAN/Diffusion本体消融，并训练NDP启发的站点集合去噪器及逐点对照。B2018属于带历史底座暴露的增量开发；C2019/D2020未读。本批48步格网训练用于机制筛查，不判断收敛或架构上限。',
        '', '## 主要发现','',
        '四项格网本体改动尚未形成值得服务器扩训的稳定优势。注意力相对匹配Diffusion基线的ES变化约百万分之几毫米；GAN两种变体几乎重合；MP变体未同时改善两折联合分数。48步范围内的结果不能排除充分训练后的收益。',
        '', 'NDP观察空间DSM路线已能实际训练、生成联合站点集合并按固定留出站时刻评分，但当前仍不提供连续高分辨率场。512成员下，GP噪声逐点版本相对独立噪声逐点版本的VS约改善6.86%/3.28%，ES却分别增加0.01202/0.01126mm；其依赖改善主要来自引入已有空间先验，尚未达到联合概率晋级条件。',
        '', '原32成员的点值差异受到约1.4mm均值采样标准误影响。固定检查点、增加到512成员并用已知零均值初始噪声做控制变量后，逐点NDP相对共同均值的站宏RMSE变化为−0.02216/+0.01079mm；十二日期配对95%区间分别为[−0.05790,0.01398]/[−0.02928,0.05917]mm，均跨0。没有把估计器方差降低归功于模型架构。',
        '', '下一步建议先诊断共同底座已有的格网纹理、提高开发集合评分的采样稳定性，再预登记更长的匹配去噪训练。保留轻量逐点去噪与GP相关先验为对照，不据本轮短筛把注意力或更复杂生成器直接移交服务器。',
        '', '|模型|折|更新/入选步|RMSE mm|ES mm|VS|CRPS mm|90%覆盖/宽度mm|均值MC标准误mm|',
        '|---|---:|---|---:|---:|---:|---:|---|---:|']
    for row in rows:
        m=row['outer'];text.append(f"|{row['arm']}|{row['fold']}|{row.get('steps',0)}/{row.get('selected_step',0)}|{m['station_macro_rmse_mm']:.5f}|{m['group_ES_mm']:.5f}|{m['group_VS']:.5f}|{m['CRPS_mm']:.5f}|{m['coverage90']:.3f}/{m['width90_mm']:.3f}|{m['mean_mc_se_mm']:.3f}|")
    text+=['','## 逐项对照','']
    for arm,base in BASE.items():
        values=[]
        for fold in (0,1):
            r=next((r for r in rows if r['arm']==arm and r['fold']==fold),None);b=next((r for r in rows if r['arm']==base and r['fold']==fold),None)
            if r and b:
                d={k:r['outer'][k]-b['outer'][k] for k in ['group_ES_mm','group_VS','CRPS_mm','station_macro_rmse_mm']}
                values.append(f"折{fold}：ΔES={d['group_ES_mm']:+.6f}、ΔVS={d['group_VS']:+.6f}、ΔCRPS={d['CRPS_mm']:+.6f}、ΔRMSE={d['station_macro_rmse_mm']:+.6f} mm")
        text.append(f"- {arm} 相对 {base}："+'；'.join(values)+'。')
    if (ROOT/'sampling_stability.json').exists():
        text+=['','## 固定检查点的512成员采样复核','',
            '原K32选模与排名全部保留。以下仅增加采样成员，不重选权重；CV是减去已知零均值初始噪声的控制变量均值估计，不改变集合样本，也不作为架构收益。完整32/128/512前缀指标见sampling_stability.json。','',
            '|模型/折|原始MC均值RMSE|CV均值RMSE|原始/CV均值MC标准误|ES|VS|CRPS|','|---|---:|---:|---|---:|---:|---:|']
        for name,record in read_json(ROOT/'sampling_stability.json').items():
            m=record['metrics']['512'];cv=m.get('point_control_variate',m)
            text.append(f"|{name}|{m['station_macro_rmse_mm']:.5f}|{cv['station_macro_rmse_mm']:.5f}|{m['mean_mc_se_mm']:.4f}/{cv['mean_mc_se_mm']:.4f}|{m['group_ES_mm']:.5f}|{m['group_VS']:.5f}|{m['CRPS_mm']:.5f}|")
        text+=['','![采样与点估计精度](sampling_estimator_comparison.png)']
    text+=['','## 解释边界','',
        '- Gaussian-calibrated只使用各折拟合B站和训练时刻的标准化残差估计共同缩放，所有格网生成器从它初始化；同时保留原Gaussian。',
        '- 格网生成器按内层12小时、K16的最低ES选模，候选含第0步。入选0表示短筛未选中学习修正；各臂最终权重仍实际更新并保留。',
        '- N-point/N-set用真实观测残差的加权去噪MSE训练480步，在有限站点集合上生成带观测误差的分布；采样不再额外添加tau。它们是NDP启发原型，不是原论文复现，也不提供连续场产品。',
        '- N-GP-point/N-GP-set为看到第一折独立噪声结果后登记的增量试验，将前向及采样噪声改为固定高斯先验相关结构，保持单位边缘方差与其余训练设置；改善可能来自引入先验相关性。不得称作新独立确认。',
        '- N-set注意力版本与N-point逐点版本的比较同时改变了信息混合能力与参数量；没有参数量完全匹配控制，不能将全部收益归因于空间注意力。',
        '- NDP路线与格网路线的训练目标和更新预算不同，表格展示可行性，不据此宣称架构公平优胜。其点值使用完整MC均值；格网路线用已知高斯均值控制变量，故同时报告MC标准误。',
        '- 各折独立报告；未把折模型拼成一个可部署联合场。十二外层时刻按月份先验选择；置信区间按采样日期重采样，仅作探索，不称全年7日/14日块验证。',
        '- 原始预测与支持人口全部保留。field_examples图片色标固定；更丰富纹理不作为真实细节证据。下一步需更长训练、更多外层天气时刻与采样前缀稳定性后才决定服务器晋级。',
        '', '![训练曲线](training_curves.png)','', '![指标比较](metric_comparison.png)','', '![场图](field_examples.png)']
    if (ROOT/'base_field_audit.json').exists():
        text+=['','## 共同底座的输出检查','',
            '代表小时的输入支持陆地上，M0东西向二阶差分RMS为8.503mm，A-IDW为3.257mm，冻结混合底座为6.366mm，K8集合均值为6.384mm。块状纹理在冻结底座中已经存在；这一单案例粗糙度检查不等于精度排序或已证明拼接错误。应另做格网周期与拼接边界对齐诊断，不将所有高频归因于新生成器。',
            '', '![共同底座检查](base_field_audit.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'cache' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.name!='delivery_manifest.json']
    write_json(ROOT/'delivery_manifest.json',dict(files={str(p.relative_to(ROOT)):sha256(p) for p in files},server_jobs=False,models=len(rows),complete=len(rows)==24,production_source_unchanged=True))
    print('REPORT_COMPLETE',len(rows))


if __name__=='__main__':main()
