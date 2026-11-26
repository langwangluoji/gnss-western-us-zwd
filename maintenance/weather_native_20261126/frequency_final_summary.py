"""Matched C64 spectral/non-spectral final3000 evidence; no fitting or sampling."""
import json
from pathlib import Path
import numpy as np
from campaign import ROOT,RESULTS,read,STOP
from weather_controls_summary import monthly_record,paired_record
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_evaluation import fair_crps


def main():
    import torch
    engine=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
    code=engine/'tools/terrain_diffusion';root=RESULTS/'20261126_native_weather_probability'
    labels=['gaussian_ddim','gaussian_copula','spatial_ddim','graph_ddim','joint_ddim','frequency_tokens_ddim',
        'preconditioned_joint_ddim','preconditioned_joint_heun','preconditioned_frequency_tokens_ddim','preconditioned_frequency_tokens_heun']
    rows=[];records={};sources={}
    for seed in (42,17):
        common=None
        for label in labels:
            if STOP.exists():raise SystemExit(75)
            mode,solver=label.rsplit('_',1);folder=root/f'transformer72_physics_aux_last_s{seed}'/mode;evaluation=folder/f'eval_{solver}'
            protocol=read(folder/'protocol.json');status=read(folder/'status.json');lineage=read(evaluation/'lineage.json');metrics=read(evaluation/'metrics.json')
            if status['status']!='COMPLETE_PROBABILITY' or status['steps']!=3000 or status['supervised_updates']!=3000:
                raise ValueError('frequency model training incomplete')
            if protocol['config'].get('condition_view','latent')!='latent' or protocol.get('condition_dimension',64)!=64 or protocol['config']['seed']!=seed:
                raise ValueError('not the registered C64 source')
            if protocol['code_sha256']!=sha(code/'native_weather_models.py'):raise ValueError('probability model implementation changed')
            for name,expected in protocol['extra_model_sources'].items():
                if sha(code/name)!=expected:raise ValueError('declared model dependency changed')
            weights=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            actual_input=weights['net.0.weight'].shape[1] if mode=='gaussian' else weights['input.weight'].shape[1]-(8 if 'frequency_tokens' in mode else 12)
            if actual_input!=64:raise ValueError('actual saved model input is not C64')
            checkpoint=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
            if checkpoint['step']!=3000 or checkpoint['model'].keys()!=weights.keys() or any(not torch.equal(weights[k],v) for k,v in checkpoint['model'].items()):
                raise ValueError('weights do not equal final3000')
            if lineage['probability_weights_sha256']!=sha(folder/'last.pt') or lineage['prediction_sha256']!=sha(evaluation/'evaluation.npz'):
                raise ValueError('frequency weights or arrays changed')
            if lineage['members']!=32 or lineage['seed']!=261126 or not lineage['independent_iid_member_noise']:
                raise ValueError('frequency member recipe differs')
            if lineage['strict_oof'] or lineage['confirmation_C_D_read']:raise ValueError('unexpected data role')
            expected_solver='ddim' if solver=='copula' else solver
            expected_steps=20 if expected_solver=='heun' or solver=='copula' else 40
            if lineage['solver']!=expected_solver or lineage['sampling_steps']!=expected_steps:raise ValueError('sampler recipe differs')
            if 'frequency_tokens' in mode:
                backend=lineage.get('frequency_backend','original')
                if backend not in ['commuted','original']:raise ValueError('unknown spectral inference backend')
                if backend=='commuted' and lineage['frequency_backend_sha256']!=sha(code/'native_spectral_fast.py'):
                    raise ValueError('spectral inference backend changed')
            if protocol['preconditioner_sha256']:
                if protocol['preconditioner_sha256']!=sha(folder.parent/'gaussian/last.pt'):raise ValueError('frozen Gaussian preconditioner changed')
            if lineage['copula_sha256']:
                if lineage['copula_sha256']!=sha(folder.parent/'gaussian/copula.json'):raise ValueError('C64 copula changed')
            cache=RESULTS/f'20261126_native_weather_cache_v2/calendar8_s{seed}'
            emb=RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_last_dev_s{seed}'
            if lineage['cache_manifest_sha256']!=sha(cache/'manifest.json') or lineage['embedding_manifest_sha256']!=sha(emb/'manifest.json'):
                raise ValueError('frequency calendar or mean embedding differs')
            if read(emb/'manifest.json')['mean_weights_sha256']!=protocol['condition_encoder_weights_sha256']:
                raise ValueError('frequency training/evaluation mean differs')
            with np.load(evaluation/'evaluation.npz') as data:
                crps=data['crps'];truth=data['truth'];prediction=data['prediction']
                if common is None:common=(truth,prediction)
                else:
                    np.testing.assert_array_equal(truth,common[0]);np.testing.assert_array_equal(prediction,common[1])
                rec={f'h{h}':monthly_record(crps[...,h],1) for h in range(7)}
                # Independently check representative point scores from stored IID members.
                samples=data['samples']
                for h in (0,6):
                    np.testing.assert_array_equal(fair_crps(samples[...,h],truth[...,h]).astype(crps.dtype),crps[...,h])
                inc=[fair_crps(samples[...,h+1]-samples[...,h],truth[...,h+1]-truth[...,h]) for h in range(6)]
                rec['increment']=monthly_record(np.stack(inc,axis=-1),1)
                np.testing.assert_allclose(rec['increment']['value'],metrics['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
                np.testing.assert_allclose([rec[f'h{h}']['value'] for h in range(7)],metrics['fair_crps_mm_by_horizon'],atol=1e-6,rtol=0)
                del samples,inc
            records[seed,label]=rec
            row=dict(seed=seed,name=label,condition_dimension=64,parameters=protocol['parameters'],
                frozen_preconditioner_parameters=5070 if protocol['preconditioner_sha256'] else 0,records=rec,
                deterministic_R_F6_rmse_mm=metrics['deterministic_mean']['horizon_macro_rmse_mm'][::6],
                sample_mean_R_F6_rmse_mm=metrics['sample_mean']['horizon_macro_rmse_mm'][::6],
                NFE=lineage['neural_evaluations_per_sample'],final_checkpoint_tensor_match=True,
                original_evaluation_code_sha256=lineage['code_sha256'],frequency_backend=lineage.get('frequency_backend','original'))
            rows.append(row);sources[f's{seed}/{label}']={n:sha(folder/n) for n in ['protocol.json','status.json','last.pt','checkpoint_3000.pt']}
            sources[f's{seed}/{label}'].update(lineage_sha256=sha(evaluation/'lineage.json'),metrics_sha256=sha(evaluation/'metrics.json'),prediction_sha256=lineage['prediction_sha256'])
            print(json.dumps(dict(seed=seed,name=label,F6=rec['h6']['value'],increment=rec['increment']['value'])),flush=True)
    edges=[(v,'gaussian_copula') for v in labels if v!='gaussian_copula']
    edges += [('frequency_tokens_ddim','joint_ddim'),('frequency_tokens_ddim','graph_ddim'),
        ('preconditioned_frequency_tokens_ddim','preconditioned_joint_ddim'),('preconditioned_frequency_tokens_heun','preconditioned_joint_heun')]
    comparisons={a+'_vs_'+b:{key:paired_record([(records[s,a][key],records[s,b][key]) for s in (42,17)]) for key in ['h0','h6','increment']} for a,b in edges}
    out=root/'paired_spectral';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,comparisons=comparisons,sources=sources,code_sha256=sha(__file__),
        dependencies={n:sha(Path(__file__).with_name(n)) for n in ['weather_controls_summary.py']},
        strict_oof=False,confirmation_C_D_read=False,interval='2000 shared-month draws conditional on two fixed seeds; exposed2020, exploratory, no multiplicity correction',
        warning='C64 comparisons only; not a matched comparison to S175/G265 richer weather conditions'))
    render(out,labels,rows,comparisons)


def render(out,labels,rows,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    shown=['gaussian_copula','joint_ddim','frequency_tokens_ddim','preconditioned_joint_heun','preconditioned_frequency_tokens_heun']
    for label in shown:
        rr=[r for r in rows if r['name']==label]
        axes[0].plot(range(7),np.mean([[r['records'][f'h{h}']['value'] for h in range(7)] for r in rr],axis=0),'o-',label=label)
    edges=['frequency_tokens_ddim_vs_joint_ddim','preconditioned_frequency_tokens_ddim_vs_preconditioned_joint_ddim','preconditioned_frequency_tokens_heun_vs_preconditioned_joint_heun']
    for j,metric in enumerate(['h0','h6','increment']):
        v=np.array([comparisons[e][metric]['delta'] for e in edges]);ci=np.array([comparisons[e][metric]['delta_ci95'] for e in edges])
        axes[1].errorbar(np.arange(3)+(j-1)*.14,v,yerr=np.maximum(np.stack([v-ci[:,0],ci[:,1]-v]),0),fmt='o',capsize=3,label=metric)
    axes[0].set(xlabel='Lead (h)',ylabel='K32 fair CRPS (mm)',title='Same C64 inputs');axes[0].legend(fontsize=7)
    axes[1].axhline(0,color='gray',ls='--');axes[1].set(xticks=range(3),xticklabels=['plain DDIM','precond DDIM','precond Heun'],ylabel='Frequency minus joint score (mm)',title='Shared-month exploratory intervals');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'spectral_comparison.png',dpi=170);plt.close(fig)
    lines=['# 时空域与频域扩散：同C64条件双种子3000步','',
        '七组训练均完成双种子各3000连续块；Gaussian另有同权重copula推断，预条件两组各有DDIM/Heun采样器。2020主日历已暴露开发，非全年、非严格OOF。',
        '概率条件均为64维冻结Transformer状态。频域使用全部七个时间DCT系数、完整空间图谱响应及频带卷积；未截断频带。其图基依赖查询人口，尚不是查询集合一致的连续潜场。','',
        '| 模型 | 新训练参数 | 冻结Gaussian参数 | 当前CRPS/mm | F6 CRPS/mm | 内部增量CRPS/mm |','|---|---:|---:|---:|---:|---:|']
    for label in labels:
        rr=[r for r in rows if r['name']==label]
        def mean(key):return np.mean([r['records'][key]['value'] for r in rr])
        lines.append(f"| {label} | {rr[0]['parameters']} | {rr[0]['frozen_preconditioner_parameters']} | {mean('h0'):.6f} | {mean('h6'):.6f} | {mean('increment'):.6f} |")
    lines+=['','| 比较 | 指标 | 改善 | 差值95%区间/mm |','|---|---|---:|---|']
    for name,table in comparisons.items():
        if not name.startswith(('frequency_tokens','preconditioned_frequency_tokens')):continue
        for key,v in table.items():lines.append(f"| {name} | {key} | {v['gain_percent']:.3f}% | {v['delta_ci95']} |")
    lines+=['','全部模型最后权重已与checkpoint3000逐tensor核对；预测、truth、冻结均值及计分来源已验收。旧协议缺少显式条件维度时，另由权重输入维度核验C64。部分旧计分源码版本不同，原始源码哈希保留，并从实际成员独立复算。频域original/commuted后端据实际来源记录，不混写。频域与joint参数约38.4k/38.5k，预条件组额外冻结5070参数Gaussian。Gaussian本体5070参数，并非容量匹配对照。',
        '主要判断应比较同信息、相近容量的频域与joint，同时检查Gaussian copula参考。新S175/G265使用更丰富的直连天气信息，其差异不能归因为频域。内部增量收益也不能从独立Gaussian单独推断为非Gaussian优势。',
        '当前场点预测仍使用约9.818mm基础场；K32样本均值含有限采样和分布中心修正。月份区间条件于固定两种子，未经多重比较校正。','',
        '![频域对照及共同月份区间](spectral_comparison.png)','','完整来源和对照见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in comparisons.items() if k in edges}),flush=True)


if __name__=='__main__':main()
