"""Same-budget C/H/S/HS point-field comparisons, strata and interaction risk."""
import json
from pathlib import Path
import numpy as np


def risk_interaction(hs,h,s,c):
    counts=np.asarray(hs['monthly_valid_count'])
    if any(not np.array_equal(counts,r['monthly_valid_count']) for r in [h,s,c]):
        raise ValueError('interaction observation population differs')
    sums=np.asarray(hs['monthly_score_sum'])-h['monthly_score_sum']-s['monthly_score_sum']+np.asarray(c['monthly_score_sum'])
    return dict(value=float(sums.sum()/counts.sum()),monthly_score_sum=sums.tolist(),monthly_valid_count=counts.tolist())


def safe_comparison(pairs):
    from weather_controls_summary import paired_record
    try:return paired_record(pairs)
    except ValueError as error:
        if str(error)!='empty bootstrap population':raise
        # A seasonal/rare stratum can have no observations in a calendar draw.
        # Preserve the estimand; do not drop empty draws or invent zero scores.
        a=float(np.mean([np.sum(x['monthly_score_sum'])/np.sum(x['monthly_valid_count']) for x,_ in pairs]))
        b=float(np.mean([np.sum(x['monthly_score_sum'])/np.sum(x['monthly_valid_count']) for _,x in pairs]))
        return dict(candidate=a,reference=b,delta=a-b,gain_percent=100*(b-a)/b if b else None,delta_ci95=None,
            positive_month_counts=[int(np.count_nonzero(x['monthly_valid_count'])) for x,_ in pairs],
            interval_status='undefined: some locked shared-calendar month resamples contain no observations in this stratum; descriptive values retained')


def score_records(error,geometry,nearest):
    from weather_controls_summary import monthly_record
    records={f'mse_h{h}':monthly_record(error[...,h]**2,1) for h in (0,6)}
    records['trajectory_increment_mse']=monthly_record(np.diff(error,axis=-1)**2,1)
    records['origin_increment_mse']=monthly_record(np.diff(error[...,0],axis=2)**2,1)
    masks={}
    for lo,hi in [(0,50),(50,100),(100,150),(150,300),(300,np.inf)]:masks[f'distance_{lo}-{hi:g}']=(nearest>=lo)&(nearest<hi)
    for feature,values,edges in [('height',geometry[:,2]*3000,[-np.inf,500,1500,np.inf]),
        ('coast',geometry[:,8]*500,[0,50,150,300,np.inf]),('roughness_code',geometry[:,5],[0,.5,1.5,np.inf])]:
        for lo,hi in zip(edges[:-1],edges[1:]):masks[f'{feature}_{lo:g}-{hi:g}']=np.broadcast_to(((values>=lo)&(values<hi))[None,None,None],error.shape[:-1])
    for season,months in [('DJF',[11,0,1]),('MAM',[2,3,4]),('JJA',[5,6,7]),('SON',[8,9,10])]:
        selected=np.zeros(error.shape[:-1],dtype=bool);selected[:,months]=True;masks['season_'+season]=selected
    for name,mask in masks.items():
        for h in (0,6):
            value=np.where(mask,error[...,h]**2,np.nan)
            if np.isfinite(value).any():records[f'{name}_h{h}']=monthly_record(value,1)
    a,b=np.triu_indices(137,1);distance=np.linalg.norm((geometry[a,:2]-geometry[b,:2])*1000,axis=-1)
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        aa=a[(distance>=lo)&(distance<hi)];bb=b[(distance>=lo)&(distance<hi)]
        for h in (0,6):
            sums=np.zeros(12);counts=np.zeros(12,dtype='int64')
            for start in range(0,len(aa),128):
                values=(error[...,aa[start:start+128],h]-error[...,bb[start:start+128],h])**2
                row=monthly_record(values,1);sums+=row['monthly_score_sum'];counts+=row['monthly_valid_count']
            records[f'pair_{lo}-{hi:g}_h{h}']=dict(value=float(sums.sum()/counts.sum()),monthly_score_sum=sums.tolist(),monthly_valid_count=counts.tolist(),pair_count=len(aa))
    return records


def main():
    import torch
    from campaign import RESULTS,read
    from paired_mean_summary import paired_interval,plain
    from weather_controls_summary import paired_record
    from lead_mean_audit import verified_geometry
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_mean_fusion_v2_experiment import verify_codes as legacy_codes
    from tools.terrain_diffusion.native_mean_hs_experiment import code_hashes,require_export_contract,require_final_ema
    from tools.terrain_diffusion.native_model_seal import validate_model_seal
    from tools.terrain_diffusion.native_weather_evaluation import point_metrics,calendar_shape,safe_mean
    root=RESULTS/'20261126_mean_hs';rows=[];errors={};records={};curves={};truths={};station_rows={};sources={}
    for label in ['source','C','H','S','HS']:
        for weights in ['last','ema']:
            metrics=[];per_seed=[];parameters=[];times=[]
            for seed in (42,17):
                cache=RESULTS/f'20261126_native_weather_cache_v2/calendar8_s{seed}';geometry,geo_sha=verified_geometry(cache)
                nearest=calendar_shape(np.load(cache/'nearest.npy'),12)
                source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
                folder=source if label=='source' else root/f'HS_s{seed}' if label=='HS' else RESULTS/f'20261126_mean_fusion_v2/{label}_s{seed}'
                embedding=(RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_{weights}_dev_s{seed}' if label=='source' else
                    root/f'HS_{weights}_dev_s{seed}' if label=='HS' else RESULTS/f'20261126_mean_fusion_v2/{label}_{weights}_dev_s{seed}')
                protocol=read(folder/'protocol.json');status=read(folder/'status.json');manifest=read(embedding/'manifest.json');evaluation=folder/f'eval_{weights}';lineage=read(evaluation/'lineage.json')
                expected='COMPLETE_MEAN_HS' if label=='HS' else 'COMPLETE_MEAN'
                if status['status']!=expected or status['steps']!=3000 or status['supervised_updates']!=3000:raise ValueError('mean must finish all3000 updates')
                if protocol['config']['seed']!=seed:raise ValueError('mean seed changed')
                if label!='source':
                    if protocol['source_checkpoint_sha256']!=sha(source/'checkpoint_3000.pt'):raise ValueError('continuation starting point changed')
                    if max(protocol['warm_start_max_difference'])>2e-5:raise ValueError('non-equivalent warm start')
                    require_export_contract(protocol,read(cache/'protocol.json'))
                    if label=='HS':validate_model_seal(folder,expected,code_hashes())
                    else:legacy_codes(protocol)
                state=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
                actual=torch.load(folder/f'{weights}.pt',map_location='cpu',weights_only=True)
                selected=state['model'] if weights=='last' else state['ema']
                if state['step']!=3000 or actual.keys()!=selected.keys() or any(not torch.equal(actual[k],v) for k,v in selected.items()):raise ValueError('selected weights differ from final3000')
                if weights=='ema':require_final_ema(actual,state)
                if manifest['cache_manifest_sha256']!=sha(cache/'manifest.json') or manifest['mean_protocol_sha256']!=sha(folder/'protocol.json') or manifest['mean_weights_sha256']!=sha(folder/f'{weights}.pt'):
                    raise ValueError('mean export source changed')
                for name,value in manifest['files'].items():
                    if sha(embedding/name)!=value:raise ValueError('mean export array changed')
                if lineage['cache_manifest_sha256']!=sha(cache/'manifest.json') or lineage['embedding_manifest_sha256']!=sha(embedding/'manifest.json') or lineage['prediction_sha256']!=sha(evaluation/'evaluation.npz'):
                    raise ValueError('evaluation lineage changed')
                with np.load(evaluation/'evaluation.npz') as data:
                    prediction=data['prediction'];truth=data['truth'];reference=data['reference']
                np.testing.assert_array_equal(prediction,calendar_shape(np.load(embedding/'mean.npy'),12))
                if seed in truths:np.testing.assert_array_equal(truth,truths[seed])
                else:truths[seed]=truth.copy()
                e=prediction.astype('float64')-truth.astype('float64');m=point_metrics(prediction,truth)
                registered=read(evaluation/'metrics.json')['deterministic_mean']
                np.testing.assert_allclose(m['horizon_macro_rmse_mm'],registered['horizon_macro_rmse_mm'],atol=1e-10,rtol=0)
                metrics.append(m);per_seed.append(e);parameters.append(protocol['parameters']);times.append(status['seconds'])
                records[seed,label,weights]=score_records(e,geometry,nearest)
                curves[label,seed]=read(folder/'curve.json')
                station=safe_mean(np.sqrt(safe_mean(e**2,(1,2))),0)
                station_rows[seed,label,weights]=station.tolist()
                sources[f'{label}/{weights}/s{seed}']=dict(weights_sha256=sha(folder/f'{weights}.pt'),checkpoint_sha256=sha(folder/'checkpoint_3000.pt'),
                    protocol_sha256=sha(folder/'protocol.json'),lineage_sha256=sha(evaluation/'lineage.json'),prediction_sha256=lineage['prediction_sha256'],
                    geometry_sha256=geo_sha,nearest_sha256=sha(cache/'nearest.npy'))
                if sha(cache/'nearest.npy')!=read(cache/'manifest.json')['files']['nearest.npy']:raise ValueError('condition distance cache changed')
                del state,actual
            errors[label,weights]=np.stack(per_seed)
            rows.append(dict(model=label,weights=weights,parameters=parameters,training_seconds=times,
                horizon_macro_rmse_mm=np.mean([m['horizon_macro_rmse_mm'] for m in metrics],axis=0).tolist(),
                current_increment_macro_rmse_mm=float(np.mean([m['current_hourly_increment_macro_rmse_mm'] for m in metrics])),
                trajectory_increment_micro_rmse_mm=float(np.mean([m['within_forecast_increment_micro_rmse_mm'] for m in metrics])),
                per_seed=metrics,records=[records[s,label,weights] for s in (42,17)]))
    comparisons={};interactions={}
    for weights in ['last','ema']:
        for reference in ['source','C','H','S']:
            a=errors['HS',weights];b=errors[reference,weights]
            keys=set(records[42,'HS',weights])&set(records[17,'HS',weights])&set(records[42,reference,weights])&set(records[17,reference,weights])
            comparisons[f'HS_{weights}_vs_{reference}_{weights}']=dict(macro_R_F6_delta_ci95_mm=paired_interval(a[...,[0,6]],b[...,[0,6]]),
                current_increment_macro_delta_ci95_mm=paired_interval(np.diff(a[...,0],axis=3)[...,None],np.diff(b[...,0],axis=3)[...,None])[0],
                mse_comparisons={key:safe_comparison([(records[s,'HS',weights][key],records[s,reference,weights][key]) for s in (42,17)]) for key in sorted(keys)})
        for h in (0,6):
            pairs=[]
            for seed in (42,17):
                row=risk_interaction(*[records[seed,m,weights][f'mse_h{h}'] for m in ['HS','H','S','C']])
                zero={**row,'monthly_score_sum':[0]*12};pairs.append((row,zero))
            interactions[f'{weights}_h{h}']=paired_record(pairs)
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',plain(dict(rows=rows,comparisons=comparisons,interaction_HS_minus_H_minus_S_plus_C_MSE_mm2=interactions,
        station_RMSE_by_seed_model_weight={f's{s}/{m}/{w}':v for (s,m,w),v in station_rows.items()},sources=sources,
        code_sha256=sha(__file__),dependencies={name:sha(Path(__file__).with_name(name)) for name in ['paired_mean_summary.py','weather_controls_summary.py','lead_mean_audit.py']},
        strict_oof=False,confirmation_C_D_read=False,primary='raw final3000; EMA displayed separately; no checkpoint selection by development score',
        interval='2000 shared-month draws across two fixed seeds; exposed2020 development, exploratory, no multiplicity correction')))
    render(out,rows,curves,comparisons,interactions)


def render(out,rows,curves,comparisons,interactions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw=[r for r in rows if r['weights']=='last'];fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for r in raw:axes[0].plot(range(7),r['horizon_macro_rmse_mm'],'o-',label=r['model'])
    for model in ['C','H','S','HS']:
        for seed,style in [(42,'-'),(17,'--')]:
            curve=curves[model,seed];axes[1].plot([r['step'] for r in curve],[r['query_loss'] for r in curve],style,label=f'{model},s{seed}',alpha=.8)
    for seed,style in [(42,'-'),(17,'--')]:
        curve=curves['HS',seed];axes[2].plot([r['step'] for r in curve],[r['spatial_gate'] for r in curve],style,label=f'S gate,s{seed}')
    axes[0].set(xlabel='Lead (h)',ylabel='Station macro RMSE (mm)',title='Raw final3000; equal continuation budget');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Consecutive cached block',ylabel='Query training loss',title='Shared dates; weather difficulty changes');axes[1].legend(fontsize=7)
    axes[2].set(xlabel='Training update',ylabel='HS same-hour spatial gate',title='Zero influence at initialization');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'hs_comparison.png',dpi=170);plt.close(fig)
    lines=['# H＋S组合：同起点同预算的真实训练结果','',
        '2016–2019原缓存连续块顺序；源Transformer step3000，C/H/S/HS各新增3000。raw为主，EMA并列；源3000另列预算参考。2020每月8–10日三布局、137注册查询为已暴露开发；不代表独立全年。',
        'H为中心化创新读出，S为同刻空间消息。HS两分支分别匹配原单臂初始化并严格继承同一非零源；旧AdamW/RNG与同一cosine周期保留。','',
        '| 模型 | 权重 | 当前宏RMSE/mm | F6宏RMSE/mm | 当前小时增量宏RMSE/mm | 同起点内部增量microRMSE/mm | 参数 |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['weights']} | {r['horizon_macro_rmse_mm'][0]:.6f} | {r['horizon_macro_rmse_mm'][6]:.6f} | {r['current_increment_macro_rmse_mm']:.6f} | {r['trajectory_increment_micro_rmse_mm']:.6f} | {r['parameters'][0]} |")
    lines+=['','| 比较 | 当前差值95%区间/mm | F6差值95%区间/mm | 当前增量差值95%区间/mm |','|---|---|---|---|']
    for name,r in comparisons.items():lines.append(f"| {name} | {r['macro_R_F6_delta_ci95_mm'][0]} | {r['macro_R_F6_delta_ci95_mm'][1]} | {r['current_increment_macro_delta_ci95_mm']} |")
    lines+=['','组合交互以HS−H−S＋C的MSE差描述，单位mm²，完整值和月份区间见JSON；它是预测风险的统计交互，不能单独排除参数容量等替代解释。距离、高程、海岸、无量纲地形起伏编码、四季及站对分层以micro MSE另存，不与总体station macro RMSE混用。季节/稀有分层若共同月份重采样产生空总体，则保留描述值、区间记NA，不删除空抽样或填零。空分层不作排名。',
        '所有raw/EMA均与final3000 checkpoint逐tensor核验，导出权重/协议/embedding及评价来源、几何与条件距离哈希绑定。点值不从有限成员概率均值估计。',
        '缓存末尾91块顺序复用但不跨回绕拼历史；目标站未来轨迹可用性筛选与strict_oof=false限制保留。确认及原C/D标签未读取。月份区间条件于两个固定种子，未经多重比较校正。','',
        '![性能、同日期训练曲线与HS空间gate](hs_comparison.png)','','完整分层、逐站、来源与交互风险见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(rows=len(rows),raw=[{k:r[k] for k in ['model','horizon_macro_rmse_mm','current_increment_macro_rmse_mm']} for r in raw],comparisons={k:v['macro_R_F6_delta_ci95_mm'] for k,v in comparisons.items()})),flush=True)


if __name__=='__main__':main()
