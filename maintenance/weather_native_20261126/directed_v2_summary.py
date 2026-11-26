"""Source-checked directed/factorized transport versus matched H_delta."""
import argparse
from pathlib import Path
import json
import numpy as np


from increment_summary import rmse_comparison


def should_supplement(rows):
    raw={r['arm']:r for r in rows if r['weights']=='last'}
    return raw['joint']['horizon_macro_rmse_mm'][6]<raw['H_delta']['horizon_macro_rmse_mm'][6]


def locations(results,arm,seed,weights,calendar):
    root=results/'20261126_directed_lag_v2'
    model=(results/f'20261126_within_forecast_increment/H_delta_s{seed}' if arm=='H_delta' else root/f'{arm}_s{seed}')
    if calendar==8:
        embedding=(results/f'20261126_within_forecast_increment/H_delta_{weights}_dev_s{seed}' if arm=='H_delta' else root/f'{arm}_{weights}_dev_s{seed}')
        evaluation=model/f'eval_{weights}'
    else:
        embedding=root/f'supplement/{arm}_{weights}_s{seed}/embedding'
        evaluation=root/f'supplement/{arm}_{weights}_s{seed}/evaluation'
    return model,embedding,evaluation


def main():
    import torch
    from campaign import RESULTS,read
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_model_seal import validate_model_seal
    from tools.terrain_diffusion.native_directed_lag_v2_experiment import code_hashes,require_export_contract,require_final_ema
    from tools.terrain_diffusion.native_increment_experiment import code_hashes as increment_codes
    from tools.terrain_diffusion.native_weather_evaluation import calendar_shape,point_metrics
    from lead_mean_audit import verified_geometry
    from mean_hs_summary import score_records,safe_comparison
    from paired_mean_summary import paired_interval,plain
    p=argparse.ArgumentParser();p.add_argument('--calendar-day',type=int,choices=[8,15],default=8);args=p.parse_args()
    root=RESULTS/'20261126_directed_lag_v2';lock=read(Path(__file__).with_name('directed_v2_lock.json'))
    if lock['code_sha256']!=code_hashes():raise ValueError('registered directed implementation changed')
    engine=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/tools/terrain_diffusion'
    for name,expected in lock['evaluation_sources'].items():
        if sha(engine/name)!=expected:raise ValueError('registered evaluation dependency changed')
    rows=[];errors={};records={};sources={};curves={};truths={};references={}
    for arm in ['H_delta','factorized','joint']:
        for weights in ['last','ema']:
            metrics=[];per_seed=[]
            for seed in (42,17):
                model,embedding,evaluation=locations(RESULTS,arm,seed,weights,args.calendar_day)
                protocol=read(model/'protocol.json');status=read(model/'status.json')
                cache=RESULTS/f'20261126_native_weather_cache_v2/calendar{args.calendar_day}_s{seed}'
                source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
                expected_status='COMPLETE_MEAN_INCREMENT' if arm=='H_delta' else 'COMPLETE_DIRECTED_LAG_V2'
                if status['status']!=expected_status or status['steps']!=3000 or status['supervised_updates']!=3000:
                    raise ValueError('incomplete matched continuation')
                expected_parameters=108393 if arm=='H_delta' else lock['parameters'][arm]
                if protocol['config']['seed']!=seed or protocol['parameters']!=expected_parameters:raise ValueError('model recipe differs')
                if arm!='H_delta':
                    validate_model_seal(model,expected_status,code_hashes())
                    if protocol['config']['mode']!=arm or protocol['config']['disabled_transport']:raise ValueError('transport mode differs')
                    if protocol['config']['increment_weight']!=.5:raise ValueError('loss weight differs')
                else:validate_model_seal(model,expected_status,increment_codes())
                if protocol['source_checkpoint_sha256']!=sha(source/'checkpoint_3000.pt') or protocol['source_checkpoint_sha256']!=lock['source'][str(seed)]['checkpoint_sha256']:
                    raise ValueError('source checkpoint differs')
                if protocol['cache_manifest_sha256']!=lock['source'][str(seed)]['cache_manifest_sha256']:raise ValueError('training cache differs')
                require_export_contract(protocol,read(cache/'protocol.json'))
                if max(protocol['warm_start_max_difference'])>2e-5:raise ValueError('warm start differs')
                checkpoint=torch.load(model/'checkpoint_3000.pt',map_location='cpu',weights_only=True)
                actual=torch.load(model/f'{weights}.pt',map_location='cpu',weights_only=True)
                chosen=checkpoint['model'] if weights=='last' else checkpoint['ema']
                if checkpoint['step']!=3000 or actual.keys()!=chosen.keys() or any(not torch.equal(actual[k],v) for k,v in chosen.items()):
                    raise ValueError('weights differ from final3000')
                if weights=='ema':require_final_ema(actual,checkpoint)
                manifest=read(embedding/'manifest.json');lineage=read(evaluation/'lineage.json')
                expected_variant='increment_H_delta' if arm=='H_delta' else 'directed_v2_'+arm
                if manifest['mean_variant']!=expected_variant:raise ValueError('export variant differs')
                if manifest['mean_protocol_sha256']!=sha(model/'protocol.json') or manifest['mean_weights_sha256']!=sha(model/f'{weights}.pt'):
                    raise ValueError('embedding model differs')
                if manifest['cache_manifest_sha256']!=sha(cache/'manifest.json') or lineage['cache_manifest_sha256']!=sha(cache/'manifest.json') or lineage['embedding_manifest_sha256']!=sha(embedding/'manifest.json'):
                    raise ValueError('calendar source differs')
                for name,expected in manifest['files'].items():
                    if sha(embedding/name)!=expected:raise ValueError('embedding array differs')
                if lineage['prediction_sha256']!=sha(evaluation/'evaluation.npz'):raise ValueError('evaluation array differs')
                with np.load(evaluation/'evaluation.npz') as data:
                    prediction=data['prediction'];truth=data['truth'];reference=data['reference']
                np.testing.assert_array_equal(prediction,calendar_shape(np.load(embedding/'mean.npy'),12))
                if seed in truths:
                    np.testing.assert_array_equal(truth,truths[seed]);np.testing.assert_array_equal(reference,references[seed])
                else:truths[seed]=truth.copy();references[seed]=reference.copy()
                e=prediction.astype('float64')-truth.astype('float64');m=point_metrics(prediction,truth)
                np.testing.assert_allclose(m['horizon_macro_rmse_mm'],read(evaluation/'metrics.json')['deterministic_mean']['horizon_macro_rmse_mm'],atol=1e-10,rtol=0)
                geometry,geo_sha=verified_geometry(cache);nearest=calendar_shape(np.load(cache/'nearest.npy'),12)
                if sha(cache/'nearest.npy')!=read(cache/'manifest.json')['files']['nearest.npy']:raise ValueError('nearest distance cache differs')
                records[seed,arm,weights]=score_records(e,geometry,nearest)
                metrics.append(m);per_seed.append(e);curves[arm,seed]=read(model/'curve.json')
                sources[f'{arm}/{weights}/s{seed}']=dict(protocol_sha256=sha(model/'protocol.json'),weights_sha256=sha(model/f'{weights}.pt'),
                    checkpoint_sha256=sha(model/'checkpoint_3000.pt'),lineage_sha256=sha(evaluation/'lineage.json'),geometry_sha256=geo_sha,
                    nearest_sha256=sha(cache/'nearest.npy'),cache_manifest_sha256=sha(cache/'manifest.json'),prediction_sha256=lineage['prediction_sha256'])
                del checkpoint,actual
            errors[arm,weights]=np.stack(per_seed)
            rows.append(dict(arm=arm,weights=weights,horizon_macro_rmse_mm=np.mean([m['horizon_macro_rmse_mm'] for m in metrics],axis=0).tolist(),
                current_increment_macro_rmse_mm=float(np.mean([m['current_hourly_increment_macro_rmse_mm'] for m in metrics])),
                within_forecast_increment_micro_rmse_mm=float(np.mean([m['within_forecast_increment_micro_rmse_mm'] for m in metrics])),per_seed=metrics))
    comparisons={}
    for weights in ['last','ema']:
        for candidate,reference in [('joint','H_delta'),('factorized','H_delta'),('joint','factorized')]:
            a=errors[candidate,weights];b=errors[reference,weights]
            common=set.intersection(*[set(records[s,arm,weights]) for s in (42,17) for arm in [candidate,reference]])
            pairs=[(records[s,candidate,weights]['trajectory_increment_mse'],records[s,reference,weights]['trajectory_increment_mse']) for s in (42,17)]
            comparisons[f'{candidate}_vs_{reference}_{weights}']=dict(internal_increment_micro_rmse_mm=rmse_comparison(pairs),
                macro_R_F6_delta_ci95_mm=paired_interval(a[...,[0,6]],b[...,[0,6]]),
                current_increment_macro_delta_ci95_mm=paired_interval(np.diff(a[...,0],axis=3)[...,None],np.diff(b[...,0],axis=3)[...,None])[0],
                mse_strata={k:safe_comparison([(records[s,candidate,weights][k],records[s,reference,weights][k]) for s in (42,17)]) for k in sorted(common)})
    out=root/f'paired_calendar{args.calendar_day}';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',plain(dict(rows=rows,comparisons=comparisons,sources=sources,calendar_day=args.calendar_day,supplement_trigger=should_supplement(rows),
        primary='joint_vs_H_delta raw final3000 F6 station macroRMSE; shared-month bootstrap2000; factorized same parameter count',
        code_sha256=sha(__file__),lock_sha256=sha(Path(__file__).with_name('directed_v2_lock.json')),
        dependencies={name:sha(Path(__file__).with_name(name)) for name in ['mean_hs_summary.py','paired_mean_summary.py','weather_controls_summary.py','lead_mean_audit.py','increment_summary.py']},
        strict_oof=False,confirmation_C_D_read=False,limitations='exposed sampled2020 development; fixed two seeds; no multiplicity correction; original future-availability target selection')))
    render(out,rows,comparisons,curves,args.calendar_day)


def render(out,rows,comparisons,curves,calendar):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for row in rows:
        axes[0].plot(range(7),row['horizon_macro_rmse_mm'],marker='o',ls='-' if row['weights']=='last' else '--',label=f"{row['arm']} {row['weights']}")
    raw=[r for r in rows if r['weights']=='last']
    axes[1].bar([r['arm'] for r in raw],[r['horizon_macro_rmse_mm'][6] for r in raw]);axes[1].set_ylim(bottom=0)
    for arm in ['H_delta','factorized','joint']:
        for seed,style in [(42,'-'),(17,'--')]:
            c=curves[arm,seed];axes[2].plot([r['step'] for r in c],[r['query_loss'] for r in c],style,label=f'{arm},s{seed}')
    axes[0].set(xlabel='Lead (h)',ylabel='Station macro RMSE (mm)',title=f'Calendar {calendar}; frozen final3000');axes[0].legend(fontsize=7)
    axes[1].set(ylabel='F6 station macro RMSE (mm)',title='Primary; raw final3000')
    axes[2].set(xlabel='Consecutive cached block',ylabel='Training query loss',title='Matched objective; changing weather blocks');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'directed_comparison.png',dpi=170);plt.close(fig)
    lines=['# 有向站点×滞后传递：同源同预算的真实训练结果','',
        f'2016–2019原连续缓存，同源TF3000各新增3000，种子42/17；2020每月{calendar}日起72h三布局，已暴露开发。raw final3000为主，EMA并列。H_delta为匹配对照；factorized/joint增加同参数四头数值传递，joint使用历史key和空间/lag交互。三者损失均含0.5内部增量及原跨起点差分。','',
        '| 模型 | 权重 | 当前站宏RMSE/mm | 六小时站宏RMSE/mm | 跨起点当前增量站宏RMSE/mm | 同起点内部增量microRMSE/mm |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['arm']} | {r['weights']} | {r['horizon_macro_rmse_mm'][0]:.6f} | {r['horizon_macro_rmse_mm'][6]:.6f} | {r['current_increment_macro_rmse_mm']:.6f} | {r['within_forecast_increment_micro_rmse_mm']:.6f} |")
    lines+=['','## 配对比较','']
    for w,c in comparisons.items():
        r=c['internal_increment_micro_rmse_mm']
        lines.append(f"- {w}：内部增量micro RMSE差值{r['delta']:.6f}mm，95%共同月份区间{r['delta_ci95']}；当前/六小时站宏RMSE区间{c['macro_R_F6_delta_ci95_mm']}；跨起点当前增量区间{c['current_increment_macro_delta_ci95_mm']}。")
    lines+=['','差值为名称中的前者减后者，负值表示误差减少。主指标是raw joint相对H_delta的F6站宏RMSE；内部增量micro为次要指标。月份重采样对两模型及两种子共享。分层记录为MSE，不与站宏RMSE混排；稀疏季节分层若空重采样，区间记NA。',
        'final3000 raw/EMA逐tensor、同一源权重/缓存、truth/reference逐值、导出数组、几何/条件距离及lineage哈希均核验。strict_oof=false，未来可用性选站等旧数据限制保留；确认及原C/D标签未读，区间条件于两个固定种子且未经多重比较校正。','',
        '![性能与连续训练曲线](directed_comparison.png)','','完整来源、分层与配对结果见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(rows=rows,comparisons={w:{k:v for k,v in c.items() if k!='mse_strata'} for w,c in comparisons.items()})),flush=True)


if __name__=='__main__':main()
