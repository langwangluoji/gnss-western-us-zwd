"""v2: mask-aware completion; original model, scores and scientific lock unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np


def should_supplement(rows):
    values={name:np.mean([r['records']['crps_h6']['value'] for r in rows if r['name']==name])
        for name in ('odd_frequency','affine_frequency')}
    return bool(values['odd_frequency']<values['affine_frequency'])


def require_supplement(summary,lock_hash,code_hash):
    if summary.get('calendar_day')!=8 or summary.get('lock_sha256')!=lock_hash or summary.get('code_sha256')!=code_hash:
        raise ValueError('primary summary does not match current lock and scorer')
    rows=summary['rows'];names=['gaussian_time','affine_time','affine_frequency','odd_time','odd_frequency']
    expected={(s,n) for s in (42,17) for n in names}
    if len(rows)!=len(expected) or {(r['seed'],r['name']) for r in rows}!=expected:
        raise ValueError('primary summary model population incomplete')
    trigger=should_supplement(rows)
    if summary.get('supplement_trigger')!=trigger or not trigger:raise ValueError('registered supplementary trigger not met')


def require_pairing(lin,family,domain,seed,sample_seed,device):
    expected=dict(family=family,domain=domain,members=32,sample_seed=sample_seed,
        layout_training_seed=seed,device=device,independent_iid_member_noise=True,coupled_gp_and_latent_noise=True)
    if any(lin.get(k)!=v for k,v in expected.items()):raise ValueError('registered IID/coupling/family/seed/device contract differs')


def strata_records(crps,geometry,nearest):
    from weather_controls_summary import monthly_record
    masks={}
    for lo,hi in [(0,50),(50,100),(100,150),(150,300),(300,np.inf)]:masks[f'distance_{lo}-{hi:g}']=(nearest>=lo)&(nearest<hi)
    for name,values,edges in [('height',geometry[:,2]*3000,[-np.inf,500,1500,np.inf]),
        ('coast',geometry[:,8]*500,[0,50,150,300,np.inf]),('roughness_code',geometry[:,5],[0,.5,1.5,np.inf])]:
        for lo,hi in zip(edges[:-1],edges[1:]):
            masks[f'{name}_{lo:g}-{hi:g}']=np.broadcast_to(((values>=lo)&(values<hi))[None,None,None],crps.shape[:-1])
    for name,months in [('DJF',[11,0,1]),('MAM',[2,3,4]),('JJA',[5,6,7]),('SON',[8,9,10])]:
        selected=np.zeros(crps.shape[:-1],dtype=bool);selected[:,months]=True;masks['season_'+name]=selected
    result={}
    for name,mask in masks.items():
        for h in (0,6):
            values=np.where(mask,crps[...,h],np.nan)
            if np.isfinite(values).any():result[f'{name}_h{h}']=monthly_record(values,1)
    return result


def main():
    from campaign import RESULTS,read
    from inducing_flow_series import ARMS,LOCK,source_paths,source_receipt
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_inducing_flow_experiment import codes,require_calendar
    from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
    from tools.terrain_diffusion.native_weather_evaluation import calendar_shape,point_metrics,fair_crps,gaussian_crps
    from weather_controls_summary import monthly_record,paired_record,event_records
    from mean_hs_summary import safe_comparison
    from lead_mean_audit import verified_geometry
    from event_audit import validate_calendar_origins
    p=argparse.ArgumentParser();p.add_argument('--calendar-day',type=int,choices=[8,15],default=8);args=p.parse_args();day=args.calendar_day
    from inducing_flow_completion import AMENDMENT,validate_amendment,require_actual_completion
    amendment=validate_amendment()
    lock=read(LOCK)
    if lock['code_sha256']!=codes():raise ValueError('locked flow code changed')
    if sha(Path(__file__).with_name('inducing_flow_series.py'))!=lock['driver_sha256']:raise ValueError('queue code changed')
    for name,expected in lock['evaluation_dependencies'].items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('locked scorer changed')
    event_lock=read(Path(__file__).with_name('event_audit_lock.json'))
    if sha(Path(__file__).with_name('event_audit_lock.json'))!=lock['event_lock_sha256']:raise ValueError('event definition changed')
    root=RESULTS/'20261126_inducing_latent_flow';rows=[];records={};sources={};curves={}
    if day==15:
        primary=read(root/'paired_calendar8_v2/summary.json')
        if primary.get('evaluation_amendment_sha256')!=sha(AMENDMENT):raise ValueError('primary summary amendment differs')
        require_supplement(primary,sha(LOCK),sha(__file__))
    for seed in (42,17):
        if source_receipt(seed)!=lock['sources'][str(seed)]:raise ValueError('frozen source changed')
        paths=source_paths(seed,day);cache=paths['cache'];embedding=paths['embeddings'];expected=lock['sources'][str(seed)]['calendars'][str(day)]
        require_calendar(read(cache/'protocol.json'),len(np.load(cache/'origins.npy')),np.load(cache/'query_ids.npy'),seed,
            sha(cache/'manifest.json'),expected['cache_manifest_sha256'])
        if sha(embedding/'manifest.json')!=expected['embedding_manifest_sha256']:raise ValueError('calendar embedding changed')
        validate_calendar_origins(np.load(cache/'origins.npy'),day)
        geometry,geo_sha=verified_geometry(cache);nearest=calendar_shape(np.load(cache/'nearest.npy'),12)
        if sha(cache/'nearest.npy')!=read(cache/'manifest.json')['files']['nearest.npy']:raise ValueError('condition distance changed')
        base=calendar_shape(np.load(embedding/'mean.npy'),12);truth0=calendar_shape(np.load(cache/'truth.npy'),12);anchor0=None;source0=None
        for family,domain in [('gaussian','time'),*ARMS]:
            label=family+'_'+domain;model=root/f'{label}_s{seed}';folder=root/f'calendar{day}/{label}_s{seed}'
            seal=None if family=='gaussian' else validate_model_seal(model,'COMPLETE_INDUCING_FLOW',codes())
            evaluation_marker(folder,seal);lin=read(folder/'lineage.json');metrics=read(folder/'metrics.json');joint=read(folder/'joint_metrics.json')
            require_pairing(lin,family,domain,seed,lock['sample_seed'],lock['evaluation_device'])
            if lin['code_sha256']!=codes() or lin['cache_manifest_sha256']!=expected['cache_manifest_sha256'] or lin['embedding_manifest_sha256']!=expected['embedding_manifest_sha256']:
                raise ValueError('evaluation source or implementation changed')
            if source0 is None:source0=lin['source'];anchor0=lin['anchors']
            if lin['source']!=source0 or lin['anchors']!=anchor0:raise ValueError('paired probability conditions differ')
            if lin['source']['weather_gp_weights_sha256']!=lock['sources'][str(seed)]['gp_weights_sha256']:
                raise ValueError('registered G265 changed')
            if seal:
                protocol=read(model/'protocol.json')
                if lin['source']!=seal['source'] or protocol['anchors']!=lin['anchors']:raise ValueError('model source or anchors differ')
                require_actual_completion(model,amendment['label_support'][str(seed)])
                curves[seed,label]=read(model/'curve.json')
            if joint['source_lineage_sha256']!=sha(folder/'lineage.json') or joint['code_sha256']!=lock['evaluation_dependencies']['probability_joint_audit.py']:
                raise ValueError('joint score source changed')
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];truth=data['truth'];prediction=data['prediction'];stored_crps=data['crps']
                np.testing.assert_array_equal(truth,truth0);np.testing.assert_array_equal(prediction,base)
                if samples.shape!=(32,3,12,72,137,7) or not np.isfinite(samples).all():raise ValueError('ensemble population differs')
                crps=fair_crps(samples,truth);np.testing.assert_allclose(stored_crps,crps,atol=2e-5,rtol=1e-6,equal_nan=True)
                r={f'crps_h{h}':monthly_record(crps[...,h],1) for h in (0,6)}
                np.testing.assert_allclose([r['crps_h0']['value'],r['crps_h6']['value']],np.asarray(metrics['fair_crps_mm_by_horizon'])[[0,6]],atol=1e-10,rtol=0)
                increment=fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1));r['increment']=monthly_record(increment,1)
                np.testing.assert_allclose(r['increment']['value'],joint['within_forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
                r.update(event_records(samples,truth,event_lock));r.update(strata_records(crps,geometry,nearest))
                for distance,table in joint['spatial_pair_crps'].items():
                    for h in ('h0','h6'):
                        v=table[h];r[f'pair_{distance}_{h}']={**v,'value':v['fair_crps_mm']}
                monthly=joint['energy']['monthly_mean_mm'];r['energy']=dict(value=joint['energy']['mean_mm'],monthly_score_sum=monthly,monthly_valid_count=[1]*12)
                if family in ('affine','gaussian'):
                    np.testing.assert_array_equal(data['gaussian_distribution_mean'],base)
                    analytic=gaussian_crps(base,data['gaussian_distribution_std'],truth)
                    r.update({f'analytic_crps_h{h}':monthly_record(analytic[...,h],1) for h in (0,6)})
                point=point_metrics(prediction,truth)
                np.testing.assert_allclose(point['horizon_macro_rmse_mm'],metrics['deterministic_mean']['horizon_macro_rmse_mm'],atol=1e-10,rtol=0)
            records[seed,label]=r
            rows.append(dict(seed=seed,name=label,records=r,coverage80=metrics['coverage80_by_horizon'],
                exact_point_R_F6_rmse_mm=point['horizon_macro_rmse_mm'][::6],sample_mean_R_F6_rmse_mm=metrics['sample_mean']['horizon_macro_rmse_mm'][::6],
                seconds_sampling_and_array_scoring=metrics['seconds']))
            sources[f'{seed}/{label}']=dict(lineage_sha256=sha(folder/'lineage.json'),metrics_sha256=sha(folder/'metrics.json'),joint_sha256=sha(folder/'joint_metrics.json'),
                evaluation_sha256=lin['prediction_sha256'],geometry_sha256=geo_sha,cache_manifest_sha256=expected['cache_manifest_sha256'])
            print(json.dumps(dict(seed=seed,name=label,F6=r['crps_h6']['value'],increment=r['increment']['value'])),flush=True)
            del samples,truth,prediction,stored_crps,crps,increment
    edges=[(f+'_'+d,'gaussian_time') for f,d in ARMS]+[(f'odd_{d}',f'affine_{d}') for d in ('time','frequency')]+[(f'{f}_frequency',f'{f}_time') for f in ('affine','odd')]
    comparisons={}
    for a,b in edges:
        keys=set.intersection(*(set(records[s,n]) for s in (42,17) for n in (a,b)))
        comparisons[a+'_vs_'+b]={key:safe_comparison([(records[s,a][key],records[s,b][key]) for s in (42,17)]) for key in sorted(keys)}
    out=root/f'paired_calendar{day}_v2';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,comparisons=comparisons,sources=sources,calendar_day=day,supplement_trigger=should_supplement(rows),
        primary=lock['primary'],lock_sha256=sha(LOCK),evaluation_amendment_sha256=sha(AMENDMENT),code_sha256=sha(__file__),strict_oof=False,confirmation_C_D_read=False,
        interval='2000 shared-month draws, fixed two seeds; exposed sampled2020; no multiplicity adjustment',analytic_point_mean_preserved=True))
    render(out,rows,comparisons,curves,day)


def render(out,rows,comparisons,curves,day):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['gaussian_time','affine_time','affine_frequency','odd_time','odd_frequency']
    value=lambda n,k:np.mean([r['records'][k]['value'] for r in rows if r['name']==n])
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    axes[0].bar(range(5),[value(n,'crps_h6') for n in names]);axes[0].set_xticks(range(5),names,rotation=30,ha='right')
    for n in names:axes[1].plot(range(4),[value(n,f'pair_{d}_h6') for d in ('0-50','50-150','150-300','300-inf')],'o-',label=n)
    for (seed,n),c in curves.items():axes[2].plot([r['step'] for r in c],[r['loss'] for r in c],'-' if seed==42 else '--',label=f'{n},s{seed}')
    axes[0].set(ylabel='F6 fair CRPS (mm)',title=f'Frozen point mean; calendar{day}')
    axes[1].set(xticks=range(4),xticklabels=['0-50','50-150','150-300','300+'],ylabel='F6 pair fair CRPS (mm)',xlabel='Station-pair distance (km)');axes[1].legend(fontsize=6)
    axes[2].set(xlabel='Chronological training block',ylabel='Joint proper-score loss');axes[2].legend(fontsize=6)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'inducing_comparison.png',dpi=170);plt.close(fig)
    lines=['# 共享诱导创新流：固定均值的联合概率评价','',
        f'2016–2019各3000连续块，种子42/17实际监督更新2999/3000；2020每月{day}日起72h三布局开发评价。Gaussian与四个流模型共享耦合GP/诱导噪声、冻结TF3000点均值和S175输入尺度。原残差strict_oof=false。','',
        '| 模型 | 当前CRPS/mm | 六小时CRPS/mm | 内部增量CRPS/mm | 近邻差值CRPS/mm | 全场energy/mm |',
        '|---|---:|---:|---:|---:|---:|']
    for n in names:lines.append('| '+n+' | '+' | '.join(f'{value(n,k):.6f}' for k in ('crps_h0','crps_h6','increment','pair_0-50_h6','energy'))+' |')
    lines+=['','## 配对比较','']
    for edge,table in comparisons.items():
        v=table['crps_h6'];lines.append(f"- {edge}：六小时CRPS收益{v['gain_percent']:.3f}%，差值{v['delta']:.6f}mm，95%月份区间{v['delta_ci95']}。")
    lines+=['','点输出为冻结基础场的解析均值，不用有限成员平均值宣称RMSE改善。Affine保持Gaussian；odd保持中心对称，不能表达所有偏态分布。此实验用终端联合proper评分训练概率流，不是传统去噪score matching。',
        '空间锚点不截断完整GP；训练Q32为几何/固定随机子集，评价Q137固定全人口。原Q128未来可用性选站、历史PWV筛选来源、末91块复用等限制继续保留，确认及原C/D标签未读取。','',
        '主指标为odd_frequency相对affine_frequency六小时fairCRPS；相对Gaussian及时间/频率、增量、站对、事件和地形分层分别归因。区间条件于两个种子及已暴露日期，未经多重探索校正。','',
        '![训练与联合评价](inducing_comparison.png)','','[完整指标与来源](summary.json)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(primary=comparisons['odd_frequency_vs_affine_frequency']['crps_h6'])),flush=True)


if __name__=='__main__':main()
