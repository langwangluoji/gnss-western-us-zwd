"""Paired, source-checked weather-GP, transport and centered-shape evidence."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np


def monthly_record(values,month_axis):
    values=np.asarray(values,dtype='float64')
    axes=tuple(i for i in range(values.ndim) if i!=month_axis)
    sums=np.nansum(values,axis=axes);counts=np.isfinite(values).sum(axis=axes)
    if not counts.sum():raise ValueError('empty observation population')
    return dict(value=float(sums.sum()/counts.sum()),monthly_score_sum=sums.tolist(),monthly_valid_count=counts.tolist())


def paired_record(pairs,draws=2000):
    if not pairs:raise ValueError('empty comparison')
    rng=np.random.default_rng(261126)
    months=len(pairs[0][0]['monthly_score_sum']);index=rng.integers(months,size=(draws,months))
    deltas=[];candidate=[];reference=[]
    for a,b in pairs:
        ac=np.asarray(a['monthly_valid_count']);bc=np.asarray(b['monthly_valid_count'])
        if not np.array_equal(ac,bc):raise ValueError('paired observation population differs')
        if len(ac)!=months:raise ValueError('different month count')
        av=np.asarray(a['monthly_score_sum']);bv=np.asarray(b['monthly_score_sum'])
        if np.any(ac[index].sum(-1)==0):raise ValueError('empty bootstrap population')
        deltas.append((av[index].sum(-1)-bv[index].sum(-1))/ac[index].sum(-1))
        candidate.append(av.sum()/ac.sum());reference.append(bv.sum()/bc.sum())
    a=float(np.mean(candidate));b=float(np.mean(reference))
    return dict(candidate=a,reference=b,delta=a-b,gain_percent=100*(b-a)/b if b else None,
                delta_ci95=np.quantile(np.mean(deltas,axis=0),[.025,.975]).tolist())


def fair_mean_mse(samples,truth):
    if len(samples)<2:raise ValueError('two IID members required')
    mean=np.mean(samples,axis=0,dtype='float64')
    result=(mean-truth)**2-np.var(samples,axis=0,dtype='float64',ddof=1)/len(samples)
    return np.where(np.isfinite(truth),result,np.nan)


def event_records(samples,truth,lock):
    from event_audit import fair_brier,tail_crps
    delta=samples[...,6].astype('float64')-samples[...,0]
    target=truth[...,6].astype('float64')-truth[...,0];valid=np.isfinite(target);records={}
    for threshold in lock['thresholds_mm']:
        for side in lock['sides']:
            event=delta>=threshold if side=='increase' else delta<=-threshold
            observed=target>=threshold if side=='increase' else target<=-threshold
            observed=np.where(valid,observed.astype('float64'),np.nan)
            key=f'event{threshold}_{side}'
            records[key+'_brier']=monthly_record(fair_brier(event,observed),1)
            records[key+'_tail']=monthly_record(tail_crps(delta,target,threshold,side),1)
            records[key+'_brier']['event_count']=int(np.nansum(observed))
    return records


def validate_family(family,seed,mode,folder,solver=None):
    from campaign import ROOT,RESULTS,read
    from tools.terrain_diffusion.data import sha
    from tools.terrain_diffusion.native_model_seal import validate_model_seal,evaluation_marker
    from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
    from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import codes as transport_codes
    from centered_artifact_audit import centered_marker,joint_marker
    from weather_gp_supplement import extra_marker
    from event_audit import validate_calendar_origins
    gp=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265'
    seal=validate_model_seal(gp,'COMPLETE_WEATHER_GP',gp_codes())
    day=15 if family=='supplement' else 8
    cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar{day}_s{seed}'
    embname='transformer72_physics_aux_last_'+('calendar15' if day==15 else 'dev')+f'_s{seed}'
    embedding=RESULTS/'20261126_native_weather_embeddings'/embname
    manifest=read(cache/'manifest.json')
    if manifest['protocol_sha256']!=sha(cache/'protocol.json'):raise ValueError('calendar protocol changed')
    if sha(cache/'origins.npy')!=manifest['files']['origins.npy']:raise ValueError('calendar origins changed')
    if sha(cache/'query_ids.npy')!=manifest['files']['query_ids.npy']:raise ValueError('calendar query identities changed')
    ids=np.load(cache/'query_ids.npy');np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(137),ids.shape))
    validate_calendar_origins(np.load(cache/'origins.npy'),day)
    model=None
    if family=='centered':
        model=folder.parent if mode!='reference' else None
        centered_marker(folder,gp,cache,embedding,mode,model)
    elif family=='supplement':
        lock=read(Path(__file__).with_name('weather_gp_supplement_lock.json'))
        expected=lock['sources'][str(seed)]['c175']
        if mode!='reference':
            gp=gp.parent/mode;seal=validate_model_seal(gp,'COMPLETE_WEATHER_GP',gp_codes())
            if mode=='c175' and (seal['weights_sha256']!=expected['weights_sha256'] or seal['protocol_sha256']!=expected['protocol_sha256']):
                raise ValueError('frozen supplement G175 differs')
        extra_marker(folder,None if mode=='reference' else seal,expected['source'],cache,embedding)
    else:
        if mode=='reference':evaluation_marker(folder,seal)
        else:
            model=folder.parent
            transport_seal=validate_model_seal(model,'COMPLETE_WEATHER_TRANSPORT',transport_codes())
            evaluation_marker(folder,transport_seal)
        lin=read(folder/'lineage.json')
        expected=seal['source'] if mode=='reference' else {**seal['source'],
            'weather_gp_protocol_sha256':seal['protocol_sha256'],'weather_gp_weights_sha256':seal['weights_sha256'],
            'weather_gp_seal_sha256':sha(gp/'model_artifact_seal.json')}
        if lin['source']!=expected or lin['code_sha256']!=(gp_codes() if mode=='reference' else transport_codes()):
            raise ValueError('transport evaluation source differs')
        if lin['cache_manifest_sha256']!=sha(cache/'manifest.json') or lin['embedding_manifest_sha256']!=sha(embedding/'manifest.json'):
            raise ValueError('transport calendar differs')
        if mode!='reference' and (lin['solver']!=solver or lin['sampling_steps']!={'heun0':20,'ddim':40}[solver]):
            raise ValueError('transport solver differs')
    lin=read(folder/'lineage.json')
    if lin['members']!=32 or lin['seed']!=261126 or not lin['independent_iid_member_noise']:
        raise ValueError('IID member recipe differs')
    if lin['strict_oof'] or lin['confirmation_C_D_read']:raise ValueError('unexpected data role')
    joint=folder/'joint_metrics.json'
    if joint.exists():joint_marker(folder,Path(__file__).with_name('probability_joint_audit.py'))
    elif solver!='ddim':raise ValueError('missing primary joint evaluation')
    if model:
        audit=read(model/'query_audit.json')
        if audit['engineering_only'] or audit['source']!=lin['source']:raise ValueError('query audit source differs')
        weightkey='weights_sha256' if family=='centered' else 'transport_weights_sha256'
        if audit[weightkey]!=lin['probability_weights_sha256']:raise ValueError('query audit weights differ')
        if family=='centered':
            if max(r['analytic_mean_max_difference_mm'] for r in audit['rows'])!=0:raise ValueError('analytic mean changed')
            if max(r['coefficient_unit_variance_error'] for r in audit['rows'])>1e-5:raise ValueError('Hermite variance normalization failed')
    return lin


def collect(family,seed,label,folder,mode,solver=None):
    from campaign import read
    from tools.terrain_diffusion.data import sha
    from tools.terrain_diffusion.native_weather_evaluation import fair_crps,gaussian_crps
    lineage=validate_family(family,seed,mode,folder,solver)
    metrics=read(folder/'metrics.json')
    with np.load(folder/'evaluation.npz') as data:
        truth=data['truth'];prediction=data['prediction'];samples=data['samples'];crps=data['crps']
        if samples.shape!=(32,3,12,72,137,7):raise ValueError('unregistered evaluation population')
        if not np.isfinite(samples).all():raise ValueError('nonfinite generated member')
        records={f'crps_h{h}':monthly_record(crps[...,h],1) for h in range(7)}
        np.testing.assert_allclose([records[f'crps_h{h}']['value'] for h in range(7)],metrics['fair_crps_mm_by_horizon'],atol=1e-6,rtol=0)
        # Recompute all marginal scores from actual members, not only stored CRPS.
        for h in (0,6):
            check=fair_crps(samples[...,h],truth[...,h])
            np.testing.assert_array_equal(check.astype(crps.dtype),crps[...,h])
        increments=[]
        for h in range(6):increments.append(fair_crps(samples[...,h+1]-samples[...,h],truth[...,h+1]-truth[...,h]))
        records['increment']=monthly_record(np.stack(increments,axis=-1),1)
        np.testing.assert_allclose(records['increment']['value'],metrics['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
        del increments
        event_lock=Path(__file__).with_name('event_audit_lock.json')
        records.update(event_records(samples,truth,read(event_lock)))
        for h in (0,6):
            records[f'fair_mean_mse_h{h}']=monthly_record(fair_mean_mse(samples[...,h],truth[...,h]),1)
            records[f'base_mse_h{h}']=monthly_record((prediction[...,h].astype('float64')-truth[...,h])**2,1)
        if family=='centered':
            np.testing.assert_array_equal(data['distribution_mean'],prediction)
            if np.any(data['distribution_std']<=0):raise ValueError('nonpositive analytical standard deviation')
            if mode in ('reference','scale'):
                analytic=gaussian_crps(data['distribution_mean'].astype('float64'),data['distribution_std'].astype('float64'),truth)
                np.testing.assert_allclose(np.nanmean(analytic,axis=(0,1,2,3)),metrics['analytic_gaussian_crps_mm_by_horizon'],atol=2e-7,rtol=0)
                for h in range(7):records[f'analytic_crps_h{h}']=monthly_record(analytic[...,h],1)
        elif family=='supplement' or mode=='reference':
            for h in range(7):
                analytic=gaussian_crps(data['gaussian_distribution_mean'][...,h].astype('float64'),data['gaussian_distribution_std'][...,h].astype('float64'),truth[...,h])
                records[f'analytic_crps_h{h}']=monthly_record(analytic,1)
            if family=='supplement':
                frozen=(data['gaussian_distribution_mean'],data['gaussian_distribution_std'])
        if (folder/'joint_metrics.json').exists():
            joint=read(folder/'joint_metrics.json')
            for distance,row in joint['spatial_pair_crps'].items():
                for lead in ('h0','h6'):
                    v=row[lead];records[f'pair_{distance}_{lead}']={**v,'value':v['fair_crps_mm']}
            monthly=joint['energy']['monthly_mean_mm']
            records['energy']=dict(value=joint['energy']['mean_mm'],monthly_score_sum=monthly,monthly_valid_count=[1]*12)
        if family=='supplement':
            for distance,row in metrics['analytic_station_pair_crps'].items():
                for lead in ('h0','h6'):
                    v=row[lead];records[f'analytic_pair_{distance}_{lead}']={**v,'value':v['analytic_crps_mm']}
        source_hashes={name:sha(folder/name) for name in ('metrics.json','lineage.json')}
        source_hashes['event_threshold_lock.json']=sha(event_lock)
        if (folder/'joint_metrics.json').exists():source_hashes['joint_metrics.json']=sha(folder/'joint_metrics.json')
        row=dict(seed=seed,name=label,records=records,coverage=metrics['coverage80_by_horizon'],rank3_30_coverage=metrics['rank3_30_coverage'],
            sample_mean_R_F6_rmse_mm=metrics['sample_mean']['horizon_macro_rmse_mm'][::6],
            deterministic_R_F6_rmse_mm=metrics['deterministic_mean']['horizon_macro_rmse_mm'][::6],
            analytic_mean_R_F6_rmse_mm=metrics.get('analytic_distribution_mean',metrics.get('gaussian_distribution_mean',{})).get('horizon_macro_rmse_mm',[None]*7)[::6],
            source=lineage['source'],source_files=source_hashes,source_prediction_sha256=lineage['prediction_sha256'])
    print(json.dumps(dict(family=family,seed=seed,model=label,R=records['crps_h0']['value'],F6=records['crps_h6']['value'],increment=records['increment']['value'])),flush=True)
    return row,truth,prediction,frozen if family=='supplement' else None


def main():
    from campaign import RESULTS,STOP
    from tools.terrain_diffusion.data import sha,write_json
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['centered','transport','supplement'],required=True);args=parser.parse_args();family=args.family
    if STOP.exists():raise SystemExit(75)
    group={'centered':'20261126_centered_shape','transport':'20261126_weather_transport','supplement':'20261126_weather_gp_calendar15_cpu'}[family]
    root=RESULTS/group;rows=[];records={}
    labels=['reference','scale','shape','both'] if family=='centered' else ['reference','local_heun0','affine_heun0','local_ddim','affine_ddim'] if family=='transport' else ['reference','c175','c265']
    for seed in (42,17):
        baseline=None
        for label in labels:
            if STOP.exists():raise SystemExit(75)
            solver=label.split('_')[-1] if family=='transport' and label!='reference' else None
            mode=label.split('_')[0] if solver else label
            folder=root/f's{seed}'/mode
            if family=='transport':folder=RESULTS/'20261126_weather_projective_gp'/f's{seed}/c265/eval' if mode=='reference' else folder/f'eval_{solver}'
            elif family=='centered' and mode!='reference':folder=folder/'eval'
            row,truth,prediction,frozen=collect(family,seed,label,folder,mode,solver)
            if baseline is None:baseline=(truth,prediction,frozen)
            else:
                np.testing.assert_array_equal(truth,baseline[0]);np.testing.assert_array_equal(prediction,baseline[1])
                if frozen is not None:
                    for a,b in zip(frozen,baseline[2]):np.testing.assert_array_equal(a,b)
            rows.append(row);records[seed,label]=row['records']
    comparisons={}
    edges=[(label,'reference') for label in labels[1:]]
    if family=='centered':edges += [('both','scale'),('both','shape'),('shape','scale')]
    elif family=='transport':edges += [('local_heun0','affine_heun0'),('local_ddim','affine_ddim')]
    else:edges += [('c265','c175')]
    for a,b in edges:
        shared=set(records[42,a])&set(records[42,b])&set(records[17,a])&set(records[17,b])
        comparisons[a+'_vs_'+b]={key:paired_record([(records[s,a][key],records[s,b][key]) for s in (42,17)]) for key in sorted(shared)}
    mean_gaps={label:{f'h{h}':paired_record([(records[s,label][f'fair_mean_mse_h{h}'],records[s,label][f'base_mse_h{h}']) for s in (42,17)]) for h in (0,6)} for label in labels}
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(family=family,rows=rows,comparisons=comparisons,fair_conditional_mean_MSE_gap_mm2=mean_gaps,
        exact_truth_and_base_prediction_verified=True,analytic_mean_equals_base_verified=family=='centered',
        exact_Gaussian_marginals_verified=family=='supplement',strict_oof=False,confirmation_C_D_read=False,
        interval='2000 shared-month draws across both seeds; equal seed weights; exploratory exposed2020; no multiplicity correction',
        code_sha256=sha(__file__),dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('event_audit.py','event_audit_lock.json','centered_artifact_audit.py','weather_gp_supplement.py')},
        interval_scope='month uncertainty conditional on the two fixed training seeds; no population-of-seeds interval'))
    render(family,out,labels,rows,comparisons,mean_gaps)


def render(family,out,labels,rows,comparisons,mean_gaps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def value(label,key):return float(np.mean([r['records'][key]['value'] for r in rows if r['name']==label]))
    primary=[s for s in labels if not s.endswith('ddim')]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for label in primary:
        axes[0].plot(range(7),[value(label,f'crps_h{h}') for h in range(7)],'o-',label=label)
        if all('pair_0-50_h6' in r['records'] for r in rows if r['name']==label):
            axes[1].plot(range(4),[value(label,f'pair_{d}_h6') for d in ('0-50','50-150','150-300','300-inf')],'o-',label=label)
    axes[0].set(xlabel='Lead (h)',ylabel='K32 fair marginal CRPS (mm)');axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(4),xticklabels=['0-50','50-150','150-300','300+'],xlabel='Pair distance (km)',ylabel='F6 pair fair CRPS (mm)');axes[1].legend(fontsize=8)
    for label in primary:
        v=np.mean([r['rank3_30_coverage'] for r in rows if r['name']==label],axis=0)
        axes[2].plot(range(7),v,'o-',label=label)
    axes[2].axhline(27/33,color='gray',ls='--');axes[2].set(xlabel='Lead (h)',ylabel='Rank3/30 coverage; IID reference 81.82%');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Two seeds, same evaluation population: {family}');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    title={'centered':'固定基础场均值：尺度与非Gaussian形状','transport':'天气G265之上的Local与Affine轨迹传输','supplement':'冻结天气GP：补充日期双种子验证'}[family]
    lines=[f'# {title}','','2016–2019训练；2020抽样日期为开发暴露。双种子、K32 IID成员；不代表全年或严格折外。','',
        '| 模型 | 当前CRPS/mm | F6 CRPS/mm | 内部增量CRPS/mm | 0–50km F6差值CRPS/mm | energy/mm |','|---|---:|---:|---:|---:|---:|']
    for label in labels:
        def optional(key):return f'{value(label,key):.6f}' if key in next(r for r in rows if r['name']==label)['records'] else 'NA'
        lines.append(f"| {label} | {value(label,'crps_h0'):.6f} | {value(label,'crps_h6'):.6f} | {value(label,'increment'):.6f} | {optional('pair_0-50_h6')} | {optional('energy')} |")
    lines+=['','| 比较 | 指标 | 改善 | 差值95%区间 |','|---|---|---:|---|']
    for label,table in comparisons.items():
        for key in ['crps_h0','crps_h6','increment','pair_0-50_h6','energy','analytic_pair_0-50_h6']:
            if key in table:
                v=table[key];lines.append(f"| {label} | {key} | {v['gain_percent']:.3f}% | {v['delta_ci95']} |")
    lines+=['','## 均值与作用范围','',
        '概率分数表使用同一总体上的K32 fair估计；Gaussian解析分数及四距离档、覆盖、逐月和来源另列JSON。DDIM为边际及增量采样器复核，未另做全场joint评分，表中NA表示未评估。',
        '均值诊断以(barY-y)^2-s²/K估计条件均值MSE；与确定性基础场相减，单位mm²。它可逐点为负，不等于生成样本自身RMSE，也不宜开方后声称精确RMSE。','',
        '| 模型 | 当前均值MSE相对基础场/mm² | 95%区间 | F6差值/mm² | 95%区间 |','|---|---:|---|---:|---|']
    for label,gap in mean_gaps.items():lines.append(f"| {label} | {gap['h0']['delta']:.6f} | {gap['h0']['delta_ci95']} | {gap['h6']['delta']:.6f} | {gap['h6']['delta_ci95']} |")
    lines+=['','## 六小时快速变化事件','',
        '沿用先前冻结的10/20/30mm阈值，主阈值20mm；将相同定义应用于本轮新模型，属开发探索。增湿与减湿均在全部有效起点上评分。所有阈值与事件计数见JSON。','',
        '| 比较 | 方向 | 指标 | 改善 | 差值95%区间 |','|---|---|---|---:|---|']
    for label,table in comparisons.items():
        if label.startswith('local_ddim') or label.startswith('affine_ddim'):continue
        for side in ['increase','decrease']:
            for metric in ['brier','tail']:
                v=table[f'event20_{side}_{metric}'];lines.append(f"| {label} | {side} | {metric} | {v['gain_percent']:.3f}% | {v['delta_ci95']} |")
    if family=='centered':lines+=['','三臂和reference的解析分布均值均逐值等于TF3000基础场，约9.818mm；K32样本均值含有限采样误差。reference与scale为Gaussian；shape/both为Hermite场，不提供虚构的Gaussian解析CRPS。']
    elif family=='transport':lines+=['','reference为冻结G265及S175解析中心/尺度，非严格TF3000中心。Local/Affine共享天气条件和GP初值；训练为逐查询7h边际score，未声称学习完整真实空间联合score。Heun0主评价39NFE，DDIM复核40NFE。']
    else:lines+=['','日期为每月15–17日。按读取成绩前冻结的政策检验全部final3000模型，不按此成绩挑选权重。CPU共同后端/噪声，基础场、Gaussian解析均值及尺度逐值相同；该验证不包含新Local/Affine。']
    lines+=['','月份共同重采样2000次，两个种子共享月份，区间未经多重比较校正。确认及原C/D标签未读取。','',
        '![概率分数、空间差值与覆盖](comparison.png)','','完整记录见[summary.json](summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(family=family,comparisons={name:{k:v for k,v in table.items() if k in ('crps_h6','increment','pair_0-50_h6','analytic_pair_0-50_h6')} for name,table in comparisons.items()})),flush=True)


if __name__=='__main__':main()
