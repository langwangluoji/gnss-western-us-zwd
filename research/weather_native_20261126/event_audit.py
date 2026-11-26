"""Frozen, prethresholded six-hour wetting/drying probability diagnostics."""
import json
import numpy as np


def fair_brier(events,truth):
    K=len(events)
    if K<2:raise ValueError('fair Brier requires at least2 IID members')
    p=np.mean(events,axis=0,dtype='float64')
    return (p-truth)**2-p*(1-p)/(K-1)


def tail_crps(samples,truth,threshold,side):
    if side=='increase':x=np.maximum(samples,threshold);y=np.maximum(truth,threshold)
    elif side=='decrease':x=np.minimum(samples,-threshold);y=np.minimum(truth,-threshold)
    else:raise ValueError(side)
    K=len(x)
    if K<2:raise ValueError('fair CRPS requires at least2 IID members')
    weight=(2*np.arange(K)-K+1).reshape((K,)+(1,)*(x.ndim-1));ordered=np.sort(x,axis=0)
    return np.mean(np.abs(x-y[None]),axis=0)-(ordered*weight).sum(0)/(K*(K-1))


def validate_event_lineage(lineage,lock,calendar,seed,mode,config,cache_sha,embedding_sha,source):
    main=lock['sources'][f'calendar8/s{seed}/{mode}'];fixed=lock['sources'].get(f'calendar{calendar}/s{seed}/{mode}')
    if lineage['probability_weights_sha256']!=main['weights_sha256']:raise ValueError('event weights differ')
    if fixed and lineage['prediction_sha256']!=fixed['prediction_sha256']:raise ValueError('frozen calendar predictions changed')
    if config['calendar_day']!=calendar or config['seed']!=seed or config['months']!=12:raise ValueError('wrong event cache calendar')
    if lineage['cache_manifest_sha256']!=cache_sha or lineage['embedding_manifest_sha256']!=embedding_sha:raise ValueError('event cache or embedding differs')
    if lineage['source']!=source:raise ValueError('frozen event mean/Gaussian source differs')
    if not lineage['independent_iid_member_noise'] or lineage['members']!=32 or lineage['seed']!=261126:
        raise ValueError('event IID member recipe differs')
    if lineage['solver']!='heun0' or lineage['sampling_steps']!=20:raise ValueError('event sampler differs')


def validate_calendar_origins(origins,day):
    expected=np.concatenate([np.datetime64(f'2020-{month:02d}-{day:02d}T00','h').astype('int64')+np.arange(72)
        for layout in range(3) for month in range(1,13)]).reshape(216,12)
    if not np.array_equal(origins,expected):raise ValueError('actual event origin hours differ from registered2020 calendar')


def main():
    import argparse,sys
    from pathlib import Path
    from campaign import ROOT,RESULTS,read
    ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
    sys.path.insert(0,str(ENGINE))
    from tools.terrain_diffusion.data import sha,write_json
    p=argparse.ArgumentParser();p.add_argument('--calendar',type=int,choices=[8,15],default=8);args=p.parse_args()
    lockpath=Path(__file__).with_name('event_audit_lock.json');lock=read(lockpath)
    modes=('reference','full','local','affine','graph_affine') if args.calendar==8 else ('reference','full','local')
    roots=RESULTS/('20261126_gp_process' if args.calendar==8 else '20261126_gp_process_calendar15')
    records={};rows=[];sources={};reference_truth=None
    for seed in (42,17):
        cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar{args.calendar}_s{seed}'
        name=f'transformer72_physics_aux_last_dev_s{seed}' if args.calendar==8 else f'transformer72_physics_aux_last_calendar15_s{seed}'
        embedding=RESULTS/'20261126_native_weather_embeddings'/name
        config=read(cache/'protocol.json')['config'];manifest=read(cache/'manifest.json')
        if manifest['protocol_sha256']!=sha(cache/'protocol.json'):raise ValueError('event cache protocol changed')
        if sha(cache/'query_ids.npy')!=manifest['files']['query_ids.npy']:raise ValueError('event query IDs changed')
        if sha(cache/'origins.npy')!=manifest['files']['origins.npy']:raise ValueError('event origin hours changed')
        validate_calendar_origins(np.load(cache/'origins.npy'),args.calendar)
        ids=np.load(cache/'query_ids.npy');np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(137),ids.shape))
        expected_source=read(RESULTS/'20261126_gp_process'/f's{seed}/reference/lineage.json')['source']
        for mode in modes:
            folder=roots/f's{seed}'/mode
            if args.calendar==8 and mode!='reference':folder=folder/'eval_heun0'
            lineage=read(folder/'lineage.json')
            validate_event_lineage(lineage,lock,args.calendar,seed,mode,config,sha(cache/'manifest.json'),sha(embedding/'manifest.json'),expected_source)
            if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('event sample array changed')
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];truth=data['truth'];delta=samples[...,6].astype('float64')-samples[...,0];target=truth[...,6].astype('float64')-truth[...,0]
            if reference_truth is None:reference_truth=truth.copy()
            else:np.testing.assert_array_equal(truth,reference_truth)
            del samples,truth
            valid=np.isfinite(target)
            sources[f's{seed}/{mode}']=dict(prediction_sha256=lineage['prediction_sha256'],lineage_sha256=sha(folder/'lineage.json'))
            for threshold in lock['thresholds_mm']:
                for side in lock['sides']:
                    event=delta>=threshold if side=='increase' else delta<=-threshold
                    observed=(target>=threshold) if side=='increase' else (target<=-threshold)
                    observed=np.where(valid,observed.astype('float64'),np.nan)
                    brier=fair_brier(event,observed);tail=tail_crps(delta,target,threshold,side);axes=(0,2,3)
                    row=dict(seed=seed,mode=mode,threshold_mm=threshold,side=side,valid_count=int(valid.sum()),event_count=int(np.nansum(observed)),
                        event_rate=float(np.nanmean(observed)),fair_brier=float(np.nanmean(brier)),tail_weighted_fair_crps_mm=float(np.nanmean(tail)),
                        monthly_count=valid.sum(axes).tolist(),monthly_brier_sum=np.nansum(brier,axes).tolist(),monthly_tail_sum=np.nansum(tail,axes).tolist())
                    rows.append(row);records[seed,mode,threshold,side]=row
            del delta,target
            print(json.dumps(dict(seed=seed,mode=mode,stage='event_scores_complete')),flush=True)
    comparisons={};pairs=[('full','reference'),('local','reference')]
    if args.calendar==8:pairs+=[('full','graph_affine'),('local','affine')]
    rng=np.random.default_rng(261126);draws=rng.integers(12,size=(2000,12))
    for candidate,reference in pairs:
        for threshold in lock['thresholds_mm']:
            for side in lock['sides']:
                for metric,sums in [('fair_brier','monthly_brier_sum'),('tail_weighted_fair_crps_mm','monthly_tail_sum')]:
                    values=[];reference_values=[];ci=[]
                    for seed in (42,17):
                        a=records[seed,candidate,threshold,side];b=records[seed,reference,threshold,side]
                        n=np.asarray(a['monthly_count']);delta=np.asarray(a[sums])-b[sums]
                        values.append(a[metric]);reference_values.append(b[metric]);ci.append(delta[draws].sum(1)/n[draws].sum(1))
                    x=float(np.mean(values));ref=float(np.mean(reference_values))
                    comparisons[f'{candidate}_vs_{reference}/{threshold}/{side}/{metric}']=dict(candidate=x,reference=ref,
                        gain_percent=100*(ref-x)/ref,delta_ci95=np.quantile(np.mean(ci,axis=0),[.025,.975]).tolist())
    out=RESULTS/'20261126_frozen_event_audit_v2'/f'calendar{args.calendar}';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'event_scores.json',dict(rows=rows,comparisons=comparisons,sources=sources,lock_sha256=sha(lockpath),code_sha256=sha(__file__),
        primary_threshold_mm=20,scope=lock['scope'],strict_oof=False,confirmation_C_D_read=False))
    lines=['# 六小时快速增湿/减湿概率诊断','','阈值10/20/30mm及主阈值20mm在读取事件分数前固定。目标为同一预报起点的ZWD(+6h)−ZWD(current)，正负两侧全部保留。2020为已暴露开发诊断。','',
        '| 比较 | 阈值/mm | 方向 | 指标 | 改善 | 差值95%区间 |','|---|---:|---|---|---:|---|']
    for label,row in comparisons.items():
        name,t,side,metric=label.split('/');lines.append(f"| {name} | {t} | {side} | {metric} | {row['gain_percent']:.2f}% | {row['delta_ci95']} |")
    lines+=['','fair Brier扣除有限IID成员概率估计方差，尾部加权fair CRPS通过对增量分布做上/下阈值截尾计算。它们均在全部有效起点上评分，不只在发生事件时评分；这两个分数理论上非负，模型间配对差值可负。共同月份2000次重采样、两个种子共享月份；未做多重比较校正。','',
        '本报告检查已冻结模型，不训练、调门控或更改权重。主日历对照含GraphAffine/Affine；补充日期仅检验冻结Full/Local相对原GP。事件计数、逐月和来源见[event_scores.json](event_scores.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in comparisons.items() if '/20/' in k}))


if __name__=='__main__':main()
