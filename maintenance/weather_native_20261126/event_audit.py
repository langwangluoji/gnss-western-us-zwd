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
    records={};rows=[];sources={};validity=None
    for seed in (42,17):
        for mode in modes:
            folder=roots/f's{seed}'/mode
            if args.calendar==8 and mode!='reference':folder=folder/'eval_heun0'
            lineage=read(folder/'lineage.json');expected=lock['sources'][f'calendar8/s{seed}/{mode}']
            if lineage['probability_weights_sha256']!=expected['weights_sha256'] or not lineage['independent_iid_member_noise']:raise ValueError('event weights or IID members differ')
            if args.calendar==8 and lineage['prediction_sha256']!=expected['prediction_sha256']:raise ValueError('frozen main-calendar predictions changed')
            if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('event sample array changed')
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];truth=data['truth'];delta=samples[...,6].astype('float64')-samples[...,0];target=truth[...,6].astype('float64')-truth[...,0]
            del samples,truth
            valid=np.isfinite(target)
            if validity is None:validity=valid
            else:np.testing.assert_array_equal(valid,validity)
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
    out=RESULTS/'20261126_frozen_event_audit'/f'calendar{args.calendar}';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'event_scores.json',dict(rows=rows,comparisons=comparisons,sources=sources,lock_sha256=sha(lockpath),code_sha256=sha(__file__),
        primary_threshold_mm=20,scope=lock['scope'],strict_oof=False,confirmation_C_D_read=False))
    lines=['# 六小时快速增湿/减湿概率诊断','','阈值10/20/30mm及主阈值20mm在读取事件分数前固定。目标为同一预报起点的ZWD(+6h)−ZWD(current)，正负两侧全部保留。2020为已暴露开发诊断。','',
        '| 比较 | 阈值/mm | 方向 | 指标 | 改善 | 差值95%区间 |','|---|---:|---|---|---:|---|']
    for label,row in comparisons.items():
        name,t,side,metric=label.split('/');lines.append(f"| {name} | {t} | {side} | {metric} | {row['gain_percent']:.2f}% | {row['delta_ci95']} |")
    lines+=['','fair Brier扣除有限IID成员概率估计方差，尾部加权fair CRPS通过对增量分布做上/下阈值截尾计算。它们均在全部有效起点上评分，不只在发生事件时评分；逐点fair估计可负，不裁零。共同月份2000次重采样、两个种子共享月份；未做多重比较校正。','',
        '本报告检查已冻结模型，不训练、调门控或更改权重。主日历对照含GraphAffine/Affine；补充日期仅检验冻结Full/Local相对原GP。事件计数、逐月和来源见[event_scores.json](event_scores.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in comparisons.items() if '/20/' in k}))


if __name__=='__main__':main()
