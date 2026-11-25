"""Independent float64, matched-source analysis for warm/Q precision campaigns."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def macro(squares,counts):
    ratio=np.divide(squares,counts,out=np.full(squares.shape,np.nan),where=counts>0)
    return float(np.nanmean(np.sqrt(ratio)))


def statistics(pred,truth):
    if pred.shape[1:]!=truth.shape or not np.isfinite(pred).all():
        raise ValueError('fixed scoring population requires finite predictions with matching shape')
    if not np.isfinite(truth).any():raise ValueError('no truth in fixed scoring population')
    error=pred.astype('float64')-truth.astype('float64')[None];ok=np.isfinite(error)
    inc=np.diff(pred.astype('float64'),axis=2)-np.diff(truth.astype('float64'),axis=1)[None]
    return dict(macro_rmse_mm=macro(np.nansum(error**2,(1,2)),ok.sum((1,2))),
        micro_rmse_mm=float(np.sqrt(np.nanmean(error**2))),mae_mm=float(np.nanmean(np.abs(error))),
        bias_mm=float(np.nanmean(error)),increment_macro_rmse_mm=macro(np.nansum(inc**2,(1,2)),np.isfinite(inc).sum((1,2))),
        valid_count=int(ok.sum()),effective_stations=int(ok.any((0,1,2)).sum()),
        negative_prediction_fraction=float((pred<0)[np.broadcast_to(np.isfinite(truth)[None],pred.shape)].mean()))


def bootstrap_macro(error,indices):
    squares=np.nansum(error**2,axis=2);counts=np.isfinite(error).sum(axis=2)
    return np.array([macro(squares[:,i].sum(1),counts[:,i].sum(1)) for i in indices])


def matched_contrast(records,draws,candidate,control):
    keys=[(case,seed) for case in (candidate,control) for seed in (42,17)]
    if not all(key in draws for key in keys):return None
    values={(r['case'],r['seed']):r['macro_rmse_mm'] for r in records}
    differences=[values[(candidate,s)]-values[(control,s)] for s in (42,17)]
    paired=sum(draws[(candidate,s)]-draws[(control,s)] for s in (42,17))/2
    interval=np.quantile(paired,[.025,.975]).tolist()
    return dict(candidate=candidate,control=control,delta_macro_rmse_mm=float(np.mean(differences)),
        common_month_two_seed_paired95_mm=interval,improves_both_seeds=all(d<0 for d in differences),
        negative_interval_upper=interval[1]<0)


def stratify(error,values,bins):
    rows=[]
    for low,high in zip(bins[:-1],bins[1:]):
        mask=np.isfinite(error)&(values>=low)&(values<high)
        e=np.where(mask,error,np.nan);n=int(mask.sum())
        rows.append(dict(lower=low,upper=None if np.isinf(high) else high,count=n,
            macro_rmse_mm=macro(np.nansum(e**2,(1,2)),mask.sum((1,2))) if n else None,
            micro_rmse_mm=float(np.sqrt(np.nanmean(e**2))) if n else None,
            mae_mm=float(np.nanmean(np.abs(e))) if n else None,bias_mm=float(np.nanmean(e)) if n else None))
    return rows


def geography(pred,truth,nearest,geo):
    error=pred.astype('float64')-truth[None];months=np.arange(1,13)[None,:,None,None]
    result={
        'nearest_current_condition_km':stratify(error,nearest,[0,50,100,200,np.inf]),
        'height_m':stratify(error,geo['dev_height'][None,None,None],[-1000,500,1500,np.inf]),
        'terrain_roughness_m':stratify(error,geo['dev_static'][None,None,None,:,5]*500,[0,100,300,np.inf]),
        'coast_km':stratify(error,geo['dev_coast'][None,None,None],[0,50,150,300,np.inf])}
    result['no_current_condition_count']=int((np.isfinite(error)&~np.isfinite(nearest)).sum())
    result['seasons']={}
    for label,ms in [('winter',[12,1,2]),('spring',[3,4,5]),('summer',[6,7,8]),('autumn',[9,10,11])]:
        err=np.where(np.isin(months,ms),error,np.nan)
        result['seasons'][label]=dict(macro_rmse_mm=macro(np.nansum(err**2,(1,2)),np.isfinite(err).sum((1,2))),
            micro_rmse_mm=float(np.sqrt(np.nanmean(err**2))),count=int(np.isfinite(err).sum()))
    xy=geo['dev_xy'];distance=np.linalg.norm(xy[:,None]-xy[None],axis=-1);i,j=np.triu_indices(len(xy),1)
    result['station_pairs']=[]
    for low,high in [(0,50),(50,150),(150,300)]:
        choose=(distance[i,j]>=low)&(distance[i,j]<high);a,b=i[choose],j[choose]
        err=error[...,a]-error[...,b];n=int(np.isfinite(err).sum())
        result['station_pairs'].append(dict(distance_km=[low,high],count=n,
            micro_rmse_mm=float(np.sqrt(np.nanmean(err**2))) if n else None))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--warm',type=Path,required=True);p.add_argument('--query',type=Path,nargs='+',required=True)
    p.add_argument('--station',type=Path,nargs='+')
    p.add_argument('--innovation',type=Path)
    args=p.parse_args();out=args.warm/'combined_precision';out.mkdir(exist_ok=True)
    wprotocol=read(args.warm/'protocol.json');source=Path(wprotocol['config']['source'])
    data=Path(wprotocol['config']['data']);data_hash=sha(data/'contract.json')
    original=read(source/'protocol.json')
    if wprotocol['data_sha256']!=data_hash or original['dataset_contract_sha256']!=data_hash:
        raise ValueError('original/warm/data contracts differ')
    geo=dict(np.load(data/'geometry.npz'));values=np.load(data/'values.npy',mmap_mode='r');hours=np.load(data/'hours.npy')
    from tools.terrain_diffusion.data import DEFAULT_DATA
    from tools.terrain_diffusion.unet_foundation import GEOMETRY
    inherited=[DEFAULT_DATA/'data.npz',GEOMETRY,data/'p14_grid_coast.npy']
    with np.load(inherited[0]) as old:
        moments=dict(mean=old['mean'].tolist(),std=old['std'].tolist())
    write_json(out/'inherited_input_lineage.json',dict(
        capture='Posthoc inherited artifact receipt; original run protocols remain unchanged',
        files=[dict(path=str(f),sha256=sha(f),size_bytes=f.stat().st_size) for f in inherited],
        normalization=moments,labels_from_inherited_pack_used_for_this_capture=False))
    nearest=[]
    for layout_seed in (8120,8121,8122):
        c=np.sort(np.random.default_rng(layout_seed).choice(410,82,False))
        distance=np.linalg.norm(geo['dev_xy'][:,None]-geo['xy'][c][None],axis=-1);rows=[]
        for month in range(1,13):
            h=int(np.datetime64(f'2020-{month:02d}-08','h').astype('int64'))+np.arange(72)
            index=np.searchsorted(hours,h);ok=np.isfinite(values[index[:,None],c[None],0])
            rows.append(np.where(ok[:,None],distance[None],np.inf).min(-1))
        nearest.append(rows)
    nearest=np.asarray(nearest);indices=np.random.default_rng(252511).integers(0,12,size=(1000,12))
    records=[];detailed={};draws={};baseline={};verification=[]
    for seed in (42,17):
        folder=source/f'gru_s{seed}';a=dict(np.load(folder/'evaluation.npz'));metrics=statistics(a['prediction'],a['truth'])
        baseline[seed]=a;records.append(dict(case='reference',seed=seed,**metrics))
        detailed[f'reference_s{seed}']=geography(a['prediction'],a['truth'],nearest,geo)
    runs=[('warm',args.warm)]+[('query',r) for r in args.query]
    if args.station is not None:runs.extend(('station',r) for r in args.station)
    if args.innovation is not None:runs.append(('innovation',args.innovation))
    for kind,root in runs:
        if not (root/'summary.json').exists():continue
        protocol=read(root/'protocol.json')
        expected=protocol.get('data_sha256')
        if expected is None and (root/'input_lineage.json').exists():expected=read(root/'input_lineage.json')['data_contract_sha256']
        if expected!=data_hash:raise ValueError(f'candidate data contract differs: {root}')
        for row in read(root/'summary.json'):
            if row['case']=='reference':continue
            if row['steps']!=3000:raise ValueError('formal precision analysis requires final3000-step candidates')
            seed=row['seed'];case=row['case'];folder=root/f'{case}_s{seed}'
            if (case,seed) in draws:raise ValueError('duplicate case/seed across runs; do not overwrite evidence')
            source_weight=source/f'gru_s{seed}/last.pt'
            if sha(source_weight)!=protocol['source_weights'][str(seed)]:raise ValueError('candidate/source weight hash differs')
            a=dict(np.load(folder/'evaluation.npz'));b=baseline[seed]
            if not np.array_equal(np.isnan(a['truth']),np.isnan(b['truth'])) or not np.array_equal(np.nan_to_num(a['truth']),np.nan_to_num(b['truth'])):raise ValueError('evaluation truth/cohort differs')
            metrics=statistics(a['prediction'],a['truth']);ref=statistics(b['prediction'],b['truth'])
            check=max(abs(metrics[k]-row['metrics'][k]) for k in ('macro_rmse_mm','micro_rmse_mm','mae_mm','bias_mm','increment_macro_rmse_mm'))
            if check>1e-5:raise ValueError('saved final metrics do not recompute')
            error=a['prediction'].astype('float64')-a['truth'][None];old=b['prediction'].astype('float64')-b['truth'][None]
            samples=bootstrap_macro(error,indices)-bootstrap_macro(old,indices);draws[(case,seed)]=samples
            record=dict(case=case,seed=seed,run=kind,steps=row['steps'],**metrics,
                fit_method=row.get('fit_method','AdamW gradient updates'),
                delta_macro_rmse_mm=metrics['macro_rmse_mm']-ref['macro_rmse_mm'],
                delta_micro_rmse_mm=metrics['micro_rmse_mm']-ref['micro_rmse_mm'],
                delta_mae_mm=metrics['mae_mm']-ref['mae_mm'],delta_bias_mm=metrics['bias_mm']-ref['bias_mm'],
                delta_increment_macro_rmse_mm=metrics['increment_macro_rmse_mm']-ref['increment_macro_rmse_mm'],
                monthly_paired95_mm=np.quantile(samples,[.025,.975]).tolist())
            records.append(record);detailed[f'{case}_s{seed}']=geography(a['prediction'],a['truth'],nearest,geo)
            verification.append(dict(case=case,seed=seed,prediction_sha256=sha(folder/'evaluation.npz'),
                source_weight_sha256=sha(source_weight),source_prediction_sha256=sha(source/f'gru_s{seed}/evaluation.npz'),
                all_five_primary_metrics_max_difference_mm=check,truth_exact=True))
    means=[]
    for case in sorted(set(r['case'] for r in records if r['case']!='reference')):
        rows=[r for r in records if r['case']==case]
        if {r['seed'] for r in rows}!={42,17}:continue
        paired=(draws[(case,42)]+draws[(case,17)])/2
        mean=float(np.mean([r['macro_rmse_mm'] for r in rows]));ref=float(np.mean([r['macro_rmse_mm'] for r in records if r['case']=='reference']))
        interval=np.quantile(paired,[.025,.975]).tolist()
        means.append(dict(case=case,macro_rmse_mm=mean,reference_macro_rmse_mm=ref,delta_macro_rmse_mm=mean-ref,
            common_month_two_seed_paired95_mm=interval,improves_both_seeds=all(r['delta_macro_rmse_mm']<0 for r in rows),
            negative_interval_upper=interval[1]<0))
    contrasts=[]
    for family in ('A','D'):
        for mode in ('raw','ema'):
            contrast=matched_contrast(records,draws,f'{family}_transformer_{mode}',f'S_transformer_{mode}')
            if contrast is not None:contrasts.append(contrast)
    for candidate,control in [('I1_history','I0_current'),('I2_seasonal','I1_history')]:
        contrast=matched_contrast(records,draws,candidate,control)
        if contrast is not None:contrasts.append(contrast)
    write_json(out/'comparison.json',dict(records=records,two_seed_means=means,direct_representation_contrasts=contrasts,
        bootstrap='1000 common month resamples shared across seeds/layouts; full72h block and station pairing',
        source='original saved GRU predictions, exact truth and source-weight hash matched',current_development_exposed=True))
    write_json(out/'geographic_and_pair_metrics.json',detailed)
    write_json(out/'verification.json',dict(records=verification,computed_in_float64=True,
        five_primary_metrics_verified=True,confirmation_C_D_labels_used=False))
    lines=['# 约10mm基础场：原模型、空间融合与查询融合的同队列分析','',
        '基准使用原GRU模型保存的2020预测，逐项核对真值与源权重哈希。所有结果为相同12个月×72小时×3布局，登记137站、135站有真值、327330个有效布局-小时-站记录。2020属于已暴露开发队列。','',
        '| 模型 | 种子 | 宏RMSE | 微RMSE | MAE | bias | 增量宏RMSE | 宏RMSE变化 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['case']} | {r['seed']} | {r['macro_rmse_mm']:.6f} | {r['micro_rmse_mm']:.6f} | {r['mae_mm']:.6f} | {r['bias_mm']:.6f} | {r['increment_macro_rmse_mm']:.6f} | {r.get('delta_macro_rmse_mm',0):.6f} |")
    lines+=['','单位均为mm。只纳入已完成最终3000步评价的结果；Qraw与Qema来自同一训练轨迹，不计为独立训练重复。T5_joint额外解冻旧时间模块原先固定为零的查询头518参数，应解释为“联合微调＋旧查询头激活”。','',
        '## 双种子及共同月份配对区间','', '| 模型 | 两种子均值RMSE/mm | 相对原参考变化/mm | 共同月份95%区间/mm |','|---|---:|---:|---|']
    for r in means:lines.append(f"| {r['case']} | {r['macro_rmse_mm']:.6f} | {r['delta_macro_rmse_mm']:.6f} | {r['common_month_two_seed_paired95_mm']} |")
    if any(r['case'].startswith('I') for r in records):
        lines+=['','I0/I1/I2累计3000个连续12小时训练块，以解析岭回归拟合3/9/27维校正头；这些不是3000次新增神经网络梯度更新。固定源为同一GRU+p14。各头训练线性代理目标，最终修正限制±5 mm。']
    if contrasts:
        lines+=['','## 输入、损失及创新历史的直接配对','',
            '| 候选−控制 | 两种子宏RMSE差/mm | 共同月份95%区间/mm | 两种子均改善 |',
            '|---|---:|---|---|']
        for r in contrasts:lines.append(f"| {r['candidate']}−{r['control']} | {r['delta_macro_rmse_mm']:.6f} | {r['common_month_two_seed_paired95_mm']} | {r['improves_both_seeds']} |")
    lines+=['','负数为改善；区间同时保持月份、布局、站点及种子配对。跨零时不把该结果称为稳定提升。海岸、地形、高程、最近有效条件站距离、季节和站对差值详见[分层结果](geographic_and_pair_metrics.json)。',
        '连续块内的一小时增量独立评分，跨月份不连差。所有五项主指标采用float64独立复算。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(completed_candidates=len(verification),two_seed_means=means)))


if __name__=='__main__':main()
