"""Recompute saved forecast metrics and export comparison/stratification plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from .data import sha, write_json
from .temporal_experiment import score


def report(out):
    out = Path(out)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    data = dict(np.load(out/'development_labels.npz'))
    truth = data['truth']
    epochs = data['epochs'].astype('datetime64[h]')
    months = (epochs.astype('datetime64[M]').astype(int)%12)+1
    grouped, traces, strata, pair_rows = {}, [], [], []
    baseline = np.load(out/'persistence.npy')
    base_score = score(baseline, truth)
    recheck_error = 0
    distance = cdist(data['xy'], data['xy'])
    pair_i, pair_j = np.triu_indices(distance.shape[0],1)
    for row in summary:
        name = f"{row['kind']}_h{row['length']}"
        if row['kind']=='spatial_persistence':
            prediction = baseline
        else:
            folder = out/f"{name}_s{row['seed']}"
            prediction = np.load(folder/'prediction.npy')
            fit = json.loads((folder/'inner/fit.json').read_text(encoding='utf-8'))
            traces.append((f"{name} s{row['seed']}",fit['trace']))
        recomputed = score(prediction,truth)
        recheck_error = max(recheck_error, abs(recomputed['macro_rmse_mm']-row['metrics']['macro_rmse_mm']))
        grouped.setdefault(name,[]).append(recomputed)
        error = prediction-truth
        masks = [('JJA',np.broadcast_to(np.isin(months,[6,7,8])[:,None,None],truth.shape)),
                 ('DJF',np.broadcast_to(np.isin(months,[12,1,2])[:,None,None],truth.shape))]
        for feature, cuts in [('coast',[0,100,300,np.inf]),('height',[0,500,1500,np.inf])]:
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                selected = (data[feature]>=lo)&(data[feature]<hi)
                masks.append((f'{feature}_{lo}_{hi}',np.broadcast_to(selected[None,None],truth.shape)))
        for lo,hi in [(0,50),(50,100),(100,150),(150,np.inf)]:
            selected = (data['nearest']>=lo)&(data['nearest']<hi)
            masks.append((f'support_{lo}_{hi}',np.broadcast_to(selected[:,None],truth.shape)))
        for group,mask in masks:
            valid = mask & np.isfinite(truth)
            if not valid.any():
                continue
            station_counts = valid.sum((0,1))
            station_mse = np.divide(np.where(valid,error**2,0).sum((0,1)),station_counts,
                                    out=np.full(truth.shape[-1],np.nan),where=station_counts>0)
            strata.append(dict(kind=row['kind'],length=row['length'],seed=row['seed'],group=group,
                macro_rmse_mm=float(np.nanmean(np.sqrt(station_mse))),stations=int((station_counts>0).sum()),
                station_hours=int(valid.sum())))
        for lo,hi in [(0,50),(50,150),(150,np.inf)]:
            keep=(distance[pair_i,pair_j]>=lo)&(distance[pair_i,pair_j]<hi)
            i,j=pair_i[keep],pair_j[keep]
            if len(i):
                pair_error=(prediction[:,:,i]-prediction[:,:,j])-(truth[:,:,i]-truth[:,:,j])
                pair_rows.append(dict(kind=row['kind'],length=row['length'],seed=row['seed'],
                    distance_km=[lo,None if np.isinf(hi) else hi],pairs=len(i),
                    pair_difference_rmse_mm=float(np.sqrt(np.nanmean(pair_error**2)))))
    write_json(out/'stratified_metrics.json',strata)
    write_json(out/'pair_metrics.json',pair_rows)
    # Paired week-block bootstrap on the fixed population. Seed results are
    # averaged as metrics, not mistaken for independent weather observations.
    weeks, inverse = np.unique(data['epochs']//168, return_inverse=True)
    counts=np.stack([np.isfinite(truth[inverse==i]).sum((0,1)) for i in range(len(weeks))])
    rng=np.random.default_rng(117)
    weights=rng.multinomial(len(weeks),np.full(len(weeks),1/len(weeks)),size=1000)
    denominators=weights@counts
    def bootstrap_rmse(pred):
        sse=np.stack([np.nansum((pred[inverse==i]-truth[inverse==i])**2,axis=(0,1)) for i in range(len(weeks))])
        per_station=np.sqrt(np.divide(weights@sse,denominators,
            out=np.full(denominators.shape,np.nan),where=denominators>0))
        return np.nanmean(per_station,axis=1)
    base_boot=bootstrap_rmse(baseline)
    current_rows=[r for r in summary if r['kind']=='none']
    current_boot=np.mean([bootstrap_rmse(np.load(out/f"none_h{r['length']}_s{r['seed']}"/'prediction.npy'))
                          for r in current_rows],axis=0) if current_rows else None
    intervals={}
    for name in grouped:
        selected=[r for r in summary if f"{r['kind']}_h{r['length']}"==name]
        boots=[]
        for row in selected:
            pred=baseline if row['kind']=='spatial_persistence' else np.load(out/f"{name}_s{row['seed']}"/'prediction.npy')
            boots.append(bootstrap_rmse(pred))
        intervals[name]=dict(delta_vs_persistence_ci95_mm=np.quantile(np.mean(boots,axis=0)-base_boot,[.025,.975]).tolist(),
                             sampled_weeks=len(weeks),replicates=1000)
        if current_boot is not None:
            intervals[name]['delta_vs_current_frame_ci95_mm']=np.quantile(np.mean(boots,axis=0)-current_boot,[.025,.975]).tolist()
    write_json(out/'paired_block_intervals.json',intervals)
    write_json(out/'verification.json',dict(recomputed_macro_max_error_mm=recheck_error,
        predictions_checked=len(summary),truth_sha256=sha(out/'development_labels.npz')))
    names=list(grouped)
    means=[np.mean([x['macro_rmse_mm'] for x in grouped[n]]) for n in names]
    fig,ax=plt.subplots(figsize=(11,6))
    ax.barh(names,means,color=['#8391a3' if n=='spatial_persistence_h0' else '#247d91' for n in names])
    for index,value in enumerate(means):
        ax.text(value+.04,index,f'{value:.3f}',va='center',fontsize=8)
    ax.set_xlim(0,max(means)+.8)
    ax.axvline(base_score['macro_rmse_mm'],ls='--',color='#d06a45',label='Spatial persistence')
    ax.set_xlabel('Station macro RMSE (mm), six forecast hours');ax.legend();ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for name,trace in traces:
        ax.plot([v['step'] for v in trace],[v['inner_macro_rmse_mm'] for v in trace],label=name,alpha=.65)
    ax.set(xlabel='Updates',ylabel='2019 inner macro RMSE (mm)')
    ax.legend(fontsize=6,ncol=3);fig.tight_layout();fig.savefig(out/'inner_learning_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name in names:
        h=np.mean([x['horizon_macro_rmse_mm'] for x in grouped[name]],axis=0)
        ax.plot([1,3,6],h,marker='o',label=name)
    ax.set(xlabel='Forecast horizon (hour)',ylabel='Station macro RMSE (mm)')
    ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/'horizon_error_growth.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,groups in zip(axes,[['JJA','DJF'],['coast_0_100','coast_100_300','coast_300_inf']]):
        for name in names:
            rows=[s for s in strata if f"{s['kind']}_h{s['length']}"==name]
            vals=[np.mean([r['macro_rmse_mm'] for r in rows if r['group']==g]) for g in groups]
            ax.plot(range(len(groups)),vals,marker='o',label=name)
        ax.set_xticks(range(len(groups)),groups,rotation=20)
        ax.set_ylabel('Station macro RMSE (mm)')
    axes[-1].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'season_coast_strata.png',dpi=180);plt.close(fig)
    best_name=names[int(np.argmin(means))]
    selected=[r for r in summary if f"{r['kind']}_h{r['length']}"==best_name][0]
    if selected['kind']=='spatial_persistence':
        best_prediction=baseline
    else:
        best_prediction=np.load(out/f"{best_name}_s{selected['seed']}"/'prediction.npy')
    count=np.isfinite(truth).sum((0,1))
    station_rmse=np.sqrt(np.divide(np.nansum((best_prediction-truth)**2,axis=(0,1)),count,
                                  out=np.full(len(count),np.nan),where=count>0))
    fig,ax=plt.subplots(figsize=(7,7))
    art=ax.scatter(data['xy'][:,0],data['xy'][:,1],c=station_rmse,cmap='magma',s=25)
    ax.set(xlabel='LCC x (km)',ylabel='LCC y (km)',title=f'Development station errors: {best_name}')
    ax.set_aspect('equal');fig.colorbar(art,ax=ax,label='RMSE (mm)')
    fig.tight_layout();fig.savefig(out/'station_error_map.png',dpi=180);plt.close(fig)
    text=['# 连续时空预报：本机配对筛查','',
        '状态：真实本机训练与2020抽样开发评价；非全年正式实验。', '',
        '输入为条件站截至锚点的ZWD/T/P、地形与查询点海岸距离；隐藏目标站的所有历史不进入模型。',
        '2016–2018抽样拟合，2019选择更新数；从头在2016–2019事件上按冻结更新数重拟合，再计算2020开发分数。开发标签预先缓存，未进入梯度或内层选模。',
        'no-history分支指当前帧模型，仍共享观测年龄等可用性元数据；不同长度切片共享最长窗口计算的age，不是严格删去所有较早元数据的消融。',
        f"三个布局共144个开发事件（48个锚点），每事件生成连续6小时；137登记验证站中{base_score['station_count']}站有效，共{base_score['valid_count']}个有效站时评分记录（含布局重复）。",
        '空间为8×8固定载体、小型卷积下采样/跳连、精确查询点IDW持久性残差跳连；海岸距离当前是C1静态特征。',
        '本轮不代表p14复现，不含C2海岸边消息、折外残差diffusion或年度评价。', '',
        '| 分支 | 双种子平均宏RMSE/mm | 1/3/6小时RMSE/mm | 增量宏RMSE/mm |','|---|---:|---|---:|']
    for name in names:
        values=grouped[name]
        text.append(f"| {name} | {np.mean([x['macro_rmse_mm'] for x in values]):.3f} | "+
            '/'.join(f'{x:.3f}' for x in np.mean([v['horizon_macro_rmse_mm'] for v in values],axis=0))+
            f" | {np.mean([x['increment_macro_rmse_mm'] for x in values]):.3f} |")
    text+=['','本表点值使用确定性均值；未将模型平滑程度当作时间联合分布证据。',
           '同人口周块配对bootstrap区间见 `paired_block_intervals.json`；1000次重采样按天气周分块，种子指标取均值，未将种子当作独立天气样本。',
           '海岸、高程、支撑距离和夏冬分层见 `stratified_metrics.json`；站对差值见 `pair_metrics.json`。',
           '开发数据已暴露，后续调参须标记开发回看。168小时、多分辨率、完整年度及diffusion仍按计划独立推进。','',
           '![模型比较](model_comparison.png)','![内层学习曲线](inner_learning_curves.png)',
           '![预测时距](horizon_error_growth.png)','![季节与海岸](season_coast_strata.png)','![站点误差](station_error_map.png)']
    (out/'REPORT.md').write_bytes(('\n'.join(text)+'\n').encode('utf-8'))
    write_json(out/'figure_manifest.json',{p.name:sha(p) for p in out.glob('*.png')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out)
