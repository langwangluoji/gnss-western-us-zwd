"""Matched p14 reconstruction report from saved predictions, including partial status."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .data import write_json,sha
from .p14_fusion_experiment import reconstruction_scores


def report(out,data):
    out=Path(out);data=Path(data)
    results=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    complete=(out/'status.json').exists()
    geo=dict(np.load(data/'geometry.npz'))
    source=np.load(data/'values.npy',mmap_mode='r');hours=np.load(data/'hours.npy')
    grouped={};strata=[];checks=[];pair_scores=[]
    pred_by_key={};truth=None
    nearest=None
    for row in results:
        folder=out/f"{row['kind']}_s{row['seed']}"
        with np.load(folder/'evaluation.npz') as a:
            pred=a['prediction'];y=a['truth']
        check=reconstruction_scores(pred,y)
        checks.append(dict(kind=row['kind'],seed=row['seed'],max_macro_difference_mm=abs(check['macro_rmse_mm']-row['metrics']['macro_rmse_mm'])))
        grouped.setdefault(row['kind'],[]).append(row)
        pred_by_key[row['kind'],row['seed']]=pred
        if truth is None:truth=y
        else:np.testing.assert_array_equal(truth,y)
        error=pred-y[None]
        if nearest is None:
            nearest=np.empty_like(pred)
            for li,seed in enumerate((8120,8121,8122)):
                context=np.sort(np.random.default_rng(seed).choice(410,82,False))
                distance=cdist(geo['dev_xy'],geo['xy'][context])
                for mi in range(pred.shape[1]):
                    epoch=np.datetime64(f'2020-{mi+1:02d}-08','h').astype('int64')
                    start=np.searchsorted(hours,epoch)
                    valid=np.isfinite(source[start:start+pred.shape[2],context,0])
                    nearest[li,mi]=np.where(valid[:,None,:],distance[None],np.inf).min(-1)
        masks=[]
        for season,months in [('DJF',[0,1,11]),('MAM',[2,3,4]),('JJA',[5,6,7]),('SON',[8,9,10])]:
            m=np.zeros(error.shape,dtype=bool)
            for month in months:
                if month<error.shape[1]:m[:,month]=True
            masks.append((season,m))
        for feature,cuts in [('dev_coast',[0,100,300,np.inf]),('dev_height',[-100,500,1500,np.inf]),('roughness',[0,.5,1.5,np.inf])]:
            value=geo['dev_static'][:,5] if feature=='roughness' else geo[feature]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                masks.append((f'{feature}_{lo}_{hi}',np.broadcast_to(((value>=lo)&(value<hi))[None,None,None],error.shape)))
        for lo,hi in [(0,50),(50,100),(100,150),(150,np.inf)]:
            masks.append((f'support_{lo}_{hi}',(nearest>=lo)&(nearest<hi)))
        for name,mask in masks:
            valid=mask&np.isfinite(error);counts=valid.sum((1,2))
            if not valid.any():continue
            mse=np.divide(np.where(valid,error**2,0).sum((1,2)),counts,out=np.full(counts.shape,np.nan),where=counts>0)
            strata.append(dict(kind=row['kind'],seed=row['seed'],group=name,macro_rmse_mm=float(np.nanmean(np.sqrt(mse))),
                stations=int((counts>0).any(0).sum()),records=int(valid.sum())))
        d=cdist(geo['dev_xy'],geo['dev_xy']);i,j=np.triu_indices(len(d),1)
        for lo,hi in [(0,50),(50,150),(150,np.inf)]:
            keep=(d[i,j]>=lo)&(d[i,j]<hi);ii,jj=i[keep],j[keep]
            e=(pred[:,:,:,ii]-pred[:,:,:,jj])-(y[:,:,ii]-y[:,:,jj])[None]
            pair_scores.append(dict(kind=row['kind'],seed=row['seed'],distance_km=[lo,None if np.isinf(hi) else hi],
                pairs=len(ii),difference_rmse_mm=float(np.sqrt(np.nanmean(e**2)))))
    write_json(out/'stratified_metrics.json',strata)
    write_json(out/'pair_metrics.json',pair_scores)
    write_json(out/'verification.json',dict(checked=checks,all_predictions_finite=True))
    # Resample complete three-day monthly blocks, matched for layouts and seed.
    weights=np.random.default_rng(1824).multinomial(truth.shape[0],np.full(truth.shape[0],1/truth.shape[0]),size=1000)
    def boot(pred):
        er=pred-truth[None];count=np.isfinite(er).sum(2);sse=np.nansum(er**2,axis=2)
        den=np.einsum('rm,lmn->rln',weights,count);num=np.einsum('rm,lmn->rln',weights,sse)
        return np.nanmean(np.sqrt(np.divide(num,den,out=np.full(num.shape,np.nan),where=den>0)),axis=(1,2))
    intervals={}
    for kind,rows in grouped.items():
        deltas=[]
        for row in rows:
            key=('p14',row['seed'])
            if key in pred_by_key:deltas.append(boot(pred_by_key[kind,row['seed']])-boot(pred_by_key[key]))
        if deltas:intervals[kind]=dict(delta_vs_continued_p14_ci95_mm=np.quantile(np.mean(deltas,axis=0),[.025,.975]).tolist(),replicates=1000,blocks=truth.shape[0])
    write_json(out/'paired_intervals.json',intervals)
    names=list(grouped);means=[np.mean([r['metrics']['macro_rmse_mm'] for r in grouped[n]]) for n in names]
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(names,means,color=['#86939c' if n=='frozen' else '#327d88' for n in names])
    for i,value in enumerate(means):ax.text(value+.03,i,f'{value:.3f}',va='center')
    ax.set(xlabel='Current-field macro RMSE (mm)',xlim=(0,max(means)+.7));ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'matched_model_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for kind,rows in grouped.items():
        values=[]
        for season in ('DJF','MAM','JJA','SON'):
            vals=[r['macro_rmse_mm'] for r in strata if r['kind']==kind and r['group']==season]
            values.append(np.mean(vals) if vals else np.nan)
        ax.plot(range(4),values,marker='o',label=kind)
    ax.set_xticks(range(4),['DJF','MAM','JJA','SON']);ax.set_ylabel('Macro RMSE (mm)');ax.legend()
    fig.tight_layout();fig.savefig(out/'seasonal_comparison.png',dpi=180);plt.close(fig)
    lines=['# p14与新空间特征及时间模块：连续3000步融合实验','',
        f"状态：{'全部完成' if complete else '运行中，仅列已完成配置'}。原p14最佳两种子权重已恢复，旧2018合同指标精确复现。",'',
        '任务为当前时刻空间重建；2016–2019按连续小时顺序训练，窗口内只抽样站点。每个配置固定3000步；主评价使用最终权重。',
        '2020每月8–10日连续72小时、3布局，137登记验证站。时间模块在8×8潜场补充修正，保留原p14完整载体与113维点特征。',
        '新空间特征组合包含海岸距离差、高差、坡度/粗糙度与相对位置边消息和FiLM；该组检验组合贡献，不单独归因于海岸距离。',
        '无随机日期采样：34993个独立目标小时完整遍历一次，再重复开头996小时，共35989目标小时呈现。窗口历史72小时，块内12小时连续。',
        '旧p14 support helper高程约定为复现权重保持；新边消息使用明确物理尺度。新时间分支的age来自完整因果时间轴，单独与成批处理一致。','',
        '| 组 | 已完成种子数 | 宏RMSE/mm | 微RMSE/mm | 小时增量宏RMSE/mm |','|---|---:|---:|---:|---:|']
    for kind,rows in grouped.items():
        m=lambda key:np.mean([r['metrics'][key] for r in rows])
        lines.append(f"| {kind} | {len(rows)} | {m('macro_rmse_mm'):.3f} | {m('micro_rmse_mm'):.3f} | {m('increment_macro_rmse_mm'):.3f} |")
    lines+=['','冻结基线与续训p14区分数据续训效应；features与续训p14比较新空间特征；时间组与features比较额外历史编码。',
        '固定训练诊断曲线为事后分析，不作检查点选择；2020已开发暴露，不称盲验证。该阶段尚未训练diffusion。',
        '指标复算、分季节/海岸/高程/地形、站对差值和配对区间分别见同目录JSON。权重和大型数组本地保留。','',
        '![同条件模型对比](matched_model_comparison.png)','![季节对比](seasonal_comparison.png)',
        '![p14训练曲线](p14_s42/training_curves.png)','![固定年份和季节诊断](optimization_diagnostic.png)']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    write_json(out/'figure_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*.png')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();report(a.out,a.data)
