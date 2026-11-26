"""Separate finite-ensemble variance from the conditional-mean forecast error."""
import json
import numpy as np


def fair_mean_mse(samples,truth):
    x=np.asarray(samples,dtype='float64');K=len(x)
    if K<2:raise ValueError('at least2 independent members required')
    error=x-np.asarray(truth,dtype='float64')[None]
    return (K*error.mean(0)**2-np.mean(error**2,axis=0))/(K-1)


def macro_from_mse(mse):
    count=np.isfinite(mse).sum((1,2));total=np.nansum(mse,(1,2))
    average=np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0)
    negative=(average<0).sum((0,1))
    with np.errstate(invalid='ignore'):macro=np.nanmean(np.sqrt(average),axis=(0,1))
    return macro,negative


def main():
    import sys
    from pathlib import Path
    from campaign import ROOT,RESULTS,read
    ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
    sys.path.insert(0,str(ENGINE))
    from tools.terrain_diffusion.data import sha,write_json
    out=RESULTS/'20261126_diffusion_mean_sampling_audit';out.mkdir(parents=True,exist_ok=True)
    records={};rows=[];sources={}
    for seed in (42,17):
        baseline=None
        for mode in ('reference','full','local','affine','graph_affine'):
            folder=RESULTS/'20261126_gp_process'/f's{seed}'/mode
            if mode!='reference':folder=folder/'eval_heun0'
            lineage=read(folder/'lineage.json');metrics=read(folder/'metrics.json')
            if not lineage['independent_iid_member_noise']:raise ValueError('conditional mean estimator requires IID members')
            if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('samples changed')
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];truth=data['truth'];base=data['prediction']
            if baseline is None:baseline=(truth,base)
            else:
                np.testing.assert_array_equal(truth,baseline[0]);np.testing.assert_array_equal(base,baseline[1])
            fair=fair_mean_mse(samples,truth);K=len(samples)
            observed=(samples.mean(0,dtype='float64')-truth)**2
            variance=np.var(samples.astype('float64'),axis=0,ddof=1)/K
            np.testing.assert_allclose(fair,observed-variance,atol=1e-8,rtol=1e-8)
            macro,negative=macro_from_mse(fair);raw,_=macro_from_mse(observed);base_mse=(base.astype('float64')-truth)**2
            np.testing.assert_allclose(raw,metrics['sample_mean']['horizon_macro_rmse_mm'],atol=2e-6,rtol=0)
            valid=np.isfinite(truth);axes=(0,2,3)
            record=dict(seed=seed,mode=mode,members=K,raw_sample_macro_R_F6_mm=raw[::6].tolist(),
                fair_estimated_mean_macro_R_F6_mm=macro[::6].tolist(),negative_layout_station_aggregates_by_lead=negative.tolist(),
                baseline_macro_R_F6_mm=metrics['deterministic_mean']['horizon_macro_rmse_mm'][::6],
                monthly_fair_mse_sum=np.nansum(fair,axes).tolist(),monthly_valid_count=valid.sum(axes).tolist(),
                monthly_baseline_mse_sum=np.nansum(base_mse,axes).tolist(),
                mean_sampling_variance_by_lead_mm2=np.nanmean(np.where(valid,variance,np.nan),axis=(0,1,2,3)).tolist(),
                analytic_gaussian_mean_R_F6_mm=(metrics.get('gaussian_distribution_mean') or {}).get('horizon_macro_rmse_mm',[None]*7)[::6],
                nested_K_macro_R_F6_mm={})
            for n in (4,8,16,32):
                error=(samples[:n].mean(0,dtype='float64')-truth)**2
                record['nested_K_macro_R_F6_mm'][str(n)]=macro_from_mse(error)[0][::6].tolist()
            records[seed,mode]=record;rows.append(record)
            sources[f's{seed}/{mode}']=dict(array_sha256=lineage['prediction_sha256'],lineage_sha256=sha(folder/'lineage.json'))
            del fair,observed,variance,samples
            print(json.dumps(dict(seed=seed,mode=mode,raw=raw[::6].tolist(),fair_estimated_mean_macro=macro[::6].tolist())),flush=True)
    comparisons={};rng=np.random.default_rng(261126);draws=rng.integers(12,size=(2000,12))
    for mode in ('reference','full','local','affine','graph_affine'):
        for reference in ('base','reference'):
            if mode==reference:continue
            deltas=[]
            for seed in (42,17):
                row=records[seed,mode];count=np.asarray(row['monthly_valid_count'])
                numerator=np.asarray(row['monthly_fair_mse_sum'])
                ref=np.asarray(row['monthly_baseline_mse_sum'] if reference=='base' else records[seed,'reference']['monthly_fair_mse_sum'])
                delta=numerator-ref
                point=delta.sum(0)/count.sum(0)
                sampled=delta[draws].sum(1)/count[draws].sum(1)
                deltas.append((point,sampled))
            point=np.mean([x[0] for x in deltas],axis=0);sampled=np.mean([x[1] for x in deltas],axis=0)
            comparisons[mode+'_vs_'+reference]=dict(mean_mse_delta_R_F6_mm2=point[::6].tolist(),
                delta_ci95_R_F6_mm2=np.quantile(sampled[:,::6],[.025,.975],axis=0).T.tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for mode in ('reference','full','local','affine','graph_affine'):
        values=np.mean([[records[s,mode]['nested_K_macro_R_F6_mm'][str(k)] for k in (4,8,16,32)] for s in (42,17)],axis=0)
        for index,ax in enumerate(axes):ax.plot((4,8,16,32),values[:,index],'o-',label=mode)
    base=np.mean([records[s,'reference']['baseline_macro_R_F6_mm'] for s in (42,17)],axis=0)
    for j,ax in enumerate(axes):
        ax.axhline(base[j],ls='--',color='#333',label='frozen deterministic field')
        ax.set(xlabel='IID members K (nested prefixes)',ylabel='Observed ensemble-mean macro RMSE (mm)',title=('Current R','Forecast +6 h')[j],xticks=(4,8,16,32));ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'finite_ensemble_mean.png',dpi=170);plt.close(fig)
    write_json(out/'mean_sampling_audit.json',dict(rows=rows,comparisons=comparisons,sources=sources,code_sha256=sha(__file__),numpy_version=np.__version__,
        estimator='bar(error)^2 - unbiased_sample_variance/K = average products of distinct member errors; unbiased for conditional-mean MSE under IID members',
        boundary='Pointwise corrected estimates may be negative; no clipping. Square-root macro is descriptive plug-in, not unbiased RMSE. CIs are2000 common-month exploratory draws, shared across seeds; no multiplicity correction.',
        strict_oof=False,confirmation_C_D_read=False))
    lines=['# 扩散均值误差：有限采样与模型修正分离','','当前/6h为同一主日历，双种子平均。32成员为独立成员；基础场与真实标签逐值一致。','',
        '| 模型 | 原32成员均值R/mm | 扣除采样方差后的R估计/mm | 原32成员均值F6/mm | 扣除采样方差后的F6估计/mm |','|---|---:|---:|---:|---:|']
    for mode in ('reference','full','local','affine','graph_affine'):
        raw=np.mean([records[s,mode]['raw_sample_macro_R_F6_mm'] for s in (42,17)],0);fair=np.mean([records[s,mode]['fair_estimated_mean_macro_R_F6_mm'] for s in (42,17)],0)
        lines.append(f'| {mode} | {raw[0]:.6f} | {fair[0]:.6f} | {raw[1]:.6f} | {fair[1]:.6f} |')
    lines+=['','无偏性针对均方误差MSE：`E[(barY-y)^2-s²/K]=(E[Y]-y)^2`。逐点估计可负，保持原值；先按原layout/站点聚合，再取平方根得到上表描述性估计，平方根本身不具无偏性。正式比较使用下列micro MSE差值及月份区间，而非把表中小数当作精确分布均值RMSE。','',
        '| 比较 | 当前MSE差/mm² | 当前差值95%区间 | F6 MSE差/mm² | F6差值95%区间 |','|---|---:|---|---:|---|']
    for label,row in comparisons.items():
        a,b=row['mean_mse_delta_R_F6_mm2'];ci=row['delta_ci95_R_F6_mm2'];lines.append(f'| {label} | {a:.6f} | {ci[0]} | {b:.6f} | {ci[1]} |')
    lines+=['','相同月份共同重采样、共享两个种子；2020已暴露开发，区间为探索性且未多重比较校正。K曲线来自嵌套成员前缀，相邻点相关，不是独立重复实验。解析Gaussian均值在JSON单列，可检验估计量的有限成员波动。','',
        '![成员数与集合均值误差](finite_ensemble_mean.png)','','所有来源哈希、采样方差项与负值计数见[mean_sampling_audit.json](mean_sampling_audit.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'));print(json.dumps(comparisons))


if __name__=='__main__':main()
