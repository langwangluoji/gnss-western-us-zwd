"""Shared-month, two-seed comparisons of native temporal means and continuation."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from campaign import ROOT,RESULTS,read

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_evaluation import point_metrics


def mean(x,axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(x,axis=axis)


def paired_interval(errors,refs,draws=2000):
    # S,L,M,T,Q,H: resample the same months in BOTH seeds and all layouts/stations.
    a=np.nansum(errors**2,3);b=np.nansum(refs**2,3)
    ac=np.isfinite(errors).sum(3);bc=np.isfinite(refs).sum(3)
    rng=np.random.default_rng(261126);delta=[]
    for _ in range(draws):
        index=rng.integers(a.shape[2],size=a.shape[2])
        def macro(square,count):
            numerator=square[:,:,index].sum(2);denominator=count[:,:,index].sum(2)
            mse=np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan),where=denominator>0)
            return mean(np.sqrt(mse),(0,1,2))
        delta.append(macro(a,ac)-macro(b,bc))
    return np.quantile(delta,[.025,.975],axis=0).T.tolist()


def load(folder):
    lineage=read(folder/'lineage.json')
    if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('prediction hash changed')
    with np.load(folder/'evaluation.npz') as data:
        prediction=data['prediction'].astype('float64');truth=data['truth'].astype('float64')
        reference=data['reference'].astype('float64')
    metrics=point_metrics(prediction,truth);registered=read(folder/'metrics.json')['deterministic_mean']
    np.testing.assert_allclose(metrics['horizon_macro_rmse_mm'],registered['horizon_macro_rmse_mm'],atol=1e-10,rtol=0)
    return prediction-truth,reference-truth,truth,metrics,lineage


def plain(v):
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fusion',action='store_true');args=p.parse_args();out=file_path(args.out);out.mkdir(parents=True,exist_ok=True)
    names=['gru72','lstm72','transformer72','gru72_physics_aux','lstm72_physics_aux','transformer72_physics_aux']
    models=[(n,'native',n) for n in names]
    if args.fusion:models += [(a+'_v2','fusion',a) for a in 'CHS']
    rows=[];allerrors={};source_hashes={};truths={};references={};curves={}
    for label,kind,name in models:
        for weights in ('last','ema'):
            errors=[];metrics=[];parameters=[];seconds=[]
            for seed in (42,17):
                folder=(RESULTS/'20261126_native_weather_means'/f'{name}_s{seed}' if kind=='native' else
                    RESULTS/'20261126_mean_fusion_v2'/f'{name}_s{seed}')
                status=read(folder/'status.json')
                if status['status']!='COMPLETE_MEAN' or status['steps']!=3000:raise ValueError('all models must complete3000 blocks')
                e,r,t,m,lineage=load(folder/f'eval_{weights}')
                cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}'
                if lineage['cache_manifest_sha256']!=sha(cache/'manifest.json'):raise ValueError('calendar changed')
                if seed in truths:np.testing.assert_array_equal(t,truths[seed]);np.testing.assert_array_equal(r,references[seed])
                else:truths[seed]=t;references[seed]=r
                errors.append(e);metrics.append(m);protocol=read(folder/'protocol.json')
                parameters.append(protocol['parameters']);seconds.append(status['seconds'])
                source_hashes[f'{label}_{weights}_s{seed}']=lineage['prediction_sha256']
                curves[f'{label}_s{seed}']=read(folder/'curve.json')
            key=label+'_'+weights;allerrors[key]=np.stack(errors)
            macro=np.mean([m['horizon_macro_rmse_mm'] for m in metrics],axis=0)
            rows.append(dict(name=key,parameters=parameters,seconds=seconds,horizon_macro_rmse_mm=macro.tolist(),
                per_seed_R_F6_mm=[[m['horizon_macro_rmse_mm'][h] for h in (0,6)] for m in metrics],
                current_increment_macro_rmse_mm=float(np.mean([m['current_hourly_increment_macro_rmse_mm'] for m in metrics])),
                R_F6_paired_delta_ci95_vs_I1=[paired_interval(allerrors[key][...,[h]],np.stack([references[s] for s in (42,17)])[...,[h]])[0] for h in (0,6)]))
    comparisons={}
    for kind in ('gru','lstm','transformer'):
        a=f'{kind}72_physics_aux_last';b=f'{kind}72_last'
        comparisons[f'{a}_vs_{b}']=paired_interval(allerrors[a][...,[0,6]],allerrors[b][...,[0,6]])
    for kind in ('gru','lstm'):
        a='transformer72_physics_aux_last';b=f'{kind}72_physics_aux_last'
        comparisons[f'{a}_vs_{b}']=paired_interval(allerrors[a][...,[0,6]],allerrors[b][...,[0,6]])
    if args.fusion:
        for arm in 'HS':
            a=f'{arm}_v2_last';b='C_v2_last'
            comparisons[f'{a}_vs_{b}']=paired_interval(allerrors[a][...,[0,6]],allerrors[b][...,[0,6]])
        comparisons['C_v2_last_vs_source']=paired_interval(allerrors['C_v2_last'][...,[0,6]],allerrors['transformer72_physics_aux_last'][...,[0,6]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    def display(name):
        name=name.removesuffix('_last')
        if name.endswith('_v2'):return name.removesuffix('_v2')+' continuation'
        return name.replace('transformer','TF').replace('gru','GRU').replace('lstm','LSTM').replace('72_physics_aux','72 joint')
    for row in rows:
        if not row['name'].endswith('_last'):continue
        axes[0].plot(range(7),row['horizon_macro_rmse_mm'],label=display(row['name']))
    axes[0].plot(range(7),np.mean([point_metrics(t+r,t)['horizon_macro_rmse_mm'] for t,r in zip(truths.values(),references.values())],0),'k--',label='I1 / persistence')
    axes[0].set(xlabel='Lead (h);0 = current field',ylabel='Station macro RMSE (mm)');axes[0].legend(fontsize=6)
    selected=[r for r in rows if r['name'].endswith('_last')]
    for i,row in enumerate(selected):
        axes[1].scatter([i,i],np.array(row['per_seed_R_F6_mm'])[:,1],color='#497c98',s=22)
    axes[1].set(xticks=range(len(selected)),xticklabels=[display(r['name']) for r in selected],ylabel='F6 RMSE per seed (mm)');axes[1].tick_params(axis='x',rotation=45)
    for label,curve in curves.items():
        if ('v2' in label if args.fusion else 'physics_aux' in label):
            axes[2].plot([r['step'] for r in curve],[r['query_loss'] for r in curve],label=display(label.split('_s')[0])+' s'+label.split('_s')[-1],alpha=.7)
    axes[2].set(xlabel='Chronological block',ylabel='Query training loss',title='Dates vary; these are training curves');axes[2].legend(fontsize=6)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    fig.suptitle('Two seeds, same 72h history and exposed 2020 calendar; R and F scored separately')
    fig.savefig(out/'paired_mean_comparison.png',dpi=170);plt.close(fig)
    payload=dict(seeds=[42,17],rows=rows,paired_R_F6_intervals_mm=comparisons,source_prediction_sha256=source_hashes,
        code_sha256=sha(Path(__file__)),figure_sha256=sha(out/'paired_mean_comparison.png'),
        bootstrap='2000 common month draws shared by both seeds/layouts/stations; exploratory exposed development',
        training='continuous3000x12h blocks per phase; fusion arms inherit source3000 then matched extra3000',
        strict_oof=False,confirmation_C_D_read=False)
    write_json(out/'paired_mean_summary.json',plain(payload))
    lines=['# 同历史逐站时间编码：双种子配对结果','','2016–2019连续训练；2020抽样开发日历、137注册查询站。每模型同72h有效历史、同站点抽样、同3000块。raw与EMA分别报告。C/H/S如列出则均从相同3000步源继续3000步。','',
        '| 模型 | R/mm | F6/mm | R小时增量/mm | 参数 |','|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['name']} | {row['horizon_macro_rmse_mm'][0]:.6f} | {row['horizon_macro_rmse_mm'][6]:.6f} | {row['current_increment_macro_rmse_mm']:.6f} | {row['parameters'][0]} |")
    lines+=['','| 配对比较 | R差值95%区间/mm | F6差值95%区间/mm |','|---|---|---|']
    for key,value in comparisons.items():lines.append(f'| {key} | {value[0]} | {value[1]} |')
    lines+=['','区间共享同月份抽样；它衡量已暴露开发日历的比较，模型选择未经多重比较校正。Transformer概率候选沿用seed42预先冻结的架构/last政策，seed17结果用于复核。','','![双种子性能与训练曲线](paired_mean_comparison.png)','',
        '全部时效、逐种子结果、训练耗时与来源哈希见[paired_mean_summary.json](paired_mean_summary.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(rows=len(rows),comparisons=comparisons)))


if __name__=='__main__':main()
