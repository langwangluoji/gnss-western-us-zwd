"""Descriptive station/tail audit of sealed directed means; no fitting or new selection."""
from pathlib import Path
import json
import numpy as np
from campaign import RESULTS,read
from directed_v2_summary import locations
from lead_mean_audit import verified_geometry
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_paths import file_path


def station_statistics(error):
    # [layout,month,hour,query,lead]; zero-label stations remain NA.
    x=error.reshape(-1,error.shape[-2],error.shape[-1]);valid=np.isfinite(x);count=valid.sum(0)
    def mean(values):return np.divide(np.nansum(values,0),count,out=np.full(count.shape,np.nan),where=count>0)
    layout_count=np.isfinite(error).sum((1,2));layout_mse=np.divide(np.nansum(error*error,(1,2)),layout_count,
        out=np.full(layout_count.shape,np.nan),where=layout_count>0)
    available=(layout_count>0).sum(0)
    rmse=np.divide(np.nansum(np.sqrt(layout_mse),0),available,out=np.full(count.shape,np.nan),where=available>0)
    mae=mean(abs(x));bias=mean(x)
    percentiles=np.full((2,*count.shape),np.nan)
    for q,h in zip(*np.nonzero(count)):
        percentiles[:,q,h]=np.quantile(abs(x[:,q,h][valid[:,q,h]]),[.9,.95])
    return dict(count=count,rmse=rmse,mae=mae,bias=bias,p90=percentiles[0],p95=percentiles[1])


def main():
    root=RESULTS/'20261126_directed_lag_v2';source=read(RESULTS/'20261126_native_weather_cache_v2/calendar8_s42/protocol.json')
    data=file_path(source['config']['data'])
    if sha(data/'contract.json')!=source['data_sha256']:raise ValueError('original data contract changed')
    if sha(data/'geometry.npz')!=read(data/'contract.json')['files']['geometry.npz']:raise ValueError('original station identity geometry changed')
    with np.load(data/'geometry.npz') as g:ids=g['dev_ids'].copy();xy=g['dev_xy'].copy()
    reports={};sources={};plot={}
    for day in (8,15):
        summary=read(root/f'paired_calendar{day}/summary.json');group={};overall={}
        for arm in ('H_delta','factorized','joint'):
            per_seed=[];global_seed=[]
            for seed in (42,17):
                _,_,evaluation=locations(RESULTS,arm,seed,'last',day)
                expected=summary['sources'][f'{arm}/last/s{seed}']['prediction_sha256']
                if sha(evaluation/'evaluation.npz')!=expected:raise ValueError('sealed prediction changed')
                with np.load(evaluation/'evaluation.npz') as z:error=z['prediction'].astype('float64')-z['truth'].astype('float64')
                stats=station_statistics(error);per_seed.append(stats)
                for h in (0,6):
                    value=np.nanmean(stats['rmse'][:,h]);row=next(r for r in summary['rows'] if r['arm']==arm and r['weights']=='last')
                    np.testing.assert_allclose(value,row['per_seed'][(42,17).index(seed)]['horizon_macro_rmse_mm'][h],atol=1e-10,rtol=0)
                valid=np.isfinite(error);global_seed.append({str(h):dict(macro_rmse=float(np.nanmean(stats['rmse'][:,h])),
                    micro_rmse=float(np.sqrt(np.nanmean(error[...,h]**2))),mae=float(np.nanmean(abs(error[...,h]))),
                    bias=float(np.nanmean(error[...,h])),p90=float(np.quantile(abs(error[...,h][valid[...,h]]),.9)),
                    p95=float(np.quantile(abs(error[...,h][valid[...,h]]),.95))) for h in (0,6)})
                sources[f'{day}/{arm}/{seed}']=expected
            group[arm]={n:np.mean([s[n] for s in per_seed],0) for n in ('rmse','mae','bias','p90','p95')}
            group[arm]['count']=per_seed[0]['count'];overall[arm]={h:{k:float(np.mean([s[h][k] for s in global_seed])) for k in global_seed[0][h]} for h in ('0','6')}
        geometry,geo_sha=verified_geometry(RESULTS/f'20261126_native_weather_cache_v2/calendar{day}_s42')
        stations=[]
        for q,name in enumerate(ids):
            def scalar(v):return float(v) if np.isfinite(v) else None
            row=dict(station=str(name),query_index=q,height_m=float(geometry[q,2]*3000),coast_km=float(geometry[q,8]*500),
                valid_h0_h6=group['joint']['count'][q,[0,6]].tolist())
            row['models']={arm:{str(h):{n:scalar(stats[n][q,h]) for n in ('rmse','mae','bias','p90','p95')} for h in (0,6)} for arm,stats in group.items()}
            row['F6_joint_minus_H_delta_mm']=scalar(group['joint']['rmse'][q,6]-group['H_delta']['rmse'][q,6])
            stations.append(row)
        worst=sorted([r for r in stations if r['models']['joint']['6']['rmse'] is not None],key=lambda r:r['models']['joint']['6']['rmse'],reverse=True)[:10]
        reports[str(day)]=dict(overall=overall,stations=stations,worst10_joint_F6=[r['station'] for r in worst],
            preselected_cases=[r for r in stations if r['station'] in ['PLTX','P475','PTEX','P510','P786']],geometry_sha256=geo_sha)
        plot[day]=group
    out=root/'station_tail_audit';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(calendars=reports,sources=sources,code_sha256=sha(__file__),source_geometry_sha256=sha(data/'geometry.npz'),
        interpretation='descriptive fixed two-seed station/tail errors, no new station-specific significance tests',strict_oof=False,confirmation_C_D_read=False))
    render(out,reports,plot,xy,ids)


def render(out,reports,plot,xy,ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for row,day in enumerate((8,15)):
        g=plot[day];value=g['joint']['rmse'][:,6];delta=value-g['H_delta']['rmse'][:,6]
        a=axes[row,0].scatter(xy[:,0],xy[:,1],c=value,cmap='viridis',vmin=7,vmax=26,s=30)
        b=axes[row,1].scatter(xy[:,0],xy[:,1],c=delta,cmap='RdBu_r',vmin=-.5,vmax=.5,s=30)
        for ax in axes[row]:
            for name in ('PLTX','P475','PTEX','P510','P786'):
                index=np.flatnonzero(ids==name)
                if len(index):q=index[0];ax.annotate(name,(xy[q,0],xy[q,1]),xytext=(3,3),textcoords='offset points',fontsize=7)
            ax.set(xlabel='East coordinate (km)',ylabel='North coordinate (km)',aspect='equal');ax.grid(alpha=.2)
        axes[row,0].set_title(f'Calendar{day}: joint F6 station RMSE')
        axes[row,1].set_title(f'Calendar{day}: joint minus H_delta')
        fig.colorbar(a,ax=axes[row,0],label='RMSE (mm)',extend='both');fig.colorbar(b,ax=axes[row,1],label='RMSE difference (mm)',extend='both')
    fig.savefig(out/'station_tail_maps.png',dpi=170);plt.close(fig)
    lines=['# 有向时滞v2逐站与尾部误差','',
        '同两种子、raw final3000、已暴露2020抽样日期。站点RMSE按每布局分别对月份/小时计算后平均布局；宏RMSE再平均站点与种子，与原评价一致。micro/MAE/Bias/P90/P95先每种子汇总全部布局再平均种子。P90/P95为绝对误差分位数，单位mm。无标签站记NA，地图为站点值，不代表格网真值。','',
        '| 日历 | 模型 | 时效 | 宏RMSE | microRMSE | MAE | Bias | P90 | P95 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for day,report in reports.items():
        for arm,stats in report['overall'].items():
            for h,r in stats.items():lines.append(f'| {day} | {arm} | {h}h | '+' | '.join(f'{r[k]:.4f}' for k in ('macro_rmse','micro_rmse','mae','bias','p90','p95'))+' |')
    lines+=['','## 预先指定的高误差站','', '| 日历 | 站点 | H_delta F6 RMSE | joint F6 RMSE | 差值 | joint P95 |','|---|---|---:|---:|---:|---:|']
    for day,report in reports.items():
        for r in report['preselected_cases']:
            h=r['models']['H_delta']['6'];j=r['models']['joint']['6']
            lines.append(f"| {day} | {r['station']} | {h['rmse']:.4f} | {j['rmse']:.4f} | {r['F6_joint_minus_H_delta_mm']:.4f} | {j['p95']:.4f} |")
    lines+=['','![逐站误差与改变](station_tail_maps.png)','','完整137站、两日历和来源见[summary.json](summary.json)。逐站描述不替代主报告的共同月份整体区间，当前结果不作完整年度或独立确认结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({d:dict(overall=r['overall'],preselected=[dict(station=s['station'],F6_delta=s['F6_joint_minus_H_delta_mm']) for s in r['preselected_cases']]) for d,r in reports.items()}),flush=True)


if __name__=='__main__':main()
