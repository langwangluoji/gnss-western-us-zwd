"""Independent float64 means, paired calendar intervals and geographic diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def curve_values(curve,key):
    return [row.get(key,row.get('loss',np.nan) if key=='query_loss' else np.nan) for row in curve]


def average(x,axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(x,axis=axis)


def macro(error):
    return average(np.sqrt(average(error.astype('float64')**2,(1,2))),(0,1))


def month_interval(error,reference,draws=2000):
    # Preserve layout/station units and dependence within each month's 72h.
    # Calendar-month resampling is exploratory: 12 sampled months, no blind test.
    a=np.nansum(error.astype('float64')**2,2);b=np.nansum(reference.astype('float64')**2,2)
    ac=np.isfinite(error).sum(2);bc=np.isfinite(reference).sum(2)
    rng=np.random.default_rng(261126);values=[]
    for _ in range(draws):
        index=rng.integers(a.shape[1],size=a.shape[1])
        def score(square,count):
            numerator=square[:,index].sum(1);denominator=count[:,index].sum(1)
            mse=np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan),where=denominator>0)
            return average(np.sqrt(mse))
        values.append(score(a,ac)-score(b,bc))
    return np.quantile(values,[.025,.975]).tolist()


def pair_errors(error,xy):
    i,j=np.triu_indices(len(xy),1);distance=np.linalg.norm(xy[i]-xy[j],axis=-1)
    totals=[];counts=[]
    for h in (0,6):
        values=error[...,h].reshape(-1,len(xy));valid=np.isfinite(values).astype('float64')
        values=np.nan_to_num(values);square=(values**2).T@valid
        total=square+square.T-2*(values.T@values)
        totals.append(np.maximum(total[i,j],0));counts.append((valid.T@valid)[i,j])
    result={}
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        keep=(distance>=lo)&(distance<hi)
        result[f'{lo}-{hi:g}']=dict(pair_count=int(keep.sum()),
            current_rmse_mm=float(np.sqrt(totals[0][keep].sum()/counts[0][keep].sum())) if counts[0][keep].sum() else None,
            future6_rmse_mm=float(np.sqrt(totals[1][keep].sum()/counts[1][keep].sum())) if counts[1][keep].sum() else None)
    return result


def stratified(error,nearest,static):
    output={}
    variables=[('distance_km',nearest,[0,50,100,150,300,np.inf]),
        ('height_m',static[:,2]*3000,[-np.inf,500,1000,2000,np.inf]),
        ('coast_km',static[:,8]*500,[0,50,150,300,np.inf]),
        ('roughness_m',static[:,5]*500,[-np.inf,100,250,500,np.inf])]
    for name,value,bins in variables:
        rows=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            keep=(value>=lo)&(value<hi)
            filtered=np.where(keep[...,None],error,np.nan)
            rows.append(dict(lower=lo,upper=hi,finite_current_cells=int(np.isfinite(filtered[...,0]).sum()),
                horizon_macro_rmse_mm=macro(filtered).tolist()))
        output[name]=rows
    output['monthly_macro_rmse_mm']=[macro(error[:,m:m+1]).tolist() for m in range(error.shape[1])]
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    cache=ENGINE/f'results/20261126_native_weather_cache_v2/calendar8_s{args.seed}'
    geometry=read(cache/'geometry.json');static=np.asarray(geometry['dev_static'])
    original=Path(read(cache/'protocol.json')['config']['data'])
    station_ids=np.load(original/'geometry.npz')['dev_ids'].astype(str)
    nearest=np.load(cache/'nearest.npy').reshape(3,12,6,12,137).reshape(3,12,72,137)
    rows=[];errors={};references={};curves={}
    source=ENGINE/'results/20261126_native_weather_means'
    for folder in sorted(source.glob(f'*_s{args.seed}')):
        status=folder/'status.json'
        if not status.exists() or read(status)['status']!='COMPLETE_MEAN':continue
        curves[folder.name]=read(folder/'curve.json')
        for weight in ('last','ema'):
            evaluation=folder/f'eval_{weight}'
            if not (evaluation/'metrics.json').exists():continue
            with np.load(evaluation/'evaluation.npz') as data:
                error=data['prediction'].astype('float64')-data['truth']
                reference=data['reference'].astype('float64')-data['truth']
            actual=macro(error);registered=read(evaluation/'metrics.json')['deterministic_mean']['horizon_macro_rmse_mm']
            np.testing.assert_allclose(actual,registered,rtol=0,atol=1e-10)
            name=folder.name.removesuffix(f'_s{args.seed}')+'_'+weight
            current=error[...,0];future=error[...,6]
            row=dict(name=name,horizon_macro_rmse_mm=actual.tolist(),
                current_delta_ci95_mm=month_interval(error[...,:1],reference[...,:1]),
                future6_delta_ci95_mm=month_interval(error[...,6:],reference[...,6:]),
                current_increment_macro_rmse_mm=float(macro(np.diff(error,axis=2))[0]),
                current_p95_absolute_mm=float(np.nanquantile(np.abs(current),.95)),
                future6_p95_absolute_mm=float(np.nanquantile(np.abs(future),.95)),
                strata=stratified(error,nearest,static),station_pairs=pair_errors(error,static[:,:2]*1000),
                source_evaluation_sha256=sha(evaluation/'evaluation.npz'),
                model_status=read(status),mean_protocol_sha256=sha(folder/'protocol.json'))
            rows.append(row);errors[name]=error;references[name]=reference
    if not rows:raise ValueError('no completed mean evaluations')
    # One geographic atlas for the best completed raw future predictor.
    candidates=[r for r in rows if r['name'].endswith('_last')]
    best=min(candidates,key=lambda r:r['horizon_macro_rmse_mm'][6]);name=best['name'];e=errors[name];ref=references[name]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    raw=[r for r in rows if r['name'].endswith('_last')]
    axes[0].plot(range(7),macro(ref),'k--',label='I1 / persistence')
    for row in raw:axes[0].plot(range(7),row['horizon_macro_rmse_mm'],label=row['name'].removesuffix('_last'))
    axes[0].set(xlabel='Forecast lead (h); 0 = reconstruction',ylabel='Station macro RMSE (mm)');axes[0].legend(fontsize=6)
    for n,curve in curves.items():
        axes[1].plot([c['step'] for c in curve],curve_values(curve,'query_loss'),label=n.removesuffix(f'_s{args.seed}'),alpha=.8)
    axes[1].set(xlabel='Chronological block',ylabel='Query training loss',title='Different dates per step; not validation curves')
    for n,curve in curves.items():axes[2].plot([c['step'] for c in curve],curve_values(curve,'future6_correction_rms_mm'),label=n)
    axes[2].set(xlabel='Chronological block',ylabel='Future6 correction RMS (mm)',title='New variants; original curves did not log this')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Seed {args.seed}: completed 3000-block mean models (development calendar)')
    fig.savefig(out/'mean_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    level=np.sqrt(average(e[...,6]**2,(0,1,2)));base=np.sqrt(average(ref[...,6]**2,(0,1,2)))
    delta=level-base;bound=max(float(np.nanmax(np.abs(delta))),.01)
    artist=axes[0].scatter(static[:,0]*1000,static[:,1]*1000,c=level,s=24,cmap='viridis');fig.colorbar(artist,ax=axes[0],label='F6 station RMSE (mm)')
    artist=axes[1].scatter(static[:,0]*1000,static[:,1]*1000,c=delta,s=24,cmap='RdBu_r',vmin=-bound,vmax=bound);fig.colorbar(artist,ax=axes[1],label='F6 change vs persistence (mm)')
    for ax,key in zip(axes[2:],('distance_km','coast_km')):
        group=best['strata'][key];labels=[f"{r['lower']:g}-{r['upper']:g}" for r in group]
        ax.bar(labels,[r['horizon_macro_rmse_mm'][6] for r in group],color='#5986ac')
        ax.set(xlabel=key.replace('_',' '),ylabel='F6 macro RMSE (mm)');ax.tick_params(axis='x',rotation=40)
    for ax in axes[:2]:ax.set(xlabel='Projected east (km)',ylabel='Projected north (km)',aspect='equal')
    fig.suptitle(f'Exploratory focus: {name}; same 2020 development calendar')
    fig.savefig(out/'regional_focus.png',dpi=170);plt.close(fig)
    ranking=np.argsort(-np.where(np.isfinite(level),level,-np.inf))[:10]
    station_data=dict(station_index=list(range(len(static))),station_ids=station_ids.tolist(),future6_rmse_mm=level.tolist(),future6_delta_mm=delta.tolist(),
        largest_error=[dict(station=station_ids[i],future6_rmse_mm=float(level[i]),future6_delta_mm=float(delta[i]),
            height_m=float(static[i,2]*3000),coast_km=float(static[i,8]*500)) for i in ranking])
    payload=dict(seed=args.seed,rows=rows,station_focus=station_data,best_completed_raw_future=name,
        interval='paired month bootstrap, 2000 draws, exploratory and conditional on this calendar; no selection correction',
        six_error_families=['time-held-out development','spatially held-out development','condition distance','height and roughness','station-pair differences','hourly increments'],
        confirmation_C_D_read=False,strict_oof=False,code_sha256=sha(Path(__file__)))
    (out/'analysis.json').write_bytes((json.dumps(plain(payload),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    table=['# 连续3000块均值模型比较', '',f'seed{args.seed}，2020已暴露主日历。所有区间为月份配对开发回看，候选选择未作多重比较校正。', '',
        '| 模型 | R/mm | F6/mm | R增量/mm | F6相对持久性95%区间/mm |','|---|---:|---:|---:|---|']
    for r in rows:
        lo,hi=r['future6_delta_ci95_mm']
        table.append(f"| {r['name']} | {r['horizon_macro_rmse_mm'][0]:.6f} | {r['horizon_macro_rmse_mm'][6]:.6f} | {r['current_increment_macro_rmse_mm']:.6f} | [{lo:.4f}, {hi:.4f}] |")
    table+=['','## 六小时误差较高的站点','',f'以当前已完成raw候选{name}为例；负变化表示优于持久性。','',
        '| 站点 | F6/mm | 变化/mm | 高程/m | 海岸距离/km |','|---|---:|---:|---:|---:|']
    for s in station_data['largest_error']:table.append(f"| {s['station']} | {s['future6_rmse_mm']:.3f} | {s['future6_delta_mm']:.3f} | {s['height_m']:.0f} | {s['coast_km']:.1f} |")
    table+=['','![均值与学习曲线](mean_comparison.png)','','![区域重点](regional_focus.png)','','全部六类分层、站对及来源哈希见[analysis.json](analysis.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(table)+'\n').encode('utf-8'))
    print(json.dumps(dict(completed_evaluations=len(rows),best_completed_raw_future=name,R=best['horizon_macro_rmse_mm'][0],F6=best['horizon_macro_rmse_mm'][6])))


if __name__=='__main__':main()
