"""Exact matched-label error decomposition; descriptive, no forecast fitting."""
import numpy as np
from campaign import RESULTS,read
from directed_v2_summary import locations
from tools.terrain_diffusion.data import sha,write_json


def main():
    root=RESULTS/'20261126_directed_lag_v2';audit=read(root/'station_tail_audit/summary.json')
    ids=[r['station'] for r in audit['calendars']['8']['stations']];rows=[];sources={}
    for day in (8,15):
        for arm in ('H_delta','factorized','joint'):
            per_seed=[]
            for seed in (42,17):
                _,_,folder=locations(RESULTS,arm,seed,'last',day)
                expected=audit['sources'][f'{day}/{arm}/{seed}']
                if sha(folder/'evaluation.npz')!=expected:raise ValueError('frozen prediction changed')
                with np.load(folder/'evaluation.npz') as z:e=z['prediction'].astype('float64')-z['truth'].astype('float64')
                a=e[...,0].reshape(-1,137);b=e[...,6].reshape(-1,137);valid=np.isfinite(a)&np.isfinite(b)
                a=np.where(valid,a,np.nan);b=np.where(valid,b,np.nan);d=b-a;count=valid.sum(0)
                def mean(x):return np.divide(np.nansum(x,0),count,out=np.full(137,np.nan),where=count>0)
                x=dict(initial_mse=mean(a*a),increment_mse=mean(d*d),cross_term=2*mean(a*d),future_mse=mean(b*b),
                    initial_bias=mean(a),increment_bias=mean(d),future_bias=mean(b),count=count)
                np.testing.assert_allclose(x['initial_mse']+x['increment_mse']+x['cross_term'],x['future_mse'],rtol=1e-12,atol=1e-10,equal_nan=True)
                per_seed.append(x);sources[f'{day}/{arm}/{seed}']=expected
            avg={n:np.mean([s[n] for s in per_seed],0) for n in per_seed[0]}
            for q,name in enumerate(ids):
                scalar=lambda n:float(avg[n][q]) if np.isfinite(avg[n][q]) else None
                rows.append(dict(calendar_day=day,arm=arm,station=name,values={n:scalar(n) for n in avg}))
    out=root/'station_tail_audit';write_json(out/'error_components.json',dict(rows=rows,sources=sources,code_sha256=sha(__file__),
        identity='E[e6^2]=E[e0^2]+E[(e6-e0)^2]+2E[e0(e6-e0)]',
        population='joint-valid h0/h6, pooled layouts/months/hours and equal seeds; differs from station/layout macro roots',
        interpretation='exact descriptive error identity, not causal attribution to atmospheric mechanisms',confirmation_C_D_read=False))
    lines=['# 当前误差与未来变化误差的精确分解','',
        '仅用当前及F6均有效的共同标签。MSE按布局、月份、小时合并，再等权平均种子；下表RMS不与原布局/站宏RMSE混排。交叉项为2E[e0(e6−e0)]，可正可负；它不是去均值协方差。恒等式逐站数值核验通过，不据此直接解释海风或天气机制。','',
        '| 日历 | 站点 | 当前误差RMS/mm | 未来增量误差RMS/mm | 交叉项/mm² | F6误差RMS/mm |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['arm']=='joint' and r['station'] in ('PLTX','P475','PTEX','P510','P786'):
            v=r['values'];lines.append(f"| {r['calendar_day']} | {r['station']} | {v['initial_mse']**.5:.3f} | {v['increment_mse']**.5:.3f} | {v['cross_term']:.3f} | {v['future_mse']**.5:.3f} |")
    lines+=['','全部站点及两对照见[error_components.json](error_components.json)。标签不足站记NA，原source及评价角色保持。']
    (out/'ERROR_COMPONENTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Exact error identity verified for137 queries x3 arms x2 calendars x2 seeds',flush=True)


if __name__=='__main__':main()
