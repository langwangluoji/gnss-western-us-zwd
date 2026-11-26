"""Read-only source, rank and station diagnostics supplementary to locked scores."""
import numpy as np


def randomized_rank80(samples,truth):
    x=np.sort(samples,axis=0);valid=np.isfinite(truth)
    if len(x)==16:
        narrow=(truth>=x[1])&(truth<=x[14]);wide=(truth>=x[0])&(truth<=x[15]);result=.7*narrow+.3*wide
    elif len(x)==32:
        wide=(truth>=x[2])&(truth<=x[29]);narrow=(truth>=x[3])&(truth<=x[28]);result=.7*wide+.3*narrow
    else:raise ValueError('registered rank diagnostics require16 or32 members')
    return np.where(valid,result,np.nan)


def main():
    import sys
    import torch
    from campaign import ROOT,ENGINE,LEGACY,RESULTS,OUTPUT,LOCK,read,paths,validate_lock
    sys.path.append(str(LEGACY))
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_paths import file_path
    from tools.terrain_diffusion.native_weather_evaluation import calendar_shape
    from tools.terrain_diffusion.native_model_seal import evaluation_marker
    from tools.terrain_diffusion.native_weather_cache import static
    from event_audit import validate_calendar_origins
    from weather_controls_summary import monthly_record
    from mean_hs_summary import safe_comparison
    torch.set_num_threads(2);validate_lock();rows=[];source_receipts={}
    names=('G265','affine','local','paired_G265','paired_affine','paired_local')
    for seed in (42,17):
        validate_lock(seed)
        for day in (8,15):
            cache,emb,_=paths(seed,day);cm=read(cache/'manifest.json');cp=read(cache/'protocol.json')
            if sha(cache/'protocol.json')!=cm['protocol_sha256']:raise ValueError('calendar protocol changed')
            if sha(cache/'geometry.json')!=cm['geometry_sha256']:raise ValueError('calendar geometry changed')
            for name in ('nearest.npy','origins.npy','query_ids.npy'):
                if sha(cache/name)!=cm['files'][name]:raise ValueError('calendar geometry/distance/identity changed')
            validate_calendar_origins(np.load(cache/'origins.npy'),day)
            ids=np.load(cache/'query_ids.npy');np.testing.assert_array_equal(ids,np.broadcast_to(np.arange(137),ids.shape))
            original=file_path(cp['config']['data']);contract=read(original/'contract.json')
            if sha(original/'contract.json')!=cp['data_sha256'] or sha(original/'geometry.npz')!=contract['files']['geometry.npz']:
                raise ValueError('original station geometry/identity source changed')
            with np.load(original/'geometry.npz') as g:
                station=g['dev_ids'].copy();expected_geometry=static(dict(g),'dev_')
            np.testing.assert_array_equal(np.asarray(read(cache/'geometry.json')['dev_static'],dtype='float32'),expected_geometry)
            oldroot=RESULTS/'20261126_directed_lag_v2'
            old=oldroot/(f'joint_s{seed}/eval_last' if day==8 else f'supplement/joint_last_s{seed}/evaluation')
            registered=read(oldroot/f'paired_calendar{day}/summary.json')['sources'][f'joint/last/s{seed}']['prediction_sha256']
            if sha(old/'evaluation.npz')!=registered:raise ValueError('original frozen joint evaluation changed')
            with np.load(old/'evaluation.npz') as z:base=z['prediction'];truth=z['truth']
            np.testing.assert_array_equal(base,calendar_shape(np.load(emb/'mean.npy'),12))
            source_receipts[f'{day}/{seed}']=dict(original_joint_evaluation_sha256=registered,
                cache_geometry_sha256=sha(cache/'geometry.json'),nearest_sha256=sha(cache/'nearest.npy'))
            for name in names:
                folder=OUTPUT/f'calendar{day}/{name}_s{seed}';evaluation_marker(folder)
                with np.load(folder/'evaluation.npz') as z:
                    np.testing.assert_array_equal(z['prediction'],base);np.testing.assert_array_equal(z['truth'],truth)
                    x=z['samples'];crps=z['crps'];rank=randomized_rank80(x,truth)
                    records={f'randomized80_h{h}':monthly_record(rank[...,h],1) for h in (0,6)}
                    station_score=np.nanmean(crps,axis=(0,1,2))
                    base_error=base.astype('float64')-truth
                    count=np.isfinite(base_error).sum((1,2));mse=np.divide(np.nansum(base_error**2,(1,2)),count,out=np.full(count.shape,np.nan),where=count>0)
                    point=np.nanmean(np.sqrt(mse),axis=0)
                    stations=[]
                    for q,label in enumerate(station):
                        f=lambda a:[float(v) if np.isfinite(v) else None for v in a]
                        stations.append(dict(station=str(label),query_index=q,point_R_F6_mm=f(point[q,[0,6]]),
                            crps_R_F6_mm=f(station_score[q,[0,6]])))
                    rows.append(dict(seed=seed,day=day,name=name,records=records,stations=stations,
                        source_evaluation_sha256=sha(folder/'evaluation.npz')))
                print('rank/station audited',seed,day,name,flush=True)
    coverage={};lookup={(r['day'],r['seed'],r['name']):r for r in rows}
    for day in (8,15):
        for a,b in [('local','affine'),('paired_local','paired_affine')]:
            coverage[f'day{day}/{a}_vs_{b}']={k:safe_comparison([(lookup[day,s,a]['records'][k],lookup[day,s,b]['records'][k]) for s in (42,17)]) for k in ('randomized80_h0','randomized80_h6')}
    out=OUTPUT/'paired';write_json(out/'diagnostics.json',dict(rows=rows,coverage_comparisons=coverage,
        original_point_arrays_exact=True,calendar_sources=source_receipts,lock_sha256=sha(LOCK),code_sha256=sha(__file__),
        rank_reference_80='K16 .7*[rank2,rank15]+.3*[rank1,rank16]; K32 .7*[rank3,rank30]+.3*[rank4,rank29]; continuous exchangeable reference',
        strict_oof=False,confirmation_C_D_read=False))
    lines=['# 融合模型补充核验：原点场、覆盖和重点站','',
        '全部24组点场/真值直接与原joint v2评价数组逐值相同；几何、原站身份、最近有效GNSS站距与日历哈希通过。以下为双种子等权开发描述，覆盖率参考80%。','',
        '| 日历 | 模型 | 当前覆盖率 | F6覆盖率 |','|---|---|---:|---:|']
    for day in (8,15):
        for name in names:
            get=lambda k:np.mean([lookup[day,s,name]['records'][k]['value'] for s in (42,17)])
            lines.append(f"| {day} | {name} | {get('randomized80_h0'):.6f} | {get('randomized80_h6'):.6f} |")
    lines+=['','## 预先关注的困难站','', '| 日历 | 站点 | 原joint F6 RMSE/mm | G265 F6 CRPS/mm | Affine F6 CRPS/mm | Local F6 CRPS/mm | 中心保持Local F6 CRPS/mm |','|---|---|---:|---:|---:|---:|---:|']
    for day in (8,15):
        for label in ('PLTX','P475','PTEX','P510','P786'):
            def get(name,key):return np.mean([next(v for v in lookup[day,s,name]['stations'] if v['station']==label)[key][1] for s in (42,17)])
            lines.append(f"| {day} | {label} | {get('local','point_R_F6_mm'):.4f} | "+' | '.join(f"{get(name,'crps_R_F6_mm'):.4f}" for name in ('G265','affine','local','paired_local'))+' |')
    lines+=['','站点CRPS与站点RMSE评价对象不同，不作同列排名。K16均值保持与K32 raw分别使用匹配对照；无标签站保留null，不补造。完整137站及来源见[diagnostics.json](diagnostics.json)。']
    (out/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
