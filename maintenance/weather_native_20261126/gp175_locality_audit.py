"""Fine near-pair distances and station maps; frozen predictions, no fitting."""
import json
import sys
from pathlib import Path
import numpy as np
from campaign import ROOT,RESULTS,read
from weather_gp_summary import interval

ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_evaluation import fair_crps,safe_mean


def main():
    root=RESULTS/'20261126_weather_projective_gp';out=root/'g175_locality';out.mkdir(parents=True,exist_ok=True)
    geo=read(RESULTS/'20261126_native_weather_cache_v2/calendar8_s42/geometry.json');st=np.asarray(geo['dev_static'])
    a,b=np.triu_indices(137,1);distance=np.linalg.norm((st[a,:2]-st[b,:2])*1000,axis=-1)
    keep=distance<50;a=a[keep];b=b[keep];distance=distance[keep]
    edges=np.bincount(np.r_[a,b],minlength=137);bins=[(0,5),(5,15),(15,50)]
    records={};station_values={};sources={}
    for seed in (42,17):
        roles={};truth_ref=None
        for mode in ('reference','c175'):
            folder=root/f's{seed}'/mode
            if mode!='reference':folder=folder/'eval'
            lineage=read(folder/'lineage.json')
            if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('locality inputs changed')
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];truth=data['truth'];prediction=data['prediction'];mu=data['gaussian_distribution_mean'];std=data['gaussian_distribution_std']
            if truth_ref is None:truth_ref=truth.copy()
            else:np.testing.assert_array_equal(truth,truth_ref)
            pair_sums=np.zeros((2,len(a)));pair_counts=np.zeros((2,len(a)),dtype='int64')
            monthly={f'{lo}-{hi}':{f'h{h}':dict(monthly_score_sum=np.zeros(12),monthly_valid_count=np.zeros(12,dtype='int64')) for h in (0,6)} for lo,hi in bins}
            for start in range(0,len(a),32):
                aa=a[start:start+32];bb=b[start:start+32];d=distance[start:start+32]
                for index,h in enumerate((0,6)):
                    score=fair_crps(samples[...,aa,h]-samples[...,bb,h],truth[...,aa,h]-truth[...,bb,h])
                    pair_sums[index,start:start+len(aa)]=np.nansum(score,axis=(0,1,2));pair_counts[index,start:start+len(aa)]=np.isfinite(score).sum((0,1,2))
                    for lo,hi in bins:
                        subset=(d>=lo)&(d<hi);row=monthly[f'{lo}-{hi}'][f'h{h}']
                        row['monthly_score_sum']+=np.nansum(score[...,subset],axis=(0,2,3));row['monthly_valid_count']+=np.isfinite(score[...,subset]).sum((0,2,3))
            for lo,hi in bins:
                for h in ('h0','h6'):
                    row=monthly[f'{lo}-{hi}'][h];row['fair_crps_mm']=float(row['monthly_score_sum'].sum()/row['monthly_valid_count'].sum())
                    row['monthly_score_sum']=row['monthly_score_sum'].tolist();row['monthly_valid_count']=row['monthly_valid_count'].tolist()
            node_sum=np.zeros(137);node_count=np.zeros(137)
            np.add.at(node_sum,a,pair_sums[1]);np.add.at(node_sum,b,pair_sums[1]);np.add.at(node_count,a,pair_counts[1]);np.add.at(node_count,b,pair_counts[1])
            node_score=np.divide(node_sum,node_count,out=np.full(137,np.nan),where=node_count>0)
            roles[mode]=node_score;records[seed,mode]=monthly
            if mode=='reference':
                error=prediction.astype('float64')-truth
                base=np.nanmean(np.sqrt(safe_mean(error**2,(1,2))),axis=0)
                z=(truth.astype('float64')-mu)/std;z_rms=np.sqrt(safe_mean(z*z,(0,1,2)))
                nearest=np.load(RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}/nearest.npy')
                station_values[seed]=dict(base=base,z_rms=z_rms,nearest=np.median(nearest.reshape(-1,137),axis=0),valid_hours=np.isfinite(truth[0]).sum((0,1)))
            sources[f's{seed}/{mode}']=dict(lineage_sha256=sha(folder/'lineage.json'),array_sha256=lineage['prediction_sha256'])
            del samples,truth,prediction,mu,std
        station_values[seed]['reference_pair']=roles['reference'];station_values[seed]['candidate_pair']=roles['c175']
        print(json.dumps(dict(seed=seed,stage='fine_distances_and_station_scores_complete')),flush=True)
    paired={}
    for lo,hi in bins:
        label=f'{lo}-{hi}';paired[label]={}
        for lead in ('h0','h6'):
            pairs=[(records[s,'c175'][label][lead],records[s,'reference'][label][lead]) for s in (42,17)]
            x=float(np.mean([a['fair_crps_mm'] for a,b in pairs]));ref=float(np.mean([b['fair_crps_mm'] for a,b in pairs]))
            paired[label][lead]=dict(pair_count=int(((distance>=lo)&(distance<hi)).sum()),candidate_mm=x,reference_mm=ref,gain_percent=100*(ref-x)/ref,delta_ci95_mm=interval(pairs))
    base=np.mean([station_values[s]['base'] for s in (42,17)],axis=0);z=np.mean([station_values[s]['z_rms'] for s in (42,17)],axis=0)
    ref=np.mean([station_values[s]['reference_pair'] for s in (42,17)],axis=0);new=np.mean([station_values[s]['candidate_pair'] for s in (42,17)],axis=0)
    gain=100*(ref-new)/ref;nearest=np.mean([station_values[s]['nearest'] for s in (42,17)],axis=0)
    metadata=np.load(RESULTS/'20261125_temporal_continuous_data/geometry.npz');names=metadata['dev_ids']
    rows=[]
    def value(x):return float(x) if np.isfinite(x) else None
    for q in range(137):rows.append(dict(station=str(names[q]),query_index=q,projected_east_km=float(st[q,0]*1000),projected_north_km=float(st[q,1]*1000),height_m=float(st[q,2]*3000),coast_km=float(st[q,8]*500),median_nearest_condition_km=float(nearest[q]),
        base_current_R_macro_rmse_mm=value(base[q,0]),base_F6_macro_rmse_mm=value(base[q,6]),gaussian_F6_standardized_error_rms=value(z[q,6]),
        valid_unique_hours_R=int(station_values[42]['valid_hours'][q,0]),valid_unique_hours_F6=int(station_values[42]['valid_hours'][q,6]),
        near_pair_degree=int(edges[q]),near_F6_pair_gain_percent=value(gain[q])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained');xy=st[:,:2]*1000
    for ax,c,cmap,title in zip(axes,(base[:,6],z[:,6],gain),('viridis','magma','RdBu'),('Frozen base F6 RMSE (mm)','Fixed Gaussian F6 standardized error RMS','Incident <50km F6 pair CRPS gain (%)')):
        valid=np.isfinite(c);ax.scatter(xy[~valid,0],xy[~valid,1],s=25,facecolors='none',edgecolors='#aaa')
        artist=ax.scatter(xy[valid,0],xy[valid,1],c=c[valid],s=26,cmap=cmap)
        fig.colorbar(artist,ax=ax,shrink=.72);ax.set(xlabel='Projected east (km)',ylabel='Projected north (km)',title=title,aspect='equal');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two-seed development maps: point/scale maps unchanged by G175; covariance changes right panel')
    fig.savefig(out/'g175_station_maps.png',dpi=170);plt.close(fig)
    write_json(out/'locality.json',dict(fine_distance_pairs=paired,stations=rows,sources=sources,code_sha256=sha(__file__),
        no_pairs_below1km=int((distance<1).sum())==0,connected_station_count=int((edges>0).sum()),query_count=137,
        interpretation='K32 IID fair pair scores; fine-distance and station diagnostics exploratory; no label fitting; empty-neighborhood stations omitted from pair gain only',strict_oof=False,confirmation_C_D_read=False))
    lines=['# G175局地概率收益与待改进站点','','对冻结主日历进行只读诊断。0–50km的720站对中，<1km为0对，0–5/5–15/15–50km分别35/174/511对；无近邻的站点在地图右图为空心灰点。','',
        '| 距离/km | 站对数 | F6 CRPS改善 | F6差值95%区间/mm |','|---|---:|---:|---|']
    for d,row in paired.items():lines.append(f"| {d} | {row['h6']['pair_count']} | {row['h6']['gain_percent']:.2f}% | {row['h6']['delta_ci95_mm']} |")
    lines+=['','![基础场误差、尺度校准和邻站联合收益](g175_station_maps.png)','','左图为基础场F6误差，中图为Gaussian预测均值/标准差下的标准化误差RMS；它们在G175与参考间不变。右图为每站关联<50km站对的fair CRPS改善，表达差值分布的收益，不代表点值RMSE改善。','',
        '## 需要优先改进的基础场站点','','| 站点 | 当前R/mm | F6/mm | F6有效小时 | 近条件站中位距离/km | 高程/m | 海岸距离/km | F6标准化误差RMS |','|---|---:|---:|---:|---:|---:|---:|---:|']
    def formatted(x):return f'{x:.3f}' if x is not None else 'NA'
    for row in sorted([r for r in rows if r['base_F6_macro_rmse_mm'] is not None],key=lambda r:r['base_F6_macro_rmse_mm'],reverse=True)[:10]:
        lines.append(f"| {row['station']} | {formatted(row['base_current_R_macro_rmse_mm'])} | {formatted(row['base_F6_macro_rmse_mm'])} | {row['valid_unique_hours_F6']} | {row['median_nearest_condition_km']:.1f} | {row['height_m']:.0f} | {row['coast_km']:.1f} | {formatted(row['gaussian_F6_standardized_error_rms'])} |")
    lines+=['',f"137注册查询中，当前/F6实际有标签的站点分别为{sum(r['valid_unique_hours_R']>0 for r in rows)}/{sum(r['valid_unique_hours_F6']>0 for r in rows)}。缺标签站保留预测、误差记NA；三布局共用的观测小时只计一次。空心灰点也可能表示缺少有效配对标签。"]
    lines+=['','按月份共同重采样、两个种子共享月份，区间未经多重比较校正。逐站地图为探索性定位；全部137站及当前差值结果见[locality.json](locality.json)。']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'));print(json.dumps(paired))


if __name__=='__main__':main()
