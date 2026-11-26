"""Training-pool transfer probes; source backgrounds already fitted2016-2019."""
import numpy as np
from pathlib import Path
import json


def fit_probe(x,y,weights,years):
    selected=np.asarray(years)<2019
    if not selected.any():raise ValueError('probe requires2016-2018 training rows')
    a=np.asarray(x[selected],dtype='float64').reshape(int(selected.sum())*x.shape[1],x.shape[-1])
    target=np.asarray(y[selected],dtype='float64').reshape(-1,y.shape[-1])
    w=np.asarray(weights[selected],dtype='float64').reshape(-1)
    if not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('invalid training features or weights')
    center=np.sum(a*w[:,None],axis=0)/w.sum()
    scale=np.maximum(np.sqrt(np.sum((a-center)**2*w[:,None],axis=0)/w.sum()),1e-3)
    design=np.c_[(a-center)/scale,np.ones(len(a))];coefficient=[];counts=[]
    for h in range(target.shape[-1]):
        valid=np.isfinite(target[:,h])&(w>0)
        if not valid.any():raise ValueError('no training labels for probe lead')
        q=design[valid];wt=w[valid];yy=target[valid,h];den=wt.sum()
        matrix=q.T@(q*wt[:,None])/den;rhs=q.T@(yy*wt)/den
        penalty=.01*np.diag(np.diag(matrix));penalty[-1,-1]=0
        coefficient.append(np.linalg.solve(matrix+penalty+np.eye(len(matrix))*1e-10,rhs));counts.append(int(valid.sum()))
    return dict(center=center,scale=scale,coefficient=np.stack(coefficient,axis=-1),training_label_count=counts)


def probe_prediction(x,fit):
    return np.concatenate([(x-fit['center'])/fit['scale'],np.ones((*x.shape[:-1],1))],axis=-1)@fit['coefficient']


def transfer_features(errors,xy,cxy,height,cheight,origins=12):
    import torch
    from tools.terrain_diffusion.innovation_refinement import innovation_fields
    error=torch.as_tensor(errors,dtype=torch.float32);valid=torch.isfinite(error)
    fields=innovation_fields(error,valid,torch.as_tensor(xy),torch.as_tensor(cxy),torch.as_tensor(height),torch.as_tensor(cheight)).numpy()
    at=np.arange(3,3+origins)
    return np.concatenate([fields[at-lag] for lag in (0,1,3)],axis=-1)/10


def diagnostic_records(error,nearest,geometry):
    from weather_controls_summary import monthly_record
    rows={f'mse_h{h}':monthly_record(error[...,h]**2,0) for h in (0,6)}
    rows['within_origin_increment_mse']=monthly_record(np.diff(error,axis=-1)**2,0)
    rows['cross_origin_current_increment_mse']=monthly_record(np.diff(error[...,0],axis=1)**2,0)
    masks={}
    for lo,hi in [(0,50),(50,100),(100,150),(150,300),(300,np.inf)]:masks[f'distance_{lo}-{hi:g}']=(nearest>=lo)&(nearest<hi)
    for name,values,edges in [('height',geometry['height'],[-np.inf,500,1500,np.inf]),('coast',geometry['coast'],[0,50,150,300,np.inf])]:
        for lo,hi in zip(edges[:-1],edges[1:]):masks[f'{name}_{lo:g}-{hi:g}']=np.broadcast_to(((values>=lo)&(values<hi))[None,None],error.shape[:-1])
    for name,months in [('DJF',[11,0,1]),('MAM',[2,3,4]),('JJA',[5,6,7]),('SON',[8,9,10])]:
        mask=np.zeros(error.shape[:-1],bool);mask[months]=True;masks['season_'+name]=mask
    for name,mask in masks.items():
        for h in (0,6):
            values=np.where(mask,error[...,h]**2,np.nan)
            if np.isfinite(values).any():rows[f'{name}_h{h}']=monthly_record(values,0)
    return rows


def main():
    from campaign import RESULTS,read
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_innovation_audit import codes
    from tools.terrain_diffusion.native_weather_evaluation import point_metrics
    from innovation_panel_campaign import schedule
    from paired_mean_summary import paired_interval,plain
    from mean_hs_summary import safe_comparison
    root=RESULTS/'20261126_leaveout_innovation_panel';all_errors={};all_metrics={};all_records={};all_fits={};sources={};rms_rows=[]
    names=['none','intercept','self','LOSO','LOGO','self_LOSO','self_LOGO']
    for seed in (42,17):
        folder=root/f's{seed}';protocol=read(folder/'protocol.json');status=read(folder/'status.json')
        if status['status']!='COMPLETE_INFORMATION_PANEL' or status['blocks']!=48 or status['protocol_sha256']!=sha(folder/'protocol.json'):
            raise ValueError('information panel must complete all48 blocks')
        if protocol['code_sha256']!=codes():raise ValueError('audit implementation changed')
        cache=RESULTS/f'20261126_native_weather_cache_v2/train_s{seed}'
        embedding=RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_last_train_s{seed}'
        if protocol['cache_manifest_sha256']!=sha(cache/'manifest.json') or protocol['mean_manifest_sha256']!=sha(embedding/'manifest.json'):
            raise ValueError('panel source changed')
        origins=np.load(cache/'origins.npy',mmap_mode='r')
        if protocol['schedule']!=schedule(origins):raise ValueError('diagnostic population changed')
        cache_arrays={n:np.load(cache/f'{n}.npy',mmap_mode='r') for n in ['truth','weights','query_ids','context_ids','nearest']}
        frozen_mean=np.load(embedding/'mean.npy',mmap_mode='r');cm=read(cache/'manifest.json');em=read(embedding/'manifest.json')
        mean_source=RESULTS/f'20261126_native_weather_means/transformer72_physics_aux_s{seed}'
        cp=read(cache/'protocol.json')
        if em['model_kind']!='transformer' or em['mean_variant']!='physics_aux' or em['weights']!='last':raise ValueError('reference mean recipe differs')
        if em['mean_protocol_sha256']!=sha(mean_source/'protocol.json') or em['mean_weights_sha256']!=sha(mean_source/'last.pt'):
            raise ValueError('reference mean source differs')
        for name in cache_arrays:
            if sha(cache/f'{name}.npy')!=cm['files'][f'{name}.npy']:raise ValueError('source cache array changed')
        if sha(embedding/'mean.npy')!=em['files']['mean.npy']:raise ValueError('source mean changed')
        data=Path(read(cache/'protocol.json')['config']['data']);contract=read(data/'contract.json')
        if sha(data/'geometry.npz')!=contract['files']['geometry.npz']:raise ValueError('geometry source changed')
        geometry=dict(np.load(data/'geometry.npz'))
        fields={n:[] for n in ['self','LOSO','LOGO']};targets=[];weights=[];years=[];queries=[];distances=[]
        for row in protocol['schedule']:
            i=row['index'];block=RESULTS/f'20261126_leaveout_innovation_audit/b{i:04d}_o12_s{seed}';summary=read(block/'summary.json')
            if summary['code_sha256']!=codes() or summary['arrays_sha256']!=sha(block/'audit_arrays.npz') or summary['cache_manifest_sha256']!=protocol['cache_manifest_sha256']:
                raise ValueError('audit block source changed')
            if summary['origins']!=12 or summary['cache_index']!=i or summary['mean_manifest_sha256']!=protocol['mean_manifest_sha256'] or not summary['complete']:
                raise ValueError('audit block identity changed')
            if summary['p14_source_weights_sha256']!=cp['source_sha256']:raise ValueError('background weights differ from cache source')
            if summary['withheld_input_perturbation_max_difference_mm']!=0 or summary['self_cache_max_difference_mm']>.05:
                raise ValueError('audit input isolation or self consistency failed')
            with np.load(block/'audit_arrays.npz') as a:
                np.testing.assert_array_equal(a['query_truth'],cache_arrays['truth'][i]);np.testing.assert_array_equal(a['query_mean'],frozen_mean[i])
                np.testing.assert_array_equal(a['query'],cache_arrays['query_ids'][i]);np.testing.assert_array_equal(a['context'],cache_arrays['context_ids'][i])
                np.testing.assert_array_equal(a['hours'],origins[i])
                for prefix,ids in [('query',a['query']),('context',a['context'])]:
                    np.testing.assert_array_equal(a[prefix+'_xy'],geometry['xy'][ids])
                    np.testing.assert_array_equal(a[prefix+'_height'],geometry['height'][ids])
                for arm in fields:
                    features=transfer_features(a[arm+'_error'],a['query_xy'],a['context_xy'],a['query_height'],a['context_height'])
                    if not np.isfinite(features).all():raise ValueError('nonfinite innovation features')
                    fields[arm].append(features.reshape(-1,9))
                targets.append((a['query_truth'].astype('float64')-a['query_mean'].astype('float64')).reshape(-1,7))
                queries.append(a['query'].copy())
            weights.append(np.asarray(cache_arrays['weights'][i]).reshape(-1));distances.append(np.asarray(cache_arrays['nearest'][i]))
            years.append(row['year']);rms_rows.append(dict(seed=seed,**row,innovation_rms_mm=summary['innovation_rms_mm'],seconds=summary['seconds']))
            sources[f's{seed}/b{i}']=dict(summary_sha256=sha(block/'summary.json'),arrays_sha256=summary['arrays_sha256'],
                p14_weights_sha256=summary['p14_source_weights_sha256'],mean_array_sha256=summary['mean_array_sha256'])
        fields={n:np.stack(v) for n,v in fields.items()};target=np.stack(targets);weight=np.stack(weights);years=np.asarray(years)
        fields['self_LOSO']=np.concatenate([fields['self'],fields['LOSO']],-1)
        fields['self_LOGO']=np.concatenate([fields['self'],fields['LOGO']],-1)
        fields['intercept']=np.empty((*target.shape[:-1],0));fields['none']=fields['intercept']
        test=np.flatnonzero(years==2019);nearest=np.full((12,12,410),np.nan)
        for mi,b in enumerate(test):nearest[mi][:,queries[b]]=distances[b]
        for name in names:
            if name=='none':correction=np.zeros_like(target)
            else:
                fit=fit_probe(fields[name],target,weight,years);all_fits[f's{seed}/{name}']=fit
                correction=probe_prediction(fields[name],fit)
            error=np.full((12,12,410,7),np.nan)
            for mi,b in enumerate(test):error[mi][:,queries[b],:]=(correction[b]-target[b]).reshape(12,128,7)
            zero=np.where(np.isfinite(error),0,np.nan)
            metrics=point_metrics(error[None],zero[None])
            metrics['correction_rms_mm_by_horizon']=np.sqrt(np.nanmean(correction[test]**2,axis=(0,1))).tolist()
            all_errors[seed,name]=error;all_metrics[seed,name]=metrics
            all_records[seed,name]=diagnostic_records(error,nearest,geometry)
    rows=[];comparisons={}
    for name in names:
        metrics=[all_metrics[s,name] for s in (42,17)]
        rows.append(dict(name=name,horizon_macro_rmse_mm=np.mean([m['horizon_macro_rmse_mm'] for m in metrics],axis=0).tolist(),
            horizon_micro_rmse_mm=np.mean([m['horizon_micro_rmse_mm'] for m in metrics],axis=0).tolist(),
            correction_rms_mm_by_horizon=np.mean([m['correction_rms_mm_by_horizon'] for m in metrics],axis=0).tolist(),per_seed=metrics))
        if name=='none':continue
        for ref in ['none','self']:
            if name==ref:continue
            a=np.stack([all_errors[s,name] for s in (42,17)])[:,None]
            b=np.stack([all_errors[s,ref] for s in (42,17)])[:,None]
            common=set.intersection(*[set(all_records[s,n]) for s in (42,17) for n in [name,ref]])
            comparisons[name+'_vs_'+ref]=dict(macro_R_F6_delta_ci95_mm=paired_interval(a[...,[0,6]],b[...,[0,6]]),
                mse_records={k:safe_comparison([(all_records[s,name][k],all_records[s,ref][k]) for s in (42,17)]) for k in sorted(common)})
    out=root/'paired';out.mkdir(parents=True,exist_ok=True)
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
        if isinstance(v,list):return [convert(x) for x in v]
        return v
    write_json(out/'summary.json',plain(convert(dict(rows=rows,comparisons=comparisons,fits=all_fits,sources=sources,innovation_rms=rms_rows,
        code_sha256=sha(__file__),campaign_sha256=sha(Path(__file__).with_name('innovation_panel_campaign.py')),
        dependencies={n:sha(Path(__file__).with_name(n)) for n in ['paired_mean_summary.py','weather_controls_summary.py','mean_hs_summary.py']},
        role='probe fit2016-2018;probe transfer2019; all source neural weights fitted2016-2019; no independent foundation holdout',
        strict_oof=False,confirmation_C_D_read=False))))
    render(out,rows,comparisons,rms_rows)


def render(out,rows,comparisons,rms_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for row in rows:axes[0].plot(range(7),row['horizon_macro_rmse_mm'],'o-',label=row['name'])
    for arm in ['self','LOSO','LOGO']:
        values=[r['innovation_rms_mm'][arm] for r in rms_rows]
        axes[1].plot(range(len(values)),values,label=arm)
    axes[0].set(xlabel='Lead (h)',ylabel='Station macro RMSE (mm)',title='2019 probe transfer; source already trained2019');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Fixed month block (seed42 then17)',ylabel='Condition innovation RMS (mm)',title='Magnitude is not predictive skill');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'innovation_panel.png',dpi=170);plt.close(fig)
    lines=['# 留站创新面板：尺度与可预测信号分开检验','',
        '两种子各48个固定月块，每块连续12h；2016–2018拟合线性探针，2019检验探针迁移。来源p14/TF已拟合2016–2019，因此不是基础模型独立时间留出。原查询由缓存既有规则确定；本轮没有新神经训练或2020评价。','',
        '| 探针 | 当前站宏RMSE/mm | 六小时站宏RMSE/mm | 当前microRMSE/mm | 六小时microRMSE/mm |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['name']} | {r['horizon_macro_rmse_mm'][0]:.6f} | {r['horizon_macro_rmse_mm'][6]:.6f} | {r['horizon_micro_rmse_mm'][0]:.6f} | {r['horizon_micro_rmse_mm'][6]:.6f} |")
    lines+=['','## 配对区间','']
    for name,c in comparisons.items():lines.append(f"- {name} 当前/F6站宏差值95%共享月份区间：{c['macro_R_F6_delta_ci95_mm']}mm。")
    lines+=['','差值为前者减后者，负为误差减少。区间共享两种子和模型的12个月抽样，条件于固定两种子且未经多重比较校正；地区/季节分层为MSE，稀疏季节空抽样区间记NA。完整系数、训练归一化、来源哈希及分层见JSON。',
        '留出在全部背景历史GNSS/T/P上生效，单块有真实扰动不变性核验；前向留站不等于训练折外，拼接LOSO不自动形成统一Gaussian误差过程。旧未来可用性选目标与历史PWV筛选限制保留，确认及原C/D标签未读。','',
        '![误差与创新尺度](innovation_panel.png)','','[完整指标与来源](summary.json)']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k!='per_seed'} for r in rows],comparisons={k:v['macro_R_F6_delta_ci95_mm'] for k,v in comparisons.items()})),flush=True)


if __name__=='__main__':main()
