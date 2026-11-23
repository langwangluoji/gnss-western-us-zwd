"""Reviewed finalization: matched point selection and same-estimator bootstrap."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import norm

from local_innovation_components import gaussian_records,gaussian_objective,fit_gaussian,InnovationHead


def bootstrap_records(error,mask,amplitude,covariance,groups,multiplicity):
    records=[]; denominator=0.
    for group in np.unique(groups):
        weight=multiplicity.get(int(group),0)
        ids=np.flatnonzero(groups==group)
        if weight==0 or len(ids)<2: continue
        if not (mask[:,ids].sum(1)>=2).any(): continue
        part=gaussian_records(error[:,ids],mask[:,ids],amplitude,covariance[np.ix_(ids,ids)],np.zeros(len(ids),int))
        for row in part: row['weight']*=weight
        denominator+=weight; records.extend(part)
    if not denominator: raise ValueError('No valid bootstrapped original groups')
    for row in records: row['weight']/=denominator
    return records


def paired_bootstrap(truth,pred,base,population,hours,week_starts,xy,replicates=2000):
    rng=np.random.default_rng(42)
    weeks=np.array([np.flatnonzero((hours>=s)&(hours<s+168)) for s in week_starts],dtype=object)
    weeks=[np.asarray(x,int) for x in weeks if len(x)]
    groups=np.floor(xy/50000).astype(int)
    _,group=np.unique(groups,axis=0,return_inverse=True)
    active=np.flatnonzero(population.any(0)); ug=np.unique(group[active])
    def score(value,mask):
        count=mask.sum(0); keep=count>0
        return np.sqrt(np.where(mask,value,0.).sum(0)[keep]/count[keep]).mean()
    pe=(pred-truth)**2; be=(base-truth)**2; time_only=[]; joint=[]
    for _ in range(replicates):
        rows=np.concatenate([weeks[i] for i in rng.integers(0,len(weeks),len(weeks))])
        time_only.append(score(pe[rows],population[rows])-score(be[rows],population[rows]))
        drawn=rng.choice(ug,len(ug),replace=True)
        cols=np.concatenate([active[group[active]==g] for g in drawn])
        joint.append(score(pe[np.ix_(rows,cols)],population[np.ix_(rows,cols)])-score(be[np.ix_(rows,cols)],population[np.ix_(rows,cols)]))
    return dict(weather_clusters=len(weeks),replicates=replicates,
        paired_delta_CI95_mm=np.quantile(time_only,[.025,.975]).tolist(),
        weather_and_50km_group_CI95_mm=np.quantile(joint,[.025,.975]).tolist())


def finalize():
    from local_innovation_pilot import Pilot,ROOT,SOURCE,PROJECT,ARMS,probability_metrics,publication_check
    from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
    from gnss_field.runtime import configure
    configure(type('R',(),{'cpus':2})(),42)
    started=time.monotonic(); data=Pilot()
    points=read_json(ROOT/'point_results.json')
    original=read_json(ROOT/'gaussian_results.json')
    outer=np.asarray(data.lock['outer_hours']); oi=data.indices(outer)
    audit=read_json(PROJECT/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json')
    week_starts=np.array([np.datetime64(s,'h').astype('i8') for s in audit['week_starts']])
    cov=data.a['projection'].astype('f8')@data.a['projection'].astype('f8').T+data.a['remainder'].astype('f8')@data.a['remainder'].astype('f8').T
    _,groups=np.unique(np.floor(data.a['xy']/200000).astype(int),axis=0,return_inverse=True)
    final={}; profiles={}; validations={}
    for fold in (0,1):
        split=data.splits[fold]; p=points[str(fold)]
        for arm in ARMS:
            s=read_json(ROOT/f'fold_{fold}/{arm}/status.json')
            if s['status']!='COMPLETE' or s['step']!=s['budget']: raise ValueError('Incomplete point training')
        # Correct the preliminary exact-diagnostic eligibility, without changing
        # any trained weights or consulting outer scores to choose a model.
        order=['zero','constant','gated','free']
        selected=min(order,key=lambda k:(p['rows'][k]['inner']['station_macro_rmse_mm'],order.index(k)))
        model=None
        if selected=='zero': mean=data.a['mean'].copy()
        else:
            state=torch.load(ROOT/f'fold_{fold}/{selected}/point_best.pt',map_location='cpu',weights_only=False)
            model=InnovationHead(selected,data.settings['grid_scale']).cuda();model.load_state_dict(state['model'])
            mean=data.predict(model,data.a['hours'])
        directory=ROOT/f'fold_{fold}/reviewed'; directory.mkdir(exist_ok=True)
        immutable_json(directory/'point_selection.lock.json',dict(selected=selected,order=order,
            selection='same new spatial x weather inner; analytic constant excluded as diagnostic',
            source=sha256(ROOT/f'fold_{fold}/point_selection.lock.json')))
        np.savez_compressed(directory/'mean.npz',hours=data.a['hours'],prediction=mean)
        pred=mean[oi]; pop=data.a['valid'][oi]&split['outer'][None]
        paired=paired_bootstrap(data.a['truth'][oi],pred,data.a['mean'][oi],pop,outer,week_starts,data.a['xy'])
        raw={}
        for arm in ARMS:
            last=torch.load(ROOT/f'fold_{fold}/{arm}/last.pt',map_location='cpu',weights_only=False)
            m=InnovationHead(arm,data.settings['grid_scale']).cuda();m.load_state_dict(last['model'])
            raw[arm]=dict(step=last['step'],inner=data.metric(data.predict(m,split['inner_hours']),split['inner_hours'],split['inner']),
                outer=data.metric(data.predict(m,outer),outer,split['outer']),interpretation='last raw diagnostic; no post-hoc checkpoint or product selection')
        train=data.indices(split['train_hours']); mask=data.a['valid'][train]&split['fit'][None]
        residual=data.a['truth'][train]-mean[train]
        amplitude=data.a['amplitude'][train]
        records=gaussian_records(residual,mask,amplitude,cov,groups)
        same=bootstrap_records(residual,mask,amplitude,cov,groups,{int(g):1 for g in np.unique(groups)})
        if abs(gaussian_objective(np.zeros(2),same,data.settings['tau'])[0]-gaussian_objective(np.zeros(2),records,data.settings['tau'])[0])>1e-10:
            raise ValueError('Bootstrap objective changed original grouping')
        fits={k:fit_gaussian(records,data.settings['tau'],k) for k in ('field_only','common','two_scales')}
        fits['original']=dict(field=1.,nugget=1.,NLL=gaussian_objective(np.zeros(2),records,data.settings['tau'])[0])
        hi=split['inner_hours']; order_g=['original','field_only','common','two_scales']
        for kind,v in fits.items():
            v['inner']=probability_metrics(data,mean[data.indices(hi)],hi,split['inner'],v['field'],v['nugget'])
        winner=min(order_g,key=lambda k:(fits[k]['inner']['group_ES_mm'],order_g.index(k)))
        # New audit version records the explicitly declared optimization tolerance.
        immutable_json(directory/'probability_selection_v03.lock.json',dict(selected=winner,fits=fits,selection='inner group ES K64; no outer selection',source_point=sha256(directory/'point_selection.lock.json')))
        for kind,v in fits.items():v['outer']=probability_metrics(data,mean[oi],outer,split['outer'],v['field'],v['nugget'])
        print('REVIEWED',fold,selected,winner,flush=True)
        grid=np.linspace(.1,1.8,41)
        profile=np.array([[gaussian_objective(np.log([f,n]),records,data.settings['tau'])[0] for n in grid] for f in grid])
        blocks=(split['train_hours']-split['train_hours'].min())//168
        ug=[int(g) for g in np.unique(groups) if ((groups==g)&split['fit']).sum()>=2]
        rng=np.random.default_rng(42); boot=[]
        for rep in range(100):
            selected_blocks=rng.choice(np.unique(blocks),len(np.unique(blocks)),replace=True)
            rows=np.concatenate([np.flatnonzero(blocks==b) for b in selected_blocks])
            drawn=rng.choice(ug,len(ug),replace=True); ids,counts=np.unique(drawn,return_counts=True)
            br=bootstrap_records(residual[rows],mask[rows],amplitude[rows],cov,groups,dict(zip(ids,counts)))
            fit=fit_gaussian(br,data.settings['tau'],'two_scales',gradient_tolerance=1e-7);boot.append([fit['field'],fit['nugget']])
        np.savez_compressed(directory/'gaussian_profile.npz',scales=grid,NLL=profile,bootstrap=np.asarray(boot))
        # Verify predictions do not read target values at inference.
        base_truth=data.a['truth'].copy()
        if model is not None:
            before=data.predict(model,hi); data.a['truth'][:]=123456.
            after=data.predict(model,hi); data.a['truth'][:]=base_truth
            if not np.array_equal(before,after):raise ValueError('Inference label leakage')
        # Finite/noise checks and published grid query were already verified on
        # trained EMA heads. Export corrected winner as a separate full field.
        if selected=='gated':
            old=np.load(ROOT/f'fold_{fold}/publication_case.npz')
            if points[str(fold)]['selected']==selected:
                np.savez_compressed(directory/'publication_case.npz',**dict(old))
            else:
                export_reviewed_case(data,fold,model,directory)
        final[str(fold)]=dict(point_selected=selected,point=p,raw_last=raw,paired=paired,
            selected_outer=data.metric(pred,outer,split['outer']),probability_selected=winner,Gaussian=fits,
            corrected_bootstrap_percentiles=np.quantile(boot,[.025,.5,.975],axis=0).tolist(),
            corrected_bootstrap_replicates=len(boot),bootstrap_spatial_unit='same original 200km composite groups',
            old_bootstrap_status='INVALID_ESTIMATOR_GROUP_CHANGE_DO_NOT_USE',
            exact_constant_status='optimization diagnostic only; excluded from matched-budget selector')
    write_json(ROOT/'reviewed_results.json',final)
    plot_and_report(data,final)
    write_json(ROOT/'review_acceptance.json',dict(complete=True,all_scientific_arms_completed=True,
        recovery=read_json(ROOT/'engineering.json'),outer_hours=len(outer),C2019=False,D2020=False,
        server_jobs=False,EMA_initial_weight_retention=.999**84,
        finalization_source=sha256(__file__),seconds=time.monotonic()-started,
        preliminary_files_superseded=['point_results.json selection','gaussian_results.json','fold_*/gaussian_profile.npz bootstrap']))


def export_reviewed_case(data,fold,model,directory):
    # The matched gated head and exact diagnostic share the very same frozen
    # base, but export from all-grid features instead of a station shortcut.
    from local_innovation_pilot import SOURCE,ROOT
    from local_context_features import LocalFeatureBuilder,VertexQuery
    from grid_pair_data import feature69,local_runtime
    from gnss_field.data import PackedStore
    from gnss_field.data.context import cycles
    from gnss_field.training.prediction import safe_query_geometry
    from local_innovation_components import aggregate_innovation
    store=PackedStore(local_runtime()['data_root'],purpose='context');aid=store.ids('A');grid=store.grid
    hour=int(data.lock['outer_hours'][0]);ix=int(data.indices([hour])[0]);nodes=np.arange(np.prod(store.shape))
    m0=np.load(SOURCE/'M0_cache/mean.npy',mmap_mode='r')[ix];aq,_=safe_query_geometry(store,aid)
    xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    b=LocalFeatureBuilder(xy,store.xy[aid],store.stations.height.to_numpy()[aid],grid['height'],grid['region'],nodes,store.stats,aq,store.query_valid[aid])
    frames=np.array([store.context_values([hour-o])[0] for o in (0,6,12,18)])
    features=feature69(b,frames,m0,cycles([hour])[0],store.shape)[:67].reshape(67,-1).T.copy()
    base=m0+data.settings['baseline_weight']*(features[:,59].reshape(store.shape)*data.settings['grid_scale']-m0)
    q=VertexQuery(store.shape,aq,np.zeros(len(aid)),nodes)
    ba=q(torch.tensor(base.ravel())).numpy();valid=np.isfinite(frames[0]).all(1)&store.query_valid[aid]
    J=np.stack([aggregate_innovation(frames[0,:,0],ba,k,valid) for k in b.kernels],-1).astype('f4')
    with torch.no_grad(): delta=model(torch.tensor(features[None],device='cuda'),torch.tensor(J[None],device='cuda'))[0].cpu().numpy()
    field=base+delta.reshape(store.shape)
    qb,bh=safe_query_geometry(store,data.a['station_ids']);qB=VertexQuery(store.shape,qb,bh,nodes)
    pred=qB(torch.tensor(field.ravel())).numpy();ref=data.predict(model,[hour])[0]
    error=float(np.max(abs(pred-ref)))
    if error>.001:raise ValueError('Reviewed publication differs from prediction')
    np.savez_compressed(directory/'publication_case.npz',hour=hour,mean_Z0=field,base_Z0=base,correction_Z0=delta.reshape(store.shape),land=grid['region'],x=grid['x'],y=grid['y'])
    (directory/'publication_check.json').write_text(json.dumps({'max_query_error_mm':error}),encoding='utf-8')


def plot_and_report(data,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from local_innovation_pilot import ROOT,ARMS
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for fold in (0,1):
        for arm in ARMS:
            lines=[json.loads(x) for x in (ROOT/f'fold_{fold}/{arm}/development.jsonl').read_text().splitlines()]
            x=[r['step'] for r in lines]; base=lines[0]['EMA']['station_macro_rmse_mm']
            axes[fold,0].plot(x,[r['EMA']['station_macro_rmse_mm']-base for r in lines],label=arm)
            axes[fold,1].plot(x,[r['raw']['station_macro_rmse_mm']-base for r in lines],label=arm)
        for k in (0,1): axes[fold,k].axhline(0,color='grey',lw=.8);axes[fold,k].legend();axes[fold,k].set(xlabel='Update',ylabel='Inner RMSE minus step0 (mm)',title=f'Fold {fold}: '+('EMA .999' if k==0 else 'Raw weights (diagnostic)'))
    fig.savefig(ROOT/'training_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for fold,ax in enumerate(axes):
        rows=results[str(fold)]['point']['rows']; keys=['zero','constant','gated','free','constant_exact']
        base=rows['zero']['outer']['station_macro_rmse_mm']
        ax.bar(np.arange(5),[rows[k]['outer']['station_macro_rmse_mm']-base for k in keys],color=['#9aa4b2','#457b9d','#2a9d8f','#e9c46a','#e76f51'])
        ax.set_xticks(np.arange(5),['Zero','Constant','Gated','Free','Exact const.\n(diagnostic)']);ax.axhline(0,color='grey',lw=.8)
        ax.set(title=f'Fold {fold}: outer288 paired cohort',ylabel='RMSE difference vs base (mm)')
    fig.savefig(ROOT/'point_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for fold,ax in enumerate(axes):
        p=np.load(ROOT/f'fold_{fold}/reviewed/gaussian_profile.npz');g=p['scales'];z=p['NLL']-p['NLL'].min()
        c=ax.contourf(g,g,z.T,levels=[0,.0001,.0005,.001,.005,.01,.05,.1,.5,2],cmap='Blues',extend='max')
        ax.plot(p['bootstrap'][:,0],p['bootstrap'][:,1],'.',color='#d95f02',alpha=.5,ms=3)
        ax.set(xlabel='Field scale',ylabel='Nugget scale',title=f'Fold {fold}: training composite NLL')
    fig.colorbar(c,ax=axes,label='NLL above grid minimum',format='%.1g');fig.savefig(ROOT/'gaussian_allocation.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for fold in (0,1):
        r=results[str(fold)];g=r['Gaussian'];keys=['original','field_only','common','two_scales']
        axes[fold,0].bar(range(4),[g[k]['outer']['group_ES_mm'] for k in keys],color='#457b9d')
        axes[fold,0].set_xticks(range(4),keys,rotation=15);axes[fold,0].set(title=f'Fold {fold}: outer group ES',ylabel='ES (mm)')
        axes[fold,1].scatter([g[k]['outer']['analytic_width90_mm'] for k in keys],[100*g[k]['outer']['analytic_coverage90'] for k in keys],c=range(4),cmap='viridis')
        for k in keys:axes[fold,1].annotate(k,(g[k]['outer']['analytic_width90_mm'],100*g[k]['outer']['analytic_coverage90']),fontsize=8)
        axes[fold,1].axhline(90,color='grey',ls='--');axes[fold,1].set(xlabel='Analytical 90% width (mm)',ylabel='Coverage (%)',title='Gaussian calibration / width')
    fig.savefig(ROOT/'probability_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for fold,ax in enumerate(axes):
        ix=data.indices(data.lock['outer_hours']);pop=data.a['valid'][ix]&data.splits[fold]['outer'][None]
        pred=np.load(ROOT/f'fold_{fold}/reviewed/mean.npz')['prediction'][ix]
        count=pop.sum(0);keep=count>0
        base=np.sqrt(np.where(pop,(data.a['mean'][ix]-data.a['truth'][ix])**2,0).sum(0)[keep]/count[keep])
        new=np.sqrt(np.where(pop,(pred-data.a['truth'][ix])**2,0).sum(0)[keep]/count[keep])
        v=ax.scatter(data.a['xy'][keep,0]/1000,data.a['xy'][keep,1]/1000,c=new-base,cmap='RdBu_r',s=15)
        ax.set(title=f'Fold {fold}: matched EMA mean minus base',xlabel='Lambert x (km)',ylabel='Lambert y (km)');fig.colorbar(v,ax=ax,label='RMSE difference (mm)')
    fig.savefig(ROOT/'station_gain_map.png');plt.close(fig)
    lines=['# 本机小实验结果与下一步判定','',
        '本轮完成了三个点式均值头的两折、两遍小训练，以及冻结均值后的Gaussian误差分配审计。小训练使用336小时，每折84次更新；内层为四个新天气块内41个缓存时刻和独立物理/50 km站组。外层为原12个注册天气周内288个缓存锚点，折0/1分别204/244站。这是B2018增量开发，不与上轮96小时或C2019全年数字直接排序。','',
        '|方法|折0外层站宏RMSE/mm|折1外层站宏RMSE/mm|','|---|---:|---:|']
    for key in ['zero','constant','gated','free','constant_exact']:
        lines.append(f"|{key}|{results['0']['point']['rows'][key]['outer']['station_macro_rmse_mm']:.6f}|{results['1']['point']['rows'][key]['outer']['station_macro_rmse_mm']:.6f}|")
    lines+=['','constant_exact为训练资料精确拟合的两系数优化诊断，不参与匹配预算的选模。初步结果曾将它纳入折0选择；审阅后已排除，并重新冻结匹配训练均值、重做全部Gaussian校准。最终结果只读取reviewed目录与reviewed_results.json。','',
        'EMA=0.999在84步后仍保留约91.9%的初始权重贡献。原始权重曲线单列诊断，不在看到外层后追加选模。当前神经头修正很小，尚不足以判断结构容量，也没有达到收敛证据。','',
        '|折|匹配预算均值|Gaussian内层入选|空间尺度|独立项尺度|外层ES 原→新|解析90%覆盖|','|---|---|---:|---:|---:|---:|']
    for fold in ('0','1'):
        r=results[fold];v=r['Gaussian'][r['probability_selected']];b=r['Gaussian']['original']
        lines.append(f"|{fold}|{r['point_selected']}|{r['probability_selected']}|{v['field']:.4f}|{v['nugget']:.4f}|{b['outer']['group_ES_mm']:.4f} → {v['outer']['group_ES_mm']:.4f}|{100*v['outer']['analytic_coverage90']:.2f}%|")
    lines+=['','所有概率模型使用同一入选均值，缩放独立项不会改变RMSE。两个尺度是统计残差分配，不是仪器噪声的物理测定。二维NLL及100次同步天气块×原200 km站群重采样只作小样本稳定性诊断；旧版50 km重采样改变了估计目标，已判无效并由reviewed结果替代。','',
        '## 点值配对区间','']
    for fold,r in results.items():lines.append(f"- 折{fold}，入选均值相对基础场：天气块95%区间 {r['paired']['paired_delta_CI95_mm']} mm；叠加50 km站组重采样 {r['paired']['weather_and_50km_group_CI95_mm']} mm。")
    lines+=['','12个天气簇各为缓存锚点，不是12个全小时完整周；原底座与概率先验有历史B暴露。C2019未读取评分，D2020保持封存。','',
        '## 下一步','',
        '保留M0＋IDW作为点值主对照。先在锁定的新空间内层上安排一次优化诊断：按实际更新总量设置EMA有效窗口，检查原始权重与EMA差异，以及当前A创新能解释的残差幅度。常数精确解跨折内层未一致改善，暂不据此替换基础均值。','',
        'Gaussian校准提供了比未校准先验更强的概率起点。后续先检验条件残差尾部/偏态与空间依赖是否仍有稳定问题，再比较确定性尺度/偏态小头；Diffusion/GAN沿用同一冻结均值和Gaussian对照。当前结果不支持直接扩大生成器，也不证明生成器无法获益。','',
        '本轮没有训练Diffusion/GAN，没有提交服务器作业。数学/查询和恢复验收用于确认工程路径；这些检查不替代独立站精度证据。','',
        '## 文件','',
        '- `reviewed_results.json`：修正选模后的全部指标、配对区间、原始权重诊断。',
        '- `fold_*/reviewed/`：最终均值、锁定选择、原估计量重采样及全格网例图数据。',
        '- `training_curves.png`、`point_comparison.png`、`station_gain_map.png`、`gaussian_allocation.png`、`probability_comparison.png`。',
        '- `experiment.lock.json`、`code_snapshot/`：实际训练输入与版本；`review_acceptance.json`：后处理验收与旧文件替代关系。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':finalize()
