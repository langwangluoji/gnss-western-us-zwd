"""Report the locked B-only experiment, without promoting or reading C/D labels."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from local_context_data import PROJECT, RUN, DESIGN
from local_context_train import StudyData, ThreeArmMean, ARMS
from local_probe_diagnostics import save_npz, MEAN
from architecture_result_analysis import rmse_interval
from gnss_field.artifacts import read_json, write_json, sha256, atomic_bytes
from gnss_field.evaluation import point_metrics
from gnss_field.evaluation.bootstrap import _blocks
from gnss_field.training.engine import HourCursor
from gnss_field.runtime import configure


def promotion_gate(scores, folds, intervals):
    gains = {name:1-scores['candidate']/scores[name] for name in ('PCA32_R64','linear_control')}
    rules = dict(three_percent=all(g >= .03 for g in gains.values()),
        better_than_zero=scores['candidate'] < scores['zero'],
        both_folds=all(row['candidate'] < min(row['PCA32_R64'],row['linear_control']) for row in folds),
        intervals=all(v[1] < 0 for v in intervals.values()))
    return dict(passed=all(rules.values()), relative_gains=gains, rules=rules)


def plots(data, predictions, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'zero':'#90969c', ARMS[0]:'#8764a0', ARMS[1]:'#c18b30', ARMS[2]:'#237c9c', 'linear_control':'#398262'}
    labels = {'zero':'Frozen mix',ARMS[0]:'PCA32 / R64', ARMS[1]:'Local / R64',ARMS[2]:'Local / grid','linear_control':'Local linear'}
    fig, axes = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for fold in range(2):
        for arm in ARMS:
            folder = RUN/f'fold_{fold}'/arm
            zero = read_json(folder/'initial_metrics.json')['joint']
            rows = [json.loads(s) for s in (folder/'development.jsonl').read_text().splitlines()]
            for col,key in enumerate(('rmse_mm','value')):
                axes[fold,col].plot([0]+[r['step'] for r in rows],
                    [zero[key]]+[r['metrics']['joint'][key] for r in rows],
                    color=colors[arm],marker='o',markersize=3,label=labels[arm])
        for col, ylabel in enumerate(('Inner B station-macro RMSE (mm)','Inner B ES, K16 (mm)')):
            axes[fold,col].set(xlabel='Update',ylabel=ylabel,title=f'Fold {fold+1}: inner time holdout, fitted groups')
            axes[fold,col].grid(alpha=.2); axes[fold,col].legend(fontsize=8)
    fig.savefig(RUN/'training_curves.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    names=list(predictions)
    bars=ax.bar(np.arange(len(names)),[metrics[k]['station_macro_rmse_mm'] for k in names],color=[colors[k] for k in names])
    ax.bar_label(bars,fmt='%.4f',padding=3)
    ax.set(xticks=np.arange(len(names)),xticklabels=[labels[k] for k in names],ylabel='Outer B station-macro RMSE (mm)',
        title='B2018 incremental development: 288 hours, held spatial groups')
    ax.set_ylim(0,max(metrics[k]['station_macro_rmse_mm'] for k in names)*1.12)
    fig.savefig(RUN/'outer_comparison.png',dpi=180); plt.close(fig)
    outer=data.a['outer']; pop=data.a['valid'][outer]; truth=data.a['truth'][outer]
    count=pop.sum(0); active=count>0
    station=pd.read_csv(data.root/'stations.csv')
    table=station.copy();table['outer_valid_hours']=count
    for name,prediction in predictions.items():
        table[name+'_RMSE_mm']=np.sqrt(np.where(pop,(prediction-truth)**2,0).sum(0)/np.maximum(count,1))
    table.to_csv(RUN/'station_results.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    limit=.5
    for ax,arm in zip(axes,ARMS[1:]):
        value=table[arm+'_RMSE_mm']-table[ARMS[0]+'_RMSE_mm']
        dots=ax.scatter(station.x[active]/1000,station.y[active]/1000,c=value[active],s=14,cmap='coolwarm',vmin=-limit,vmax=limit)
        ax.set(title=labels[arm]+' minus matched PCA',xlabel='Lambert x (km)',ylabel='Lambert y (km)');ax.set_aspect('equal')
        fig.colorbar(dots,ax=ax,label='Station RMSE change (mm); blue improves',extend='both')
    fig.savefig(RUN/'station_gain_maps.png',dpi=180);plt.close(fig)


def field_plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hour_index=int(data.a['outer'][0])  # fixed first outer anchor; no error/event selection
    features=data.features_for([hour_index]);amp=data.t['amplitude'][hour_index]
    raw=np.asarray(data.features[hour_index])
    scale=data.settings['grid_scale'];mix=data.settings['baseline_weight']
    base=scale*(raw[:,58]+mix*(raw[:,59]-raw[:,58]))
    corrections=[]
    for arm in ARMS[1:]:
        selected=read_json(RUN/'fold_0'/arm/'selection.lock.json')
        if sha256(selected['selected']['checkpoint']) != selected['sha256']:
            raise ValueError('Selected field checkpoint differs')
        model=ThreeArmMean(arm,data.a,scale).to(data.device)
        model.load_state_dict(torch.load(selected['selected']['checkpoint'],map_location=data.device,weights_only=False)['ema'])
        with torch.no_grad():
            value=model.head(features)*scale
            if arm==ARMS[1]:
                value=(value[:,data.a['land']] @ model.operator.T) @ torch.tensor(data.a['grid_basis'],device=data.device).T
            corrections.append((value*amp).cpu().numpy()[0])
    shape=tuple(data.settings['shape']);land=data.a['land'];nodes=data.a['nodes']
    fields=[]
    for row in (base,*corrections):
        full=np.full(np.prod(shape),np.nan,'f4');full[nodes[land]]=row[land];fields.append(full.reshape(shape))
    fig,axes=plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
    for ax,field,title in zip(axes,fields,('Frozen mixture Z0 (mm)','Local / R64 correction (mm)','Local / grid correction (mm)')):
        kw={} if title.startswith('Frozen') else dict(cmap='coolwarm',vmin=-3,vmax=3)
        image=ax.imshow(field,origin='lower',**kw);ax.set_title(title,fontsize=10);ax.set_xlabel('25 km grid column');ax.set_ylabel('Grid row')
        fig.colorbar(image,ax=ax,shrink=.7)
    fig.suptitle(f'First registered outer anchor: {pd.to_datetime(data.a["hours"][hour_index],unit="h")} | fold1 model; no dense truth')
    fig.savefig(RUN/'field_example.png',dpi=180);plt.close(fig)
    save_npz(RUN/'field_example.npz',hour=data.a['hours'][hour_index],base=fields[0],local_R64=fields[1],local_grid=fields[2])


def main():
    configure(type('Runtime',(),{'cpus':2})(),42)
    complete=read_json(RUN/'training_complete.json')
    if complete['scientific_updates']>6000 or complete['status']!='COMPLETE':
        raise ValueError('Training is not complete within the registered cap')
    data=StudyData(); outer=data.a['outer'];pop=data.a['valid'][outer];truth=data.a['truth'][outer]
    predictions={'zero':data.a['mean'][outer].copy()}
    fold_results={};costs=[]
    for name in (*ARMS,'linear_control'):
        mean=data.a['mean'][outer].copy();fold_results[name]=[]
        for fold in range(2):
            directory=RUN/f'fold_{fold}'/name
            result=read_json(directory/'results.json');fold_results[name].append(result)
            forecast=dict(np.load(directory/'outer_predictions.npz'))
            np.testing.assert_array_equal(forecast['hours'],data.a['hours'][outer])
            held=data.a['folds']==fold
            np.testing.assert_array_equal(forecast['population'],pop & held[None])
            mean[:,held]=forecast['mean'][:,held]
            if name in ARMS:
                state=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
                cursor=HourCursor(data.a['hours'][data.a['core']],42)
                supervised=0
                for _ in range(state['step']):
                    ix=np.searchsorted(data.a['hours'],cursor.take(8))
                    supervised+=int((data.a['valid'][ix] & ~held[None]).sum())
                costs.append(dict(arm=name,fold=fold,updates=state['step'],
                    elapsed_update_and_inner_evaluation_seconds=state['elapsed'],
                    supervised_station_hours_replayed=supervised,
                    legacy_log_zero_counter='not instrumented; use exact replay above'))
        predictions[name]=mean
    metrics={name:data.metrics(prediction,outer,np.ones(data.n,bool),64,20261118)
        for name,prediction in predictions.items()}
    station=pd.read_csv(data.root/'stations.csv');groups=station.dense_group.to_numpy()
    intervals={};gates={}
    for name in ARMS[1:]:
        intervals[name]={control:{str(days):rmse_interval(data.a['hours'][outer],truth,predictions[name],predictions[control],
            pop,days=days,groups=groups) for days in (7,14)} for control in (ARMS[0],'linear_control')}
        scores=dict(candidate=metrics[name]['station_macro_rmse_mm'], **{k:metrics[k]['station_macro_rmse_mm'] for k in (ARMS[0],'linear_control','zero')})
        folds=[dict(candidate=fold_results[name][f]['outer_B']['station_macro_rmse_mm'],
            **{k:fold_results[k][f]['outer_B']['station_macro_rmse_mm'] for k in (ARMS[0],'linear_control')}) for f in range(2)]
        gates[name]=promotion_gate(scores,folds,{k:intervals[name][k]['7']['ci95_mm'] for k in (ARMS[0],'linear_control')})
    training_diagnostics={}
    for name in ARMS:
        rows=[]
        for fold in range(2):
            selection=read_json(RUN/f'fold_{fold}'/name/'selection.lock.json')
            if sha256(selection['selected']['checkpoint'])!=selection['sha256']:
                raise ValueError('Selected checkpoint differs from lock')
            model=ThreeArmMean(name,data.a,data.settings['grid_scale']).to(data.device)
            model.load_state_dict(torch.load(selection['selected']['checkpoint'],map_location=data.device,weights_only=False)['ema'])
            fit=data.a['folds']!=fold;core=data.a['core']
            rows.append(dict(fold=fold,core_model=point_metrics(data.a['truth'][core],data.predict(model,core),data.a['valid'][core]&fit[None]),
                core_zero=point_metrics(data.a['truth'][core],data.a['mean'][core],data.a['valid'][core]&fit[None])))
        training_diagnostics[name]=rows
    assert sha256(MEAN)=='605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0'
    block7=_blocks(data.a['hours'][outer],7);block14=_blocks(data.a['hours'][outer],14)
    same_partition=len(block7)==len(block14) and all(np.array_equal(a,b) for a,b in zip(block7,block14))
    result=dict(status='COMPLETE',evaluation='B2018 incremental development, outer288 hours with two held physical-group folds; historical base/prior have B exposure',
        population=dict(stations=int(pop.any(0).sum()),station_hours=int(pop.sum()),hours=len(outer),
            time_blocks_7day=12, nominal_B_stations=data.n),metrics=metrics,paired_intervals=intervals,
        gates=gates,training_diagnostics=training_diagnostics,costs=costs,
        scientific_updates=complete['scientific_updates'],new_server_jobs=False,
        pooled_ES_scope='cross-fitted stitched joint diagnostic, not one publishable full-field model; per-fold ES uses each selected model',
        bootstrap_support=dict(blocks7=[len(b) for b in block7],blocks14=[len(b) for b in block14],
            partitions_identical=same_partition,
            interpretation='Twelve 24-observed-hour anchors; 7/14-day bins do not provide separate complete-week dependence robustness checks'),
        C2019=False,D2020=False,M0_unchanged=True,
        decision='LOCAL_GATE_PASSED_REQUIRES_MEASURED_SERVER_BUDGET' if any(r['passed'] for r in gates.values()) else 'STOP_BEFORE_SERVER_NO_PROMOTED_ARCHITECTURE')
    write_json(RUN/'results.json',result)
    save_npz(RUN/'outer_combined_predictions.npz',hours=data.a['hours'][outer],truth=truth,population=pop,**predictions)
    plots(data,predictions,metrics);field_plot(data)
    lines=['# 局部条件残差三臂短试结果','',
        '本轮在本机完成三臂、两折、每组1000步，共6000次正式更新。比较人口为2018年B站的288个外层时刻；冻结底座与先验曾使用B资料，因此结果属于增量开发检验。','',
        '| 模型 | 外层站宏RMSE / mm | 交叉拟合拼接ES K64 / mm |','|---|---:|---:|']
    names={'zero':'冻结M0与IDW混合底座',ARMS[0]:'PCA32→R64',ARMS[1]:'局部99维→R64',ARMS[2]:'局部99维→格网','linear_control':'内层选定局部线性对照'}
    for name in predictions:
        lines.append(f'| {names[name]} | {metrics[name]["station_macro_rmse_mm"]:.6f} | {metrics[name]["ES_mm"]:.6f} |')
    lines+=['',f'实际评分为{int(pop.any(0).sum())}站、{int(pop.sum()):,}站时刻。三臂神经模型均由内层ES选择EMA，零修正可参与选模；线性对照按相同内层ES选定参数。外层结果不用于挑选训练步数。',
        '表中总体ES将两个折模型各自留出站的均值拼接后计算，属于交叉拟合联合诊断，不是一套可发布模型的整场ES；每折单独ES及checkpoint记录在各折results.json。',
        '外层仅包含12个24小时锚点段。7日和14日时间箱在这些锚点上形成完全相同的12组，区间相同；它们不构成完整7日/14日连续记录的两项独立稳健性检验。','',
        '局部两臂没有通过相对匹配PCA和简单线性控制改善至少3%、两折同向及配对区间共同成立的晋级门槛。本轮队列在本机结束，保留既有成熟mean产品。', '',
        '第一折的局部修正出现小幅收益，第二折方向不一致。局部条件和自由格网在当前预算下未形成稳定空间泛化证据；这尚不能区分优化不足、监督覆盖与输入信息不足。三组均在预算末选中checkpoint，也不足以宣称已收敛。','',
        '下一项有价值的诊断是先检验训练组可拟合收益与留组收益的差距，并核对独立天气背景能够补充哪些A站IDW条件中没有的信息；单独登记后再决定训练规模。','',
        '本轮尚未运行C2019全年评价，D2020保持封存。表中约7–8mm是B2018小样本开发口径，不能与此前C2019全年9.908mm直接排名。','',
        '图件：training_curves.png、outer_comparison.png、station_gain_maps.png、field_example.png。场图展示预登记首个外层时刻的公共高度Z0和修正量，25km为输出间距。','',
        '首个场图的冻结混合底座在13个陆地格点出现负Z0，均位于当前输入支持域外；当前支持域2087格、全陆地3724格。该现象单列于field_support_diagnostics.json，后续场发布需要同步提供支持标记并核查物理范围。','',
        '监控：http://127.0.0.1:8778 。只读追加日志，每5秒刷新。日志中的旧累计监督计数未接线，results.json另按相同采样游标精确重放给出站时刻数。']
    atomic_bytes(RUN/'RESULTS.md',('\n'.join(lines)+'\n').encode('utf-8'))
    print('RESEARCH_RESULT',result['decision'],{k:v['station_macro_rmse_mm'] for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    main()
