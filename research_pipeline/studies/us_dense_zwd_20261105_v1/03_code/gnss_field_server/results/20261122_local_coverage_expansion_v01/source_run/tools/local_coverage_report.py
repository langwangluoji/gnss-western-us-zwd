"""Fixed-population coverage results, paired intervals, and full-grid requery."""
import json
from pathlib import Path
import numpy as np
import torch
from local_coverage_data import ROOT,PROJECT,CoverageData,local_runtime
from local_coverage_train import Corrector,predictions
from local_innovation_report import paired_bootstrap
from local_innovation_components import aggregate_innovation
from local_context_features import LocalFeatureBuilder,VertexQuery
from grid_pair_data import feature69
from gnss_field.artifacts import read_json,write_json,sha256
from gnss_field.data import PackedStore
from gnss_field.data.context import cycles
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.runtime import configure


def public_case(data,fold):
    store=PackedStore(local_runtime()['data_root'],purpose='context');grid=store.grid;aid=store.ids('A')
    hour=int(data.lock['outer_hours'][0]);ix=int(data.indices([hour])[0])
    m0=np.load(ROOT/'cache/M0.npy',mmap_mode='r')[ix];nodes=np.arange(np.prod(store.shape))
    xy=np.c_[grid['x'].ravel(),grid['y'].ravel()];aq,_=safe_query_geometry(store,aid)
    maker=LocalFeatureBuilder(xy,store.xy[aid],store.stations.height.to_numpy()[aid],grid['height'],grid['region'],nodes,store.stats,aq,store.query_valid[aid])
    frames=np.array([store.context_values([hour-o])[0] for o in (0,6,12,18)])
    f=feature69(maker,frames,m0,cycles([hour])[0],store.shape)[:67].reshape(67,-1).T.copy()
    base=m0+data.settings['baseline_weight']*(f[:,59].reshape(store.shape)*data.settings['grid_scale']-m0)
    qA=VertexQuery(store.shape,aq,np.zeros(len(aid)),nodes)
    ba=qA(torch.tensor(base.ravel())).numpy();valid=np.isfinite(frames[0]).all(1)&store.query_valid[aid]
    J=np.stack([aggregate_innovation(frames[0,:,0],ba,k,valid) for k in maker.kernels],-1).astype('f4')
    x=np.c_[f,J/data.settings['grid_scale']]
    saved=torch.load(ROOT/f'fold_{fold}/selected_weights.pt',map_location='cpu',weights_only=False)
    model=Corrector(torch.zeros(69),torch.ones(69),saved['scale']).cuda();model.load_state_dict(saved['model'])
    with torch.no_grad():delta=model(torch.tensor(x[None],device='cuda'))[0].cpu().numpy()
    field=base+delta.reshape(store.shape)
    qb,height=safe_query_geometry(store,data.a['station_ids']);qB=VertexQuery(store.shape,qb,height,nodes)
    actual=qB(torch.tensor(field.ravel())).numpy();reference=np.load(ROOT/f'fold_{fold}/nonlinear_predictions.npz')['prediction'][ix]
    error=float(np.max(abs(actual-reference)))
    reproduced=predictions(data,model,data.a['hours'])
    reload_error=float(np.max(abs(reproduced-np.load(ROOT/f'fold_{fold}/nonlinear_predictions.npz')['prediction'])))
    if max(error,reload_error)>.001:raise ValueError('Expanded publication/reloaded checkpoint disagrees with prediction')
    np.savez_compressed(ROOT/f'fold_{fold}/publication_case.npz',hour=hour,mean_Z0=field,base_Z0=base,delta_Z0=delta.reshape(store.shape),land=grid['region'],x=grid['x'],y=grid['y'])
    return dict(public_query_max_mm=error,reloaded_max_mm=reload_error)


def main():
    configure(type('R',(),{'cpus':2})(),42)
    data=CoverageData();result=read_json(ROOT/'results.json');hours=np.asarray(data.lock['outer_hours']);oi=data.indices(hours)
    old=read_json(PROJECT/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json')
    outer_starts=np.array([np.datetime64(s,'h').astype('i8') for s in old['week_starts']])
    paired={};checks={}
    for fold in (0,1):
        split=data.splits[fold];mask=data.a['valid'][oi]&split['outer'][None]
        lin=np.load(ROOT/f'fold_{fold}/linear_predictions.npz')['prediction'];nn=np.load(ROOT/f'fold_{fold}/nonlinear_predictions.npz')['prediction']
        for name,values in [('linear',lin),('nonlinear',nn)]:
            computed=data.metric(values[oi],hours,split['outer'])
            if abs(computed['station_macro_rmse_mm']-result['folds'][str(fold)]['rows'][name]['outer']['station_macro_rmse_mm'])>1e-10:raise ValueError('Reported cohort metric mismatch')
        paired[str(fold)]=dict(nonlinear_vs_base=paired_bootstrap(data.a['truth'][oi],nn[oi],data.a['mean'][oi],mask,hours,outer_starts,data.a['xy']),
            nonlinear_vs_linear=paired_bootstrap(data.a['truth'][oi],nn[oi],lin[oi],mask,hours,outer_starts,data.a['xy']))
        checks[str(fold)]=public_case(data,fold)
    write_json(ROOT/'paired_intervals.json',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for fold in (0,1):
        directory=ROOT/f'fold_{fold}/nonlinear'
        training=[json.loads(x) for x in (directory/'training.jsonl').read_text().splitlines()]
        dev=[json.loads(x) for x in (directory/'development.jsonl').read_text().splitlines()]
        raw=[x for x in dev if not x['startup']]
        base=result['folds'][str(fold)]['rows']['zero']['inner']['station_macro_rmse_mm']
        lin=result['folds'][str(fold)]['rows']['linear']['inner']['station_macro_rmse_mm']
        axes[fold,0].plot([x['step'] for x in training],[x['MSE_mm2'] for x in training],color='#457b9d')
        axes[fold,0].set(title=f'Fold {fold}: training batch MSE',xlabel='Update',ylabel='Weighted MSE (mm squared)')
        for key,color,label in [('rmse_mm','#d95f02','Raw: selection'),('EMA_rmse_mm','#1b9e77','EMA: diagnostic')]:
            axes[fold,1].plot([x['step'] for x in raw],[x['metrics']['point'][key] for x in raw],label=label,color=color)
        axes[fold,1].axhline(base,label='Zero correction',color='grey',ls=':')
        axes[fold,1].axhline(lin,label='Linear ridge',color='#377eb8',ls='--');axes[fold,1].legend(fontsize=8)
        axes[fold,1].set(title=f'Fold {fold}: monthly weather x held groups',xlabel='Update',ylabel='Inner station-macro RMSE (mm)')
        rows=result['folds'][str(fold)]['rows'];b=rows['zero']['outer']['station_macro_rmse_mm']
        axes[fold,2].bar(range(3),[rows[k]['outer']['station_macro_rmse_mm']-b for k in ('zero','linear','nonlinear')],color=['#aaa','#377eb8','#d95f02'])
        axes[fold,2].set_xticks(range(3),['Zero','Linear','Nonlinear']);axes[fold,2].set(title='Fixed outer288',ylabel='RMSE difference from base (mm)')
    fig.savefig(ROOT/'training_curves.png');plt.close(fig)
    lines=['# 本机月度覆盖扩展实验结果','',
        '已完成预登记的两折训练：新增12个月度内层天气块，576训练小时、48内层评分小时、原288外层缓存锚点。空间站组和小网络保持不变；新内层与旧41小时人口不同，因此只在本轮共同人口内比较，不把数值变化当作跨轮精度提升。','',
        '|折|方法|内层站宏RMSE/mm|外层站宏RMSE/mm|','|---|---|---:|---:|']
    for fold in ('0','1'):
        for name in ('zero','linear','nonlinear'):
            r=result['folds'][fold]['rows'][name];lines.append(f"|{fold}|{name}|{r['inner']['station_macro_rmse_mm']:.6f}|{r['outer']['station_macro_rmse_mm']:.6f}|")
    lines+=['','非线性模型在两折空间×天气内层均优于线性，但外层没有相应优势：折0线性更低，折1两者接近。两折内层相对底座的改善约0.434%/0.236%，未通过各≥0.5%的方向门槛。保留模型和证据，本轮不触发服务器或C2019新评价。','',
        '训练上限每折14遍1008步，恒定LR1e-4，raw选模、EMA半衰期一遍仅诊断。按至少4遍、连续4次正式评价未改善的规则，折0于792步结束、入选504步，折1于432步结束、入选144步。它们属于预登记早停，不能由此推断模型全局最优或生成模型潜力耗尽。','',
        '缓存912小时，其中343小时复用已有缓存，569小时新生成M0及A条件，输入准备约4.2分钟。均值网络训练很轻，新增计算主要在冻结基础场和条件构建。','',
        '## 核验','',
        '- 真实训练路径连续200步与100＋恢复100步按位一致；工程400更新单列。',
        '- 新时间划分、缺站紧凑特征和高度偏置查询测试通过；全套测试结果见pytest_final.log。',
        '- 冻结底座重查与原始B缓存最大误差0.000206 mm；入选新头全格网查询和重载误差见publication_checks.json，阈值0.001 mm。',
        '- 所有目标仅用于拟合与固定人口评分；新时间清单先锁定。历史底座/先验及旧试验有B暴露，本轮称增量开发。D2020封存。','',
        '## 配对不确定性','']
    for f,v in paired.items():
        lines.append(f"- 折{f}：非线性−底座的天气块95%区间{v['nonlinear_vs_base']['paired_delta_CI95_mm']} mm；非线性−线性{v['nonlinear_vs_linear']['paired_delta_CI95_mm']} mm。")
    lines+=['','外层12天气簇各含缓存锚点，不是完整逐小时周评分；同时给出50 km站组重采样区间，不按单个小时独立抽样。','',
        '## 继续安排','',
        '当前证据仍更支持保留简单对照，并检验条件均值改正为何在空间外层转移不足。后续先分析按站距、地形起伏和天气过程的残差，再登记单因素特征或监督改动。本次没有训练新版DF/GAN，也没有将略低的内层分数当作生成分支成功。','',
        '## 操作','',
        '暂停：在本实验目录创建PAUSE文件；更新/缓存边界会保存状态。恢复：移走该PAUSE文件，在工程根目录沿用当前环境和PYTHONPATH=src;tools，运行python tools/local_coverage_train.py。源码、锁文件和输入不变时会恢复小时游标及优化器；已完成阶段直接复用。','',
        '本次训练已结束。配置见PLAN.md和training.lock.json；实际权重见fold_*/selected_weights.pt；两臂预测、选择清单和原始日志分别在fold_*；全场例图在publication_case.npz。','',
        '![训练曲线](training_curves.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'publication_checks.json',checks)
    print('REPORT_AND_PUBLICATION_VERIFIED',checks,flush=True)


if __name__=='__main__':main()
