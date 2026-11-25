"""Archive lightweight evidence and independently recompute saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.metrics import evaluate
from tools.terrain_diffusion.joint_experiment import fair_crps,load_events,scores
from tools.terrain_diffusion.data import sha,write_json


def json_read(p):return json.loads(p.read_text(encoding='utf-8'))


def strata(samples,truth,cohort):
    error=samples.mean(0)-truth;crps=fair_crps(samples,truth);valid=np.isfinite(truth)
    rows=[]
    for feature,edges in cohort.items():
        values,bounds=edges
        for low,high in zip(bounds[:-1],bounds[1:]):
            mask=valid&(values>=low)&(values<high);count=int(mask.sum())
            rows.append(dict(feature=feature,lower=low,upper=None if np.isinf(high) else high,count=count,
                micro_rmse_mm=float(np.sqrt(np.mean(error[mask]**2))) if count else None,
                fair_crps_mm=float(crps[mask].mean()) if count else None))
    return rows


def pair_scores(samples,truth,xy):
    distance=np.linalg.norm(xy[:,None]-xy[None],axis=-1);pairs=np.array(np.triu_indices(len(xy),1)).T
    rows=[]
    for low,high in ((0,50),(50,150),(150,300)):
        chosen=pairs[(distance[pairs[:,0],pairs[:,1]]>=low)&(distance[pairs[:,0],pairs[:,1]]<high)]
        square=0.;crps=0.;count=0
        for start in range(0,len(chosen),64):
            p=chosen[start:start+64];target=truth[...,p[:,0]]-truth[...,p[:,1]]
            pred=samples[...,p[:,0]]-samples[...,p[:,1]];ok=np.isfinite(target)
            square+=float(((pred.mean(0)-target)[ok]**2).sum())
            crps+=float(fair_crps(pred,target)[ok].sum());count+=int(ok.sum())
        rows.append(dict(distance_km=[low,high],valid_pair_hours=count,rmse_mm=(square/count)**.5 if count else None,fair_crps_mm=crps/count if count else None))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    assert json_read(run/'status.json')['status']=='COMPLETE_ALL'
    assert json_read(run/'velocity_control/status.json')['status']=='COMPLETE_4_VELOCITY_CONTROLS'
    geo=dict(np.load(ENGINE/'results/20261125_temporal_continuous_data/geometry.npz'))
    hours=np.load(ENGINE/'results/20261125_temporal_continuous_data/hours.npy')
    values=np.load(ENGINE/'results/20261125_temporal_continuous_data/values.npy',mmap_mode='r')
    records=[];checks=[];manifest={};analyses={}
    train=load_events(run/'oof','train');inner=load_events(run/'oof','inner')
    foundation_base=scores(np.repeat(inner['mu'][None],2,axis=0),inner['truth'])
    foundation_train=scores(np.repeat(train['mu'][None],2,axis=0),train['truth'])
    cohort={};nearest=[]
    for f in range(5):
        pack=dict(np.load(run/'oof'/f'fold{f}/inner.npz'));c=pack['condition_ids'];distance=np.linalg.norm(geo['xy'][:,None]-geo['xy'][c][None],axis=-1)
        for origins in pack['origin_hours']:
            index=np.searchsorted(hours,origins);observed=np.isfinite(values[index[:,None],c[None],0])
            nearest.append(np.where(observed[:,None],distance[None],np.inf).min(-1))
    cohort['nearest_current_condition_km']=(np.stack(nearest),[0,50,100,200,np.inf])
    cohort['height_m']=(geo['height'][None,None],[float('-inf'),500,1500,np.inf])
    cohort['roughness_m']=(geo['static'][None,None,:,5]*500,[0,100,300,np.inf])
    cohort['coast_km']=(geo['coast'][None,None],[0,50,150,np.inf])
    # JSON uses explicit open-ended bounds below.
    cohort['height_m']=(geo['height'][None,None],[-1000,500,1500,np.inf])
    for source,prefix in ((run,'epsilon'),(run/'velocity_control','velocity')):
        for folder in sorted(source.iterdir()):
            if not folder.is_dir() or not (folder/'results.json').exists():continue
            contract=json_read(folder/'contract.json');result=json_read(folder/'results.json')
            eligibility=json_read(folder/'eligibility.json')
            for family,item in result.items():
                path=folder/f'{family}_evaluation.npz';pack=dict(np.load(path));s=pack['samples'];y=pack['truth'];K,E,T,N=s.shape
                present=np.isfinite(y).any((0,1));n=int(present.sum())
                actual=evaluate(s[...,present].reshape(K,E*T,n),y[...,present].reshape(E*T,n))
                discrepancy=max(abs(actual['station_macro_rmse_mm']-item['metrics']['macro_rmse_mm']),
                    abs(actual['fair_crps_mm']-item['metrics']['fair_crps_station_macro_mm']))
                assert discrepancy<.003,(path,discrepancy)
                checks.append(dict(file=str(path.relative_to(run)),rmse_crps_max_difference_mm=discrepancy))
                manifest[str(path.relative_to(run))]=sha(path)
                records.append(dict(arm=folder.name,prediction=prefix,family=family,task='F' if contract['forecast'] else 'R',
                    eligibility=eligibility['passed'],parameters=item['parameters'],steps=item['steps'],**item['metrics']))
                key=f'{prefix}_{folder.name}_{family}'
                if not contract['forecast']:
                    rng=np.random.default_rng(2525);shuffled=s.copy()
                    for i in range(E):
                        for t in range(T):shuffled[:,i,t]=s[rng.permutation(K),i,t]
                    analyses[key]=dict(strata=strata(s,y,cohort),pairs=pair_scores(s,y,geo['xy']),
                        increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(s,axis=2),np.diff(y,axis=1)))),
                        shuffled_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(shuffled,axis=2),np.diff(y,axis=1)))),
                        block_boundary='independent6h starts; cross-block increments excluded',
                        season='2019January08 only; no summer evaluation in this engineering pilot')
            for f in folder.glob('*.json'):manifest[str(f.relative_to(run))]=sha(f)
    write_json(run/'comparison.json',dict(records=records,base_oof_inner2019=foundation_base,base_oof_training2016=foundation_train,
        core_training=dict(foundation_fits=11,foundation_steps_each=60,representation_steps=100,probability_fits=18,
            probability_steps_each=100,velocity_fits=8,velocity_steps_each=100),
        evaluation='2019Jan8,24h,5outer folds,one fixed layout each; R and F separate cohorts',
        training_fragment='2016Jan8–12 continuous120h per fold; not full2016–2018',
        diagnostic=True,formal_3000_step_claim=False))
    write_json(run/'stratified_and_joint_metrics.json',analyses)
    write_json(run/'verification.json',dict(saved_prediction_checks=checks,count=len(checks),
        all_predictions_finite=True,maximum_metric_difference_mm=max(x['rmse_crps_max_difference_mm'] for x in checks),
        formal_gates_passed=False,confirmation_C_D_read=False))
    write_json(run/'local_array_manifest.json',manifest)
    # Plot actual traces; batches vary with time and cannot be compared as a
    # fixed validation learning curve.
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    foundation=json_read(run/'foundation_summary.json')
    for row in foundation:
        fit=json_read(run/'foundation'/row['kind']/'fit.json');trace=fit['trace']
        ax[0].plot([x['step'] for x in trace],[x['batch_rmse_mm'] for x in trace],label=row['kind'])
    ax[0].set(xlabel='Updates',ylabel='Changing training-block RMSE (mm)',title='Real continuous-time short training');ax[0].legend(ncol=3)
    ax[1].bar([r['kind'] for r in foundation],[r['macro_rmse_mm'] for r in foundation],color='#42729b')
    ax[1].set(ylabel='Station macro RMSE (mm)',title='2019 January08: 24h internal monitor')
    for a in ax:a.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(run/'foundation_learning.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,4.5));arms=['L0','L1','A0','A1','A2','B0','B1','J0','J1'];x=np.arange(len(arms))
    for a,key,title in zip(ax,['macro_rmse_mm','fair_crps_station_macro_mm'],['RMSE','fair CRPS']):
        for j,(pred,family,color,label) in enumerate([('epsilon','gaussian','#5f8c79','Gaussian100'),('epsilon','diffusion','#b77456','epsilon100'),('velocity','diffusion','#4b79a7','velocity100')]):
            vals=[next((r[key] for r in records if r['arm']==arm and r['prediction']==pred and r['family']==family),np.nan) for arm in arms]
            a.bar(x+(j-1)*.25,vals,.25,color=color,label=label)
        a.axvline(6.5,color='#666',ls='--');a.set(xticks=x,xticklabels=arms,yscale='log',ylabel=f'{title} (mm; log scale)',title='Engineering pilot; R | F cohorts separate');a.grid(axis='y',alpha=.2);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(run/'probability_comparison.png',dpi=160);plt.close(fig)
    audit=json_read(run/'representation_support_audit.json')['results'];chosen=[r for r in audit if r['ridge']==.0001]
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(chosen))
    for j,(key,label,color) in enumerate([('train_reconstruction_mm','Available-label training reconstruction','#42729b'),('inner2019_reconstruction_mm','Available-label inner reconstruction','#5f8c79'),('spatial_group_masked_mm','Held-out physical-group error','#b77456')]):
        ax.bar(x+(j-1)*.25,[r[key] for r in chosen],.25,label=label,color=color)
    ax.axhline(1,color='#555',ls='--',label='1 mm self-reconstruction target');ax.set(xticks=x,xticklabels=[r['support'] for r in chosen],yscale='log',ylabel='RMSE (mm)',title='Self-reconstruction and spatial extrapolation are distinct');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(run/'representation_audit.png',dpi=160);plt.close(fig)
    lines=['# 时空融合与谱残差：真实本机工程验证','',
        '完成T0–T5各60步、五折T5各60步、表示100步、九组概率模型及其Gaussian各100步，另完成四组v预测及Gaussian各100步。GPU为本机RTX4060，Python为Anaconda pytorch。此轮验证全流程与数值问题；未执行新架构3000步双种子正式比较。','',
        '训练基础场按2016–2018连续小时顺序；60步实际只消费最初360目标小时。残差缓存使用2016年1月8日起连续120小时，2019年1月8日起24小时诊断；5折的目标站集合不同，不合并成一个场。2020标签未参与拟合、选择或评分。','',
        '## 1. 表示瓶颈与改进','',
        '规则256锚点原配置的训练/2019自重构为6.854/8.615mm，遮标签误差17.249mm。仅降低正则到1e-6，自重构仍为5.344/6.916mm，说明误差并非全部来自正则。几何最远点256锚点在1e-4正则下为0.609/0.677mm，但按物理组留出为20.209mm；410站点锚点的自重构降至0.009/0.014mm，物理组留出仍为22.840mm。改进支撑几何能保留已观测残差，空间生成能力仍需另行学习。该支持审计是小试后的诊断，不回改A组成绩或作为2020选模。','',
        '![表示审计](representation_audit.png)','',
        '## 2. 九组概率分支与数值稳定性','',
        '| 实验 | 任务 | epsilon100 RMSE | Gaussian100 RMSE | epsilon100 fair CRPS | v100 RMSE |',
        '|---|---|---:|---:|---:|---:|']
    for arm in arms:
        er=next(r for r in records if r['arm']==arm and r['prediction']=='epsilon' and r['family']=='diffusion')
        gr=next(r for r in records if r['arm']==arm and r['prediction']=='epsilon' and r['family']=='gaussian')
        vr=next((r for r in records if r['arm']==arm and r['prediction']=='velocity' and r['family']=='diffusion'),None)
        lines.append(f"| {arm} | {er['task']} | {er['macro_rmse_mm']:.3f} | {gr['macro_rmse_mm']:.3f} | {er['fair_crps_station_macro_mm']:.3f} | {vr['macro_rmse_mm']:.3f} |" if vr else f"| {arm} | {er['task']} | {er['macro_rmse_mm']:.3f} | {gr['macro_rmse_mm']:.3f} | {er['fair_crps_station_macro_mm']:.3f} | — |")
    lines += ['', '全部为mm。R为当前场重建，F为起点持久性均值上的未来6小时预测；两任务人口与信息条件不同。当前表示均未满足相应正式门槛，概率训练使用显式diagnostic覆盖，不能据此宣布新模型优于原p14。L1与L0的容量及训练状态也应在正式重复中匹配。', '',
        'epsilon预测在高噪声端通过`x0=(xk−sqrt(1−a)*eps_hat)/sqrt(a)`重建，100步短训下误差会被很小的sqrt(a)放大；标准化本身不保证采样稳定。v对照使用`v=sqrt(a)*eps−sqrt(1−a)*x0`及无除小量的旋转逆变换。它检验数值参数化，不代表100步已经收敛或概率分布校准。', '',
        '![概率短训比较](probability_comparison.png)', '', '## 3. 实现与验证', '',
        '- 修正时间16×16与p14大支撑的投影坐标对齐；首轮错误目录保留，v2已重训。',
        '- LS按实际ZWD覆盖率掩码缺测，保存幅度、正余弦相位和覆盖率。',
        '- 每层去噪token都对4×4空间记忆cross-attention；两域共用状态和噪声。',
        '- 五折划分、目标来源小时、均值来源小时、文件哈希和诊断模式在入口校验。',
        '- 综合收集测试时定位到旧test_fixed_support向sys.path插入子目录、遮蔽标准库profile的错误；改成包相对导入。第一次合并运行88项失败的原始日志保留，修正后另跑完整测试。', '',
        f"保存预测独立复算共{len(checks)}组，RMSE/CRPS最大差{max(x['rmse_crps_max_difference_mm'] for x in checks):.6g}mm。训练曲线、分层与站对误差见相应JSON；小时增量只在连续6小时块内计算，跨块首态重置。当前片段仅1月，夏季误差和全年结果尚无此轮证据。", '',
        '![基础场实际训练曲线](foundation_learning.png)', '', '## 4. 下一步实验调整', '',
        '1. 基础场保持p14同起点、同数据条件比较，按3000步和12小时连续批覆盖训练年份，再完成双种子与季节诊断。',
        '2. 使用几何适应锚点或站点锚点保留可观测残差；将自重构与物理组外推分开验收，引入局地残差消息与支持度条件，保持固定查询算子。',
        '3. v预测作为稳定性候选，保留epsilon控制；在相同表示、步数、种子和Gaussian控制下比较CRPS、覆盖、增量及站对分布，避免仅看RMSE。',
        '4. 增加连续时间缓存分片和季节覆盖，再做完整2016–2018残差训练、2019选择、锁定后2016–2019重拟合及2020开发评价。',
        '5. J组补充经过训练的起点未来均值预测头；当前J仅是origin-persistence控制，尚无该头的效果。', '',
        '[实现与全部架构图](../../../../../../../docs/research_notes/2026-11-25-joint-spectral-implementation.md)']
    (run/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps(dict(model_records=len(records),verified_prediction_arrays=len(checks),plots=3,report=str(run/'REPORT.md'))))


if __name__=='__main__':main()
