"""Verify paired populations, compute block uncertainty and terrain diagnostics."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore_sequence_mechanisms import ROOT,SOURCE,grid_features,GridRandom,MultiScaleRandom
import sequence_mechanism_spike as core
from gnss_field.artifacts import sha256,write_json


def components(p):
    y=p['target'];m=p['mask'];out={'count':m.sum((1,2))}
    if 'samples' in p:
        s=p['samples'].astype('f8');k=s.shape[1]
        def crps(x,y):
            coeff=(2*np.arange(k)-k+1)[None,:,None,None]
            return abs(x-y[:,None]).mean(1)-(np.sort(x,axis=1)*coeff).sum(1)/(k*(k-1))
        score=crps(s,y)
        dm=m[:,1:]&m[:,:-1]
        delta=crps(np.diff(s,axis=2),np.diff(y,axis=1))
        out.update(crps_sum=(score*m).sum((1,2)),increment_sum=(delta*dm).sum((1,2)),increment_count=dm.sum((1,2)))
        pred=s.mean(1)
    else:pred=p['prediction']
    out.update(sse=(((pred-y)**2)*m).sum(1),station_count=m.sum(1))
    return out


def paired(control,intervention,probability):
    for key in ('target','mask','station_ids','hours'):assert np.array_equal(control[key],intervention[key])
    c,i=components(control),components(intervention)
    rng=np.random.default_rng(20261123)
    weights=np.stack([np.bincount(rng.integers(0,12,12),minlength=12) for _ in range(3000)])
    result={}
    if probability:
        for metric,num,den in [('crps','crps_sum','count'),('increment_crps','increment_sum','increment_count')]:
            delta=weights@(i[num]-c[num])/(weights@c[den])
            result[metric]=dict(difference=float((i[num]-c[num]).sum()/c[den].sum()),ci95=np.quantile(delta,[.025,.975]).tolist())
    else:
        def rmse(o,w):
            counts=w@o['station_count'];sse=w@o['sse'];active=counts>0
            return (np.sqrt(sse/np.maximum(counts,1))*active).sum(-1)/active.sum(-1)
        delta=rmse(i,weights)-rmse(c,weights)
        result['macro_rmse']=dict(difference=float(rmse(i,np.ones((1,12)))[0]-rmse(c,np.ones((1,12)))[0]),ci95=np.quantile(delta,[.025,.975]).tolist())
    return result


def load(arm,seed,budget,pop):return dict(np.load(ROOT/f'{arm}_{seed}_{budget}_{pop}.npz'))


def main():
    torch.set_num_threads(2)
    r=json.loads((ROOT/'results.json').read_text());lock=json.loads((ROOT/'experiment.lock.json').read_text())
    assert sha256(ROOT/'data.npz')==r['runtime']['data_sha256']
    assert sha256(ROOT/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha256(core.PROJECT/'tools/explore_sequence_mechanisms.py')==lock['code_sha256']==r['runtime']['code_sha256']
    for name,digest in lock['dependencies'].items():assert sha256(core.PROJECT/'tools'/name)==digest
    for name,digest in r['runtime']['checkpoints'].items():assert sha256(ROOT/name)==digest
    a=dict(np.load(ROOT/'data.npz'));mini=dict(a)
    for suffix in ('raw','hours','y'):mini['inner_'+suffix]=a['inner_'+suffix][:1]
    plain=grid_features(mini,'inner',False);height=grid_features(mini,'inner',True)
    assert torch.equal(plain['base'],height['base']) and torch.equal(plain['mask'],height['mask'])
    assert torch.equal(plain['x'][...,15:],height['x'][...,15:])
    assert not torch.equal(plain['x'][...,:15],height['x'][...,:15])
    del plain,height
    scalar=GridRandom(True);vector=MultiScaleRandom();vector.head.load_state_dict(scalar.head.state_dict())
    scalar.rho_raw.data.fill_(1.);vector.rho_raw.data.fill_(1.)
    torch.manual_seed(91);h=torch.randn(2,30,5,4,33);h[...,-1]=.25;noise=torch.randn(2,8,30,8)
    with torch.no_grad():
        assert torch.allclose(scalar(h,noise),vector(h,noise),atol=1e-6)
        changed=noise.clone();changed[:,:,15:]+=5
        assert torch.equal(vector(h,noise)[:,:,:15],vector(h,changed)[:,:,:15])
    diagnostics={};terrain={};old=json.loads((SOURCE/'results.json').read_text())
    for seed in ('42','17'):
        assert abs(r['distance_'+seed]['outer']['500']['macro_rmse']-old[seed]['frame']['outer']['macro_rmse'])<1e-6
        diagnostics[seed]={}
        for pop in ('outer','confirmation'):
            diagnostics[seed][pop]={}
            for name,ca,cb,ia,ib,prob in [('more_mean_training','distance',500,'distance',2000,False),
                    ('height_encoder','distance',2000,'height',2000,False),
                    ('more_random_training','single',350,'single',1400,True),
                    ('multiple_timescales','single',1400,'multi',1400,True)]:
                c=load(ca,seed,cb,pop);i=load(ia,seed,ib,pop)
                assert np.isfinite(i['samples'] if prob else i['prediction']).all()
                diagnostics[seed][pop][name]=paired(c,i,prob)
                if prob:
                    for arm,budget,p in [(ca,cb,c),(ia,ib,i)]:
                        comp=components(p)
                        assert abs(comp['crps_sum'].sum()/comp['count'].sum()-r[f'{arm}_{seed}'][pop][str(budget)]['k64']['crps'])<2e-5
            terrain[seed]={}
            if pop=='confirmation':
                for arm,budget in [('distance',500),('distance',2000),('height',2000)]:
                    p=load(arm,seed,budget,pop);hs=a['heights'][p['station_ids']]
                    for label,chosen in [('below500',hs<500),('500to1500',(hs>=500)&(hs<1500)),('above1500',hs>=1500)]:
                        if not p['mask'][...,chosen].any():continue
                        n=int(chosen.sum())
                        terrain[seed].setdefault(f'{arm}_{budget}',{})[label]=core.metrics(p['prediction'][...,chosen].reshape(-1,n),p['target'][...,chosen].reshape(-1,n),p['mask'][...,chosen].reshape(-1,n))
    checks=dict(source_protocol_data_checkpoint_hashes=True,matched_populations=True,finite_predictions=True,
        height_arm_same_baseline_support_static_features=True,scalar_vector_equivalence=True,
        vector_latent_causality=True,previous_500_step_replication=True,stored_crps_reproduced=True)
    write_json(ROOT/'verification.json',dict(checks=checks,paired=diagnostics,terrain=terrain,
        bootstrap='3000 matched resamples of 12 monthly blocks; pooled metric recomputed; exploratory multiple contrasts',code_sha256=sha256(__file__)))
    lines=['# 进一步训练与机制探索：实测结果','',
        '本轮比较训练预算、地形条件聚合和多时间尺度随机隐状态。所有对照为相同允许输入Z0/T/P、统一格网算子、两种子本机训练。仍沿用历史PWV筛选队列，C/D标签未读取。',
        '', '确认集为2018每月22日的24小时片段，另加6小时热身，沿用B外层站组；不用于本轮训练或选择，但不保证项目历史从未暴露。以下指标均为该确认集。',
        '', '|种子|均值编码/预算|站宏RMSE/mm|有效站|站时|', '|---|---|---:|---:|---:|']
    for seed in ('42','17'):
        for arm,budget in [('distance','500'),('distance','2000'),('height','2000')]:
            v=r[f'{arm}_{seed}']['confirmation'][budget]
            lines.append(f"|{seed}|{arm}/{budget}|{v['macro_rmse']:.4f}|{v['stations']}|{v['station_hours']}|")
    lines+=['','|种子|随机机制/预算|CRPS/mm|1h增量CRPS/mm|90%覆盖|宽度/mm|','|---|---|---:|---:|---:|---:|']
    for seed in ('42','17'):
        for arm,budget in [('single','350'),('single','1400'),('multi','1400')]:
            v=r[f'{arm}_{seed}']['confirmation'][budget]['k64']
            lines.append(f"|{seed}|{arm}/{budget}|{v['crps']:.4f}|{v['increment_crps']:.4f}|{v['coverage90']:.2%}|{v['width90']:.3f}|")
    lines+=['','## 配对判断','', '差值为干预减对照；负值表示改善。区间按12个月份天气块同步重采样，重算与点估计相同的指标；多对照、短片段探索不作最终确认性显著结论。']
    verdict={}
    for name in diagnostics['42']['confirmation']:
        effects=[diagnostics[s]['confirmation'][name] for s in ('42','17')]
        if 'macro_rmse' in effects[0]:passed=all(e['macro_rmse']['difference']<0 for e in effects)
        else:
            passed=all(e['crps']['difference']<0 and e['increment_crps']['difference']<0 for e in effects)
        verdict[name]=dict(k64_both_seeds_all_metrics_improve=bool(passed),effects=effects,
            scope='K64 only; not a cross-noise robustness verdict')
        lines.append(f'\n- {name}：K64下两种子全部主要指标同向改善={passed}。')
        for seed,e in zip(('42','17'),effects):
            for metric,v in e.items():lines.append(f"  - 种子{seed} {metric}差{v['difference']:+.4f}，95%探索区间[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] mm。")
    lines+=['', '独立噪声K128复核：延长随机分支训练的边缘/增量CRPS在两个种子均同向改善。多时间尺度相对单尺度，种子42边缘CRPS由5.7018变为5.7121 mm，改善方向反转；种子17仍改善。故多时间尺度只保留为采样敏感的小幅候选，暂不替换单尺度。',
        '', '## 解释与后续','',
        '预算对照是在同一训练轨迹上保留各预算内的inner优选；多训练结果不等于选最后一个检查点。随机生成对照均共享distance/2000冻结均值，不能把有限集合均值RMSE变化当作确定性均值改善。',
        '', '高差机制只改变编码器的观测聚合，未改变背景、评分人口或网络参数量。固定750m尺度只代表这个候选，不否定地形建模整体。多时间尺度由1个rho增加至8个，仅增加7个参数，不代表新的完整Diffusion/GAN/VAE。',
        '', '地形分层有效站、站时及RMSE保留在verification.json；高海拔分组若样本少，不能作整个山地代表。独立噪声K128与全部学习曲线见results.json。外层和确认集方向不一致时保留结果，不靠改评价集挑胜者。',
        '', '下一阶段先完善允许变量原始质控，再考虑空间分块与站点遮蔽自监督、观测创新校正和局地随机场表示；这些机制尚未由本轮验证。',
        '', '方法背景：[地形条件convCNP](https://gmd.copernicus.org/articles/15/251/2022/)支持检验地形条件的研究动机；[依赖预测神经过程](https://arxiv.org/abs/2203.08775)支持联合分布建模的动机。具体750m核和逐潜维AR是本项目待检验假设，不是论文已证明对ZWD有效的结论。',
        '', f"成功训练与评价耗时{r['runtime']['seconds']:.1f}秒，设备{r['runtime']['device']}。内置浏览器自动化仍初始化失败，网页模型复核未完成。"]
    (ROOT/'RESULTS.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    write_json(ROOT/'verdict.json',verdict)
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for col,seed in enumerate(('42','17')):
        for arm in ('distance','height'):
            hist=r[f'{arm}_{seed}']['history'];axs[0,col].plot([v['step'] for v in hist],[v['inner']['macro_rmse'] for v in hist],label=arm)
        for arm in ('single','multi'):
            hist=r[f'{arm}_{seed}']['history'];axs[1,col].plot([v['step'] for v in hist],[v['inner']['crps']+.5*v['inner']['increment_crps'] for v in hist],label=arm)
        axs[0,col].set(title=f'Seed {seed}: mean',ylabel='Inner station macro RMSE (mm)')
        axs[1,col].set(title=f'Seed {seed}: random field',ylabel='Inner selection score (mm)',xlabel='Training updates')
        for row in (0,1):axs[row,col].legend();axs[row,col].grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'learning_curves.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(checks=checks,verdict=verdict)),flush=True)


if __name__=='__main__':main()
