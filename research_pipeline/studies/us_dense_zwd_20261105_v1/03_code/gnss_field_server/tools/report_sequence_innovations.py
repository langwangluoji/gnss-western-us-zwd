"""Paired checks of the frozen-field innovation adaptation."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sequence_innovation_spike import ROOT,SOURCE,correction_features,point_score,probability_score
from report_sequence_exploration import paired
import sequence_mechanism_spike as core
from gnss_field.artifacts import sha256,write_json


def main():
    torch.set_num_threads(2)
    record=json.loads((ROOT/'results.json').read_text());r=record['results'];lock=json.loads((ROOT/'experiment.lock.json').read_text())
    assert sha256(ROOT/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha256(core.PROJECT/'tools/sequence_innovation_spike.py')==lock['code_sha256']==record['runtime']['code_sha256']
    assert sha256(ROOT/'selection.lock.json')==record['runtime']['selection_sha256']
    for name,digest in lock['dependency_sha256'].items():assert sha256(core.PROJECT/'tools'/name)==digest
    for seed in (42,17):
        for name in (f'distance_{seed}_2000.pt',f'single_{seed}_1400.pt'):
            assert sha256(SOURCE/name)==lock['source_models'][name]
    a=dict(np.load(SOURCE/'data.npz'));assert sha256(SOURCE/'data.npz')==lock['source_data']
    # Future A perturbations cannot change earlier current/lagged innovation features.
    mini=dict(a);mini['confirmation_raw']=a['confirmation_raw'][:1].copy()
    mu=np.zeros((1,30,len(a['heights'])),dtype='f4')
    f=correction_features(mini,'confirmation',mu)
    mini['confirmation_raw'][:,20:,:,0]+=20
    changed=correction_features(mini,'confirmation',mu)
    assert np.array_equal(f[:,:14],changed[:,:14])
    diagnostics={};terrain={};max_identity=0.
    for seed in ('42','17'):
        diagnostics[seed]={};terrain[seed]={}
        for pop in ('confirmation','reserved'):
            predictions={arm:dict(np.load(ROOT/f'{arm}_{seed}_{pop}.npz')) for arm in ('none','bias','current','history')}
            base=predictions['none'];diagnostics[seed][pop]={}
            for arm,p in predictions.items():
                for key in ('target','mask','station_ids','hours'):assert np.array_equal(p[key],base[key])
                assert np.isfinite(p['samples']).all() and np.isfinite(p['prediction']).all()
                point=point_score(p['prediction'],p['target'],p['mask'])
                probability=probability_score(p['samples'],p['target'],p['mask'])
                assert abs(point['macro_rmse']-r[seed][pop][arm]['point']['macro_rmse'])<1e-7
                assert abs(probability['crps']-r[seed][pop][arm]['k64']['crps'])<1e-7
                error=float(abs((p['samples']-base['samples'])-(p['prediction']-base['prediction'])[:,None]).max())
                max_identity=max(max_identity,error);assert error<1e-5
            for control,intervention in [('none','bias'),('none','current'),('none','history'),('bias','history'),('current','history')]:
                c,i=predictions[control],predictions[intervention]
                cmean={k:v for k,v in c.items() if k!='samples'};imean={k:v for k,v in i.items() if k!='samples'}
                diagnostics[seed][pop][f'{intervention}_minus_{control}']=dict(
                    point=paired(cmean,imean,False),probability=paired(c,i,True))
            if pop=='reserved':
                hs=a['heights'][base['station_ids']]
                for arm,p in predictions.items():
                    terrain[seed][arm]={}
                    for label,chosen in [('below500',hs<500),('500to1500',(hs>=500)&(hs<1500)),('above1500',hs>=1500)]:
                        if p['mask'][...,chosen].any():terrain[seed][arm][label]=point_score(p['prediction'][...,chosen],p['target'][...,chosen],p['mask'][...,chosen])
    checks=dict(source_protocol_selection_checkpoint_hashes=True,matched_populations=True,finite_predictions=True,
        stored_metrics_reproduced=True,causal_innovation_lags=True,unchanged_random_spread_max_identity_error=max_identity)
    write_json(ROOT/'verification.json',dict(checks=checks,paired=diagnostics,terrain=terrain,
        code_sha256=sha256(__file__),bootstrap='3000 matched resamples of12 weather blocks; adaptive exploratory comparisons'))
    lines=['# 观测创新与历史条件：连续生成场小试','',
        '在v03结果之后追加的适应性探索。冻结distance/2000均值与single/1400随机场，仅用A站Z0观测创新拟合小型线性校正。B-fit估计系数，B-inner选择正则化，两个种子的系数锁定后才评分。',
        '', '主要结果为2018每月27日24小时片段、71有效B外层站、20108站时。月22日回看结果另保留。二者均不是整个项目的最终盲测；C/D标签未读取。',
        '', '|种子|校正|站宏RMSE/mm|CRPS/mm|1h增量CRPS/mm|','|---|---|---:|---:|---:|']
    for seed in ('42','17'):
        for arm in ('none','bias','current','history'):
            v=r[seed]['reserved'][arm]
            lines.append(f"|{seed}|{arm}|{v['point']['macro_rmse']:.4f}|{v['k64']['crps']:.4f}|{v['k64']['increment_crps']:.4f}|")
    lines+=['','## 机制判断','',
        'none为未校正；bias为公共高度Z0常数偏移，经高度算子转换后不是地面ZWD统一常数；current使用当前A创新的六个几何核；history进一步加入1h和3h历史，总18分量。']
    for seed in ('42','17'):
        none=r[seed]['reserved']['none'];hist=r[seed]['reserved']['history']
        lines.append(f"\n- 种子{seed}：历史创新校正相对未校正的站宏RMSE改善{100*(1-hist['point']['macro_rmse']/none['point']['macro_rmse']):.2f}%，CRPS改善{100*(1-hist['k64']['crps']/none['k64']['crps']):.2f}%。")
        for comparator in ('none','bias','current'):
            d=diagnostics[seed]['reserved'][f'history_minus_{comparator}']
            for metric,v in [('macro_rmse',d['point']['macro_rmse']),('increment_crps',d['probability']['increment_crps'])]:
                lines.append(f"  - 相对{comparator}：{metric}差{v['difference']:+.4f}，95%探索区间[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] mm。")
    lines+=['', '当前创新改善点值，但两个种子的小时增量评分均变差；加入历史创新后，该项评分恢复并优于未校正。月22日回看的点值与概率方向一致；独立噪声K128复核的概率指标也同向，确定性均值本身不随集合大小变化。相同随机样本只作确定性平移，逐时区间宽度保持不变，收益不是靠放宽区间取得。',
        '', '这支持“输入观测约束＋因果历史校正”作为下一阶段候选。核函数和ridge本身是既有方法，本项目旧输入条件下也探索过创新校正；新的证据针对禁用水汽柱、统一格网和连续随机场的组合，不据此声称已形成论文创新或已恢复全部细尺度物理过程。',
        '', '输入Z0/T/P仍来自历史PWV完整性筛选队列，正式研究须从原始文件重建允许变量质控。海岸距离、自监督编码、连续大空间块留出与长序列状态衔接尚未验证。',
        '', '地形三组的点值结果和全部配对区间见verification.json。不要将高程分组等同于完整地形类型分类。',
        '', f"本机拟合与评价耗时{record['runtime']['seconds']:.1f}秒。此轮无需增加深网络训练；系数、选择记录、源码/权重哈希与预测已保存。"]
    (ROOT/'RESULTS.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    arms=['none','bias','current','history']
    for seed in ('42','17'):
        axs[0].plot(arms,[r[seed]['reserved'][a]['point']['macro_rmse'] for a in arms],'o-',label=f'Seed {seed}')
        axs[1].plot(arms,[r[seed]['reserved'][a]['k64']['increment_crps'] for a in arms],'o-',label=f'Seed {seed}')
    axs[0].set(ylabel='Station macro RMSE (mm)',title='Point reconstruction')
    axs[1].set(ylabel='1h increment CRPS (mm)',title='Temporal change distribution')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Reserved monthly day-27 sequences; matched observations and samples')
    fig.tight_layout();fig.savefig(ROOT/'mechanism_comparison.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(checks=checks,paired=diagnostics)),flush=True)


if __name__=='__main__':main()
