"""Recompute saved predictions and summarize paired development evidence."""
import json
import numpy as np
import masked_graph_spike as m
from gnss_field.artifacts import write_json,sha256


def macro(p,y,mask):
    count=mask.sum((0,1));valid=count>0
    return float(np.sqrt((((p-y)**2*mask).sum((0,1))/np.maximum(count,1))[valid]).mean())


def main():
    result=json.loads((m.ROOT/'results.json').read_text());old=json.loads((m.old.inv.ROOT/'results.json').read_text())
    report={};max_error=0.;rng=np.random.default_rng(619);boot=rng.integers(12,size=(1000,12))
    lines=['# 动态遮站图生成场：配对小试','',
       '本轮完成从头图编码/观测融合、动态遮站预训练及图条件时序潜变量生成。按预锁协议完成后进行开发集合评价。','',
       '结论：遮站预训练相对直接监督有局部开发集收益，但整体仍比v04高约1.24/1.27mm，保留v04，不直接扩大本配置训练。当前固定人口没有达到8mm。时间相关潜变量在本配置的增量评分中有收益。','',
       '## 固定评价结果','',
       '2018每月27日12段24小时；71有效站、20108站时。宏RMSE单位mm，均值为解析条件均值，不使用有限成员平均代替。','',
       '| 方法 | seed42 | seed17 |','|---|---:|---:|',
       f"| 原v04历史创新 | {old['results']['42']['reserved']['history']['point']['macro_rmse']:.4f} | {old['results']['17']['reserved']['history']['point']['macro_rmse']:.4f} |"]
    for arm in ('direct','masked'):
        vals=[result['results'][f'{arm}_{seed}_ar_reserved']['point']['macro_rmse'] for seed in (42,17)]
        lines.append(f"| {'直接监督图模型' if arm=='direct' else '遮站预训练图模型'} | {vals[0]:.4f} | {vals[1]:.4f} |")
    for name in ('confirmation','reserved'):
        for seed in (42,17):
            baseline=np.load(m.old.inv.ROOT/f'history_{seed}_{name}.npz')
            pred={}
            for arm,kind in [('direct','ar'),('masked','ar'),('masked','ind')]:
                key=f'{arm}_{seed}_{kind}_{name}';d=np.load(m.ROOT/f'{key}.npz');pred[arm+'_'+kind]=d
                for field in ('mask','target','station_ids','hours'):assert np.array_equal(d[field],baseline[field])
                p=d['prediction'];y=d['target'];mask=d['mask'];assert np.isfinite(p).all() and np.isfinite(d['samples']).all()
                point=m.old.inv.point_score(p,y,mask);prob=m.old.inv.probability_score(d['samples'],y,mask)
                for k,v in point.items():max_error=max(max_error,abs(v-result['results'][key]['point'][k]))
                for k,v in prob.items():max_error=max(max_error,abs(v-result['results'][key]['k64'][k]))
            assert max_error<1e-5
            masked=pred['masked_ar'];y=masked['target'];mask=masked['mask'];p=masked['prediction']
            comparisons={}
            for label,other in [('direct',pred['direct_ar']['prediction']),('v04',baseline['prediction'])]:
                diffs=[macro(p[idx],y[idx],mask[idx])-macro(other[idx],y[idx],mask[idx]) for idx in boot]
                comparisons[label]=dict(delta_macro_rmse=macro(p,y,mask)-macro(other,y,mask),weather_block_ci95=np.quantile(diffs,[.025,.975]).tolist())
            # Each hour keeps its exact ensemble; only member trajectories change.
            s=masked['samples'].copy();sr=np.random.default_rng(771)
            for b in range(len(s)):
                for t in range(s.shape[2]):s[b,:,t]=s[b,sr.permutation(s.shape[1]),t]
            shuffled=m.old.inv.probability_score(s,y,mask)
            report[f'{seed}_{name}']=dict(comparisons=comparisons,time_member_shuffle=shuffled)
    assert max_error<1e-5
    lines+=['','## 机制比较','',
        '| 遮站模型的生成方式（月27日） | seed42 CRPS | seed42 增量CRPS | seed17 CRPS | seed17 增量CRPS |',
        '|---|---:|---:|---:|---:|']
    for kind in ('ind','ar'):
        v=[result['results'][f'masked_{seed}_{kind}_reserved']['k128'] for seed in (42,17)]
        lines.append(f"| {'独立潜变量' if kind=='ind' else '时间相关潜变量'}，K128 | {v[0]['crps']:.4f} | {v[0]['increment_crps']:.4f} | {v[1]['crps']:.4f} | {v[1]['increment_crps']:.4f} |")
    lines+=['','两种子均按B-inner选点，外层未参与调参。逐时成员重排保持单时刻集合不变，打破轨迹对应；评分和12天气块配对bootstrap区间见audit.json。区间描述当前开发日历，不作为最终泛化显著性。','',
        'seed42在月22日的遮站收益区间跨零，不能称跨日历稳定显著。时间成员重排诊断采用K64，表中独立/相关对照采用独立噪声K128。','',
        '## 实现与证据范围','',
        f"本机 {result['runtime']['device']}，训练和评价耗时 {result['runtime']['seconds']:.1f} 秒。两臂各1600次均值更新，四次共6400次；六个生成头各400次，共2400次。",
        'direct前400步使用32个B-fit目标，masked前400步使用44个隐藏A目标；随后1200步B-fit批次相同。比较匹配更新次数，不宣称相同标签预算或严格同算力。',
        '隐藏站在完整30小时窗口中从条件、插值和统计中移除；预检隐藏值/未来值扰动均为零。统一格网均值和共享随机序列回查结果见verification.json，完整8成员×24小时×144×96样例保留本地。',
        '生成头使用神经系数、共享空间潜变量及AR时间演化，随机残差解析均值为零；属于生成网络机制原型，尚非VAE或Diffusion。新增模块当前不含距海岸特征。',
        '月22/27日属于已暴露开发日历，C/D未读取。输入值仅Z0/T/P，历史队列仍含PWV筛选，正式协议须重建允许变量质控。','',
        '## 下一步判断','',
        '保留遮站隔离训练及共享时序潜变量两个机制。下一候选应保留原模型的多尺度聚合和当前/历史创新锚定，再检验图编码是否增加信息。此次新图骨干同时改变了多尺度背景和读出，不足以把相对v04的差距归因于自监督或图网络本身。该后续组合尚未实测。','']
    write_json(m.ROOT/'audit.json',dict(saved_metric_max_error=max_error,paired=report,code_sha256=sha256(__file__)))
    (m.ROOT/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(saved_metric_max_error=max_error,paired=report)))


if __name__=='__main__':main()
