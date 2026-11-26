"""Observation-space diffusion factorial results and geographic diagnostics."""
import json
import numpy as np
from scipy.spatial.distance import cdist
import diffusion_observation_train as e
from gnss_field.artifacts import write_json,sha256


def main():
    r=json.loads((e.ROOT/'results.json').read_text());old=json.loads((e.d.old.inv.ROOT/'results.json').read_text())
    a=np.load(e.d.old.exp.ROOT/'data.npz');ids=a['split_outer'];dist=cdist(a['xy'][ids],a['xy'][ids]);np.fill_diagonal(dist,np.inf)
    pairs=sorted({tuple(sorted((i,int(j)))) for i,j in enumerate(dist.argmin(1)) if dist[i,j]<=100000})
    left,right=np.array(pairs).T;diagnostics={};comparison={}
    lines=['# 多尺度残差与观测空间时序训练','',
       '2×2配对：24/64维表示，站点CRPS/站点CRPS+时间增量CRPS。冻结v04背景，两种子。主采样为统一6步DDIM，保留不微调起点，以区分采样器改变与训练收益。','',
       '## 固定月27日：71站、20108站时，独立K128','',
       '| 表示与训练 | RMSE42/mm | RMSE17/mm | CRPS42 | CRPS17 | 增量CRPS42 | 增量CRPS17 | 90%覆盖42/17 |',
       '|---|---:|---:|---:|---:|---:|---:|---|']
    baseline=[old['results'][str(seed)]['reserved']['history'] for seed in (42,17)]
    lines.append(f"| 原v04 | {baseline[0]['point']['macro_rmse']:.4f} | {baseline[1]['point']['macro_rmse']:.4f} | {baseline[0]['k128']['crps']:.4f} | {baseline[1]['k128']['crps']:.4f} | {baseline[0]['k128']['increment_crps']:.4f} | {baseline[1]['k128']['increment_crps']:.4f} | {baseline[0]['k128']['coverage90']:.3f}/{baseline[1]['k128']['coverage90']:.3f} |")
    labels=dict(pre='去噪预训练',marginal='观测CRPS',temporal='观测+时间CRPS')
    for rank in (24,64):
        for arm in ('pre','marginal','temporal'):
            key=f'r{rank}_{arm}';v=[r['results'][f'{key}_{seed}_reserved']['k128'] for seed in (42,17)]
            lines.append(f"| {rank}维 {labels[arm]} | {v[0]['point']['macro_rmse']:.4f} | {v[1]['point']['macro_rmse']:.4f} | {v[0]['crps']:.4f} | {v[1]['crps']:.4f} | {v[0]['increment_crps']:.4f} | {v[1]['increment_crps']:.4f} | {v[0]['coverage90']:.3f}/{v[1]['coverage90']:.3f} |")
            comparison[key]={}
            for name in ('confirmation','reserved'):
                comparison[key][name]={}
                for k in ('k64','k128'):
                    val=[r['results'][f'{key}_{seed}_{name}'][k] for seed in (42,17)]
                    comparison[key][name][k]=dict(mean_macro_rmse=float(np.mean([x['point']['macro_rmse'] for x in val])),
                        mean_crps=float(np.mean([x['crps'] for x in val])),mean_increment=float(np.mean([x['increment_crps'] for x in val])),
                        mean_score=float(np.mean([x['crps']+.5*x['increment_crps'] for x in val])))
    for name in ('confirmation','reserved'):
        for seed in (42,17):
            sources={f'r{rank}_{arm}':e.ROOT/f'r{rank}_{arm}_{seed}_{name}.npz' for rank in (24,64) for arm in ('pre','marginal','temporal')}
            sources['v04']=e.d.old.inv.ROOT/f'history_{seed}_{name}.npz'
            for arm,path in sources.items():
                p=np.load(path);s=p['samples'];y=p['target'];m=p['mask'];mean=p['prediction']
                spatial=e.d.old.inv.probability_score(s[:,:,:,left]-s[:,:,:,right],y[:,:,left]-y[:,:,right],m[:,:,left]&m[:,:,right])['crps']
                bins={}
                for label,lo,hi in [('low',-np.inf,500),('middle',500,1500),('high',1500,np.inf)]:
                    keep=(a['heights'][ids]>=lo)&(a['heights'][ids]<hi)
                    bins[label]=e.d.old.inv.point_score(mean[:,:,keep],y[:,:,keep],m[:,:,keep])
                diagnostics[f'{arm}_{seed}_{name}']=dict(geographic_pair_crps=spatial,pairs=len(pairs),
                    negative_sample_fraction=float(((s<0)*m[:,None]).sum()/(m.sum()*s.shape[1])),height_bins=bins)
    lines+=['','点RMSE使用采样均值，原v04为解析均值。K64复算及独立K128均保留。地理最近邻100km以内成对CRPS、海拔分组和负值比例见comparison.json；它们属于开发诊断。','',
       '## 表示与训练','']
    for key,diag in r['diagnostics'].items():
        fraction=1-(diag['projection_rmse']/diag['residual_rmse'])**2
        lines.append(f"- {key}：训练投影减少残差平方误差 {fraction:.1%}，投影后RMSE {diag['projection_rmse']:.4f}mm。")
    lines+=['','64维包含原24维与40个局地正交模式；扩展表示也会改变ridge系数及标准化，不能称旧24个潜系数数值保持不变。空间基只使用B-fit几何，潜系数只使用B-fit训练残差，仍非稠密真值。',
      '每个空间/种子1200步去噪预训练，随后两个同起点损失臂各200步。真实4小时K4采样链反传，初始前小时训练系数用于训练窗口启动，评价24小时从零自由滚动。这种训练/评价差异通过自由滚动与shuffle判断，不能仅凭结构声称动态合理。',
      '所有配置使用同一inner综合准则选择，包含零步。扩大表示改变参数量与算力；同rank两损失臂共享窗口/噪声/保持项采样。预训练及微调曲线、选中步数见results.json。',
      '当前是固定已暴露日历上的适应性开发，未读取C/D标签。A输入仅Z0/T/P，历史队列PWV筛选限制仍适用。',
      f"本机 {r['runtime']['device']}，训练和评价耗时 {r['runtime']['seconds']:.1f}秒。共4800预训练更新、1600观测空间更新。",'']
    (e.ROOT/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    write_json(e.ROOT/'comparison.json',dict(comparison=comparison,geography=diagnostics,code_sha256=sha256(__file__)))
    print(json.dumps(comparison))


if __name__=='__main__':main()
