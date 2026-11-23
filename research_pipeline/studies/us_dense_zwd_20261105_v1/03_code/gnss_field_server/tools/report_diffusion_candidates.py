"""Compact screening tables with independent ensemble and cost comparisons."""
import json
import numpy as np
import diffusion_candidates as d
from gnss_field.artifacts import write_json,sha256


def main():
    r=json.loads((d.ROOT/'results.json').read_text());baseline=json.loads((d.old.inv.ROOT/'results.json').read_text())
    labels=dict(gaussian='条件Gaussian',diffusion='MLP Diffusion',transformer='Diffusion Transformer',sparse='稀疏FFN Diffusion Transformer')
    lines=['# 扩散、Transformer与稀疏FFN配对筛查','',
      '固定v04背景与24维观测约束残差空间，比较四种条件生成器。每种两种子、1500更新，所有选择完成后评价外层。','',
      '## 固定月27日结果','',
      '2018年每月27日，71有效站、20108站时；下表采用独立噪声K128。点RMSE使用生成集合均值；v04为解析均值，有限成员误差需结合K64检查。','',
      '| 方法 | RMSE42/mm | RMSE17/mm | CRPS42 | CRPS17 | 1h增量CRPS42 | 1h增量CRPS17 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    b=[baseline['results'][str(s)]['reserved']['history'] for s in (42,17)]
    lines.append(f"| 原v04 | {b[0]['point']['macro_rmse']:.4f} | {b[1]['point']['macro_rmse']:.4f} | {b[0]['k128']['crps']:.4f} | {b[1]['k128']['crps']:.4f} | {b[0]['k128']['increment_crps']:.4f} | {b[1]['k128']['increment_crps']:.4f} |")
    comparison={}
    for kind in d.KINDS:
        v=[r['results'][f'{kind}_{s}_reserved']['k128'] for s in (42,17)]
        lines.append(f"| {labels[kind]} | {v[0]['point']['macro_rmse']:.4f} | {v[1]['point']['macro_rmse']:.4f} | {v[0]['crps']:.4f} | {v[1]['crps']:.4f} | {v[0]['increment_crps']:.4f} | {v[1]['increment_crps']:.4f} |")
        comparison[kind]={}
        for name in ('confirmation','reserved'):
            comparison[kind][name]={}
            for members in ('k64','k128'):
                values=[r['results'][f'{kind}_{s}_{name}'][members] for s in (42,17)]
                comparison[kind][name][members]=dict(mean_crps=float(np.mean([x['crps'] for x in values])),mean_increment_crps=float(np.mean([x['increment_crps'] for x in values])),
                    mean_macro_rmse=float(np.mean([x['point']['macro_rmse'] for x in values])),mean_selection_score=float(np.mean([x['crps']+.5*x['increment_crps'] for x in values])))
    lines+=['','## 采样、校准与代价','',
      '| 模型 | 参数量 | seed42训练/选择秒 | seed17训练/选择秒 | K128最大截断比例 | 月27日90%覆盖率42/17 |',
      '|---|---:|---:|---:|---:|---|']
    for kind in d.KINDS:
        s=[r['selection'][f'{kind}_{seed}'] for seed in (42,17)]
        v=[r['results'][f'{kind}_{seed}_reserved']['k128'] for seed in (42,17)]
        clip=max(r['results'][f'{kind}_{seed}_{name}']['k128']['clip_fraction'] for seed in (42,17) for name in ('confirmation','reserved'))
        lines.append(f"| {labels[kind]} | {s[0]['parameters']} | {s[0]['seconds']:.1f} | {s[1]['seconds']:.1f} | {clip:.6%} | {v[0]['coverage90']:.3f}/{v[1]['coverage90']:.3f} |")
    lines+=['','训练步数、批量和抽样流匹配；架构参数量、算力不同。稀疏模型为4专家top2路由，当前原型实际计算四专家，不声称加速。专家最后采样调用占比保留在results.json，不代表全数据长期负载。','',
      '## 监督含义与验证','']
    for seed,diag in r['latent_diagnostics'].items():
        fraction=1-(diag['projection_rmse']/diag['residual_rmse'])**2
        lines.append(f"- seed{seed}：B-fit训练残差RMSE {diag['residual_rmse']:.4f}mm，24维ridge投影后 {diag['projection_rmse']:.4f}mm；减少残差平方误差 {fraction:.1%}。这是训练表示诊断，不是可部署模型成绩。")
    lines+=['','ridge系数是B-fit残差的受限子空间估计，不是稠密真值或潜变量后验样本。解码器所不能表示的残差没有独立新增噪声分支，覆盖率可暴露这一限制。此候选筛查不能直接证明未观测细节恢复。',
      '原理与所有模块见ARCHITECTURE.md；前向加噪/v恢复、因果性、梯度、源哈希、保存预测复算及完整背景＋残差场回查见verification.json。完整格网合成在verify_diffusion_candidates.py中实现，补齐设计文档中原先仅描述的发布步骤。',
      '不读取C/D；输入仅A的Z0/T/P，历史制包PWV筛选限制继续适用。月22/27日均属已暴露开发日历，不作最终盲测。','',
      f"本机 {r['runtime']['device']}；训练与外层评价总耗时 {r['runtime']['seconds']:.1f}秒。八次训练共12000更新。","",
      '## 独立集合复核','',
      '| 模型 | 月27 K64综合分 | 月27 K128综合分 | 月22 K64综合分 | 月22 K128综合分 |',
      '|---|---:|---:|---:|---:|']
    for kind in d.KINDS:
        v=comparison[kind]
        lines.append(f"| {labels[kind]} | {v['reserved']['k64']['mean_selection_score']:.4f} | {v['reserved']['k128']['mean_selection_score']:.4f} | {v['confirmation']['k64']['mean_selection_score']:.4f} | {v['confirmation']['k128']['mean_selection_score']:.4f} |")
    lines+=['','综合分为两种子平均CRPS+0.5×1h增量CRPS，越低越好；用于开发候选筛查，不替代单项指标。','',
       '## 时间成员对应诊断','',
       '对普通DiT月27日K64集合逐小时共同重排所有站的成员编号，保持单时刻空间联合场，打破跨时成员对应。','']
    shuffle={}
    for seed in (42,17):
        p=np.load(d.ROOT/f'transformer_{seed}_reserved.npz');s=p['samples'].copy();rng=np.random.default_rng(771)
        for b in range(len(s)):
            for t in range(s.shape[2]):s[b,:,t]=s[b,rng.permutation(s.shape[1]),t]
        score=d.old.inv.probability_score(s,p['target'],p['mask']);original=r['results'][f'transformer_{seed}_reserved']['k64']
        assert abs(score['crps']-original['crps'])<1e-5
        shuffle[str(seed)]=dict(original=original['increment_crps'],shuffled=score['increment_crps'])
        lines.append(f"- seed{seed}：原增量CRPS {original['increment_crps']:.4f}，重排后 {score['increment_crps']:.4f}。")
    lines+=['','此诊断评估成员轨迹中的时间依赖，不证明细尺度空间结构真实。','']
    (d.ROOT/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    write_json(d.ROOT/'comparison.json',dict(comparison=comparison,time_shuffle=shuffle,code_sha256=sha256(__file__)))
    print(json.dumps(comparison))


if __name__=='__main__':main()
