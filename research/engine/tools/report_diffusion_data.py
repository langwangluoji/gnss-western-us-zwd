"""Independent data-expansion checks and fixed-population comparison."""
import json
import numpy as np
import diffusion_data_budget as e
from gnss_field.artifacts import write_json,sha256


def main():
    r=json.loads((e.ROOT/'results.json').read_text());lock=json.loads((e.ROOT/'experiment.lock.json').read_text())
    source=json.loads((e.d.ROOT/'results.json').read_text());baseline=json.loads((e.d.old.inv.ROOT/'results.json').read_text())
    assert sha256(e.__file__)==r['runtime']['code_sha256']==lock['code_sha256']
    assert sha256(e.ROOT/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha256(e.d.__file__)==lock['diffusion_source_sha256']
    assert sha256(e.d.ROOT/'experiment.lock.json')==lock['source_experiment_lock_sha256']
    assert sha256(e.d.ROOT/'results.json')==lock['source_results_sha256']
    source_lock=json.loads((e.d.ROOT/'experiment.lock.json').read_text())
    assert sha256(e.d.ROOT/'geometry.npz')==source_lock['geometry_sha256']
    assert sha256(e.d.old.exp.ROOT/'data.npz')==source_lock['data_sha256']
    assert sha256(e.d.old.inv.ROOT/'selection.lock.json')==source_lock['v04_selection_sha256']
    for name,h in source['runtime']['checkpoints'].items():assert sha256(e.d.ROOT/name)==h
    for name,h in source_lock['source_checkpoints'].items():assert sha256(e.d.old.exp.ROOT/name)==h
    for name,h in source_lock['dependencies'].items():assert sha256(e.d.old.core.PROJECT/'tools'/name)==h
    assert sha256(e.ROOT/'expanded_data.npz')==lock['expanded_data_sha256']
    assert sha256(e.ROOT/'selection.lock.json')==r['runtime']['selection_sha256']
    for name,h in r['runtime']['checkpoints'].items():assert sha256(e.ROOT/name)==h
    original=np.load(e.d.old.exp.ROOT/'data.npz');expanded=np.load(e.ROOT/'expanded_data.npz')
    assert len(expanded['train_hours'])==144
    for suffix in ('hours','raw','y'):assert np.array_equal(original['train_'+suffix],expanded['train_'+suffix][:72],equal_nan=True)
    extra=expanded['train_hours'][72:];assert np.array_equal(extra,np.array(lock['extra_hours']))
    assert len(np.unique(expanded['train_hours']))==144*30
    for name in ('inner','outer','confirmation'):assert not np.intersect1d(extra,original[name+'_hours']).size
    nonb=np.setdiff1d(np.arange(len(original['heights'])),original['bids'])
    assert np.isnan(expanded['train_y'][...,nonb]).all()
    maxerr=0.
    for key,score in r['results'].items():
        p=np.load(e.ROOT/f'{key}.npz');arm,seed,name=key.split('_');ref=np.load(e.d.old.inv.ROOT/f'history_{seed}_{name}.npz')
        for field in ('station_ids','hours','target','mask'):assert np.array_equal(p[field],ref[field])
        assert np.isfinite(p['samples']).all()
        point=e.d.old.inv.point_score(p['prediction'],p['target'],p['mask']);prob=e.d.old.inv.probability_score(p['samples'],p['target'],p['mask'])
        for k,v in point.items():maxerr=max(maxerr,abs(v-score['k64']['point'][k]))
        for k,v in prob.items():maxerr=max(maxerr,abs(v-score['k64'][k]))
    assert maxerr<1e-5
    rows={}
    for name in ('confirmation','reserved'):
        rows[name]={}
        for seed in (42,17):
            rows[name][seed]={
                '72x1500':source['results'][f'transformer_{seed}_{name}'],
                '72x3000':r['results'][f'original_{seed}_{name}'],
                '144x3000':r['results'][f'expanded_{seed}_{name}']}
    lines=['# Diffusion Transformer样本量与步数对照','',
        '保持A174观测网、输入变量、背景、24维残差基、归一化和评价集合不变，增加训练片段数量。此实验在v07筛查后开展，属于适应性开发实验。','',
        '## 月27日固定71站/20108站时，独立K128','',
        '| 条件 | RMSE42/mm | RMSE17/mm | CRPS42 | CRPS17 | 增量CRPS42 | 增量CRPS17 |','|---|---:|---:|---:|---:|---:|---:|']
    for key in ('72x1500','72x3000','144x3000'):
        v=[rows['reserved'][s][key]['k128'] for s in (42,17)]
        lines.append(f"| {key} | {v[0]['point']['macro_rmse']:.4f} | {v[1]['point']['macro_rmse']:.4f} | {v[0]['crps']:.4f} | {v[1]['crps']:.4f} | {v[0]['increment_crps']:.4f} | {v[1]['increment_crps']:.4f} |")
    b=[baseline['results'][str(s)]['reserved']['history'] for s in (42,17)]
    lines.append(f"| 原v04 | {b[0]['point']['macro_rmse']:.4f} | {b[1]['point']['macro_rmse']:.4f} | {b[0]['k128']['crps']:.4f} | {b[1]['k128']['crps']:.4f} | {b[0]['k128']['increment_crps']:.4f} | {b[1]['k128']['increment_crps']:.4f} |")
    lines+=['','## 独立集合和日历复核','',
        '| 条件 | 月27 K64综合分 | 月27 K128综合分 | 月22 K64综合分 | 月22 K128综合分 |','|---|---:|---:|---:|---:|']
    means={}
    for key in ('72x1500','72x3000','144x3000'):
        values=[]
        for name in ('reserved','confirmation'):
            for members in ('k64','k128'):
                v=[rows[name][s][key][members] for s in (42,17)]
                values.append(float(np.mean([x['crps']+.5*x['increment_crps'] for x in v])))
        means[key]=values;lines.append('| '+key+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    lines+=['','综合分为CRPS+0.5×1h增量CRPS，两种子平均，越低越好。点RMSE使用集合均值，原v04为解析均值。','',
        '## 数据与验证','',
        '新增2016/2017每月4、12、22日，共72段24小时，另带6小时暖启动。原72段逐值保留，所有训练窗口互不重叠；A输入站数没有增加。保持3000次更新意味着144段中每段平均访问次数更少，不是等epoch扩张。',
        '标准化和空间基沿用v07，B-fit系数仅使用各自训练时段监督。评价人口与原v04逐字段一致，保存预测K64与点指标复算误差见verification.json。所有原数据前1500步内层轨迹与v07核对。',
        '基础24维投影只覆盖约18%训练残差平方误差；扩数据未增加可表示空间。覆盖率、CRPS与时间评分仍是扩散模型验收条件。','',
        f"本轮四次训练共12000更新，训练与外层评价耗时 {r['runtime']['seconds']:.1f}秒。C/D未读取，历史PWV队列筛选限制继续适用。",'']
    (e.ROOT/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    checks=dict(source_hashes=True,training_windows=144,added_windows=72,original_data_preserved=True,only_B_labels=True,same_evaluation_population=True,
                saved_metric_max_error=maxerr,original_trajectory_errors={str(s):r['selection'][f'original_{s}']['original_trajectory_max_error'] for s in (42,17)})
    write_json(e.ROOT/'verification.json',dict(checks=checks,comparison=means,code_sha256=sha256(__file__)))
    print(json.dumps(dict(checks=checks,comparison=means)))


if __name__=='__main__':main()
