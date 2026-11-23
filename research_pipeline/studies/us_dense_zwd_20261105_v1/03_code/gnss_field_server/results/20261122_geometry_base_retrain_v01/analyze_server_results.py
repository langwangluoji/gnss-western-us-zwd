"""Verify archived point scores and summarize the finite corrected-base experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ARMS = ('gan', 'diffusion', 'ndp_point', 'ndp_GP_point')
LABELS = ('GAN', 'Diffusion', 'NDP independent', 'NDP GP')


def main():
    receipt = json.loads((ROOT / 'collection_complete.json').read_text())
    assert receipt['passed']
    source = Path(receipt['directory'])
    manifest = json.loads((source / 'collection_manifest.json').read_text())
    rows = []; curves = {}; inner = {}; baselines = {}
    for fold in (0, 1):
        for arm in ('gaussian', *ARMS):
            folder = source / f'fold_{fold}' / arm
            if (folder / 'training.jsonl').exists():
                curves[fold, arm] = [json.loads(v) for v in (folder / 'training.jsonl').read_text().splitlines()]
            inner[fold, arm] = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob('inner_*.json'))}
            result_path = folder / 'result.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            products = result.get('products', {'baseline': {'metrics': result.get('metrics')}})
            for product, content in products.items():
                filename = 'predictions.npz' if product == 'baseline' else f'{product}_predictions.npz'
                path = folder / filename
                if product != 'baseline':
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == content['prediction_sha256']
                    checkpoint = folder / content['selected']['checkpoint']
                    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == content['checkpoint_sha256']
                arrays = np.load(path)
                mean = arrays['mean'].astype('f8'); truth = arrays['truth'].astype('f8')
                mask = arrays['population'].astype(bool)
                assert np.isfinite(mean).all() and np.isfinite(arrays['samples']).all()
                assert np.isfinite(truth[mask]).all()
                errors = (mean - truth)**2
                per_station = [np.sqrt(errors[:, i][mask[:, i]].mean())
                               for i in range(mask.shape[1]) if mask[:, i].any()]
                computed = float(np.mean(per_station))
                assert abs(computed - content['metrics']['station_macro_rmse_mm']) < 1e-7
                if product == 'baseline':
                    baselines[fold] = content['metrics']
                rows.append(dict(fold=fold, arm=arm, product=product,
                                 selected_step=content.get('selected', {}).get('step'),
                                 recomputed_macro_RMSE_mm=computed, **content['metrics']))
    state = torch.load(source / 'fold_1/ndp_GP_point/latest.pt', map_location='cpu', weights_only=False)
    paused = dict(fold=1, arm='ndp_GP_point', step=state['step'], target=state['metadata']['steps'])
    summary = dict(accounting=manifest['accounting'], completed_models=7, planned_models=8,
                   paused=paused, metric_verification_passed=True, products=rows,
                   inner_metrics={f'{f}/{a}': v for (f, a), v in inner.items()},
                   population='B2018 exposed incremental diagnostic;12 fixed outer monthly hours;K128;folds reported separately',
                   C2019_residual_head_evaluation=False, D2020=False)
    (ROOT / 'SERVER_RESULTS_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lookup = {(r['fold'], r['arm'], r['product']): r for r in rows}
    colors = ('#2878A5', '#D97736')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, product, metric, title in (
        (axes[0, 0], 'point', 'station_macro_rmse_mm', 'Point product: RMSE reduction vs new base'),
        (axes[0, 1], 'joint', 'group_ES_mm', 'Joint product: group ES reduction vs Gaussian')):
        for fold in (0, 1):
            values = []
            for arm in ARMS:
                row = lookup.get((fold, arm, product))
                values.append(np.nan if row is None else baselines[fold][metric] - row[metric])
            positions = np.arange(len(ARMS)) + (fold - .5) * .32
            bars = ax.bar(positions, values, width=.3, color=colors[fold], label=f'Fold {fold}')
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.annotate(f'{value:+.4f}', (bar.get_x() + bar.get_width()/2, value),
                                ha='center', va='bottom' if value >= 0 else 'top', fontsize=8,
                                xytext=(0, 4 if value >= 0 else -4), textcoords='offset points')
            if fold == 1:
                ax.text(positions[-1], 0, 'incomplete', ha='center', va='bottom', rotation=90, fontsize=8)
        ax.axhline(0, color='#555555', lw=.8);ax.set_xticks(np.arange(4), LABELS)
        ax.set_xlim(-.6, 3.6)
        ax.set_title(title, fontsize=11);ax.set_ylabel('Reduction (mm); positive = better')
        ax.margins(y=.28);ax.legend(frameon=False);ax.grid(axis='y', alpha=.18)
    for fold in (0, 1):
        for arm, style in (('gan', '-'), ('diffusion', '--')):
            history = curves[fold, arm]
            values = [r['ES_mm'] for r in history]
            smooth = np.convolve(values, np.ones(8)/8, mode='valid')
            axes[1, 0].plot(np.arange(8, len(values)+1), smooth, color=colors[fold],
                            ls=style, label=f'{arm} fold {fold}')
        for arm, style in (('ndp_point', '-'), ('ndp_GP_point', '--')):
            history = curves[fold, arm]
            values = [r['loss'] for r in history]
            if len(values) < 48:
                continue
            axes[1, 1].plot(np.arange(48, len(values)+1), np.convolve(values, np.ones(48)/48, mode='valid'),
                            color=colors[fold], ls=style, label=f'{arm} fold {fold}')
    for ax, title, ylabel in ((axes[1, 0], 'Grid-model training ES (8-update moving average)', 'Training ES (mm)'),
                              (axes[1, 1], 'Point NDP training DSM (48-update moving average)', 'Weighted denoising loss')):
        ax.set_title(title, fontsize=11);ax.set_xlabel('Optimizer update');ax.set_ylabel(ylabel)
        ax.grid(alpha=.18);ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Corrected-base retraining: B2018 diagnostic, not annual C2019 evaluation', fontsize=13)
    fig.savefig(ROOT / 'server_retraining_results.png', dpi=180)
    plt.close(fig)
    table = ['| 方法（点产品） | 第0折RMSE / mm | 第1折RMSE / mm |', '|---|---:|---:|']
    for arm, name in (('gaussian', '新几何底座'), ('gan', 'GAN'), ('diffusion', 'Diffusion'),
                      ('ndp_point', '独立噪声NDP'), ('ndp_GP_point', 'GP噪声NDP')):
        values = []
        for fold in (0, 1):
            row = lookup.get((fold, arm, 'baseline' if arm == 'gaussian' else 'point'))
            values.append('未完成' if row is None else f"{row['station_macro_rmse_mm']:.6f}")
        table.append(f'| {name} | {values[0]} | {values[1]} |')
    joint_table = ['| 联合产品 | 第0折组ES / mm | 第1折组ES / mm |', '|---|---:|---:|']
    for arm, name in (('gaussian', '新底座＋高斯残差'), ('gan', 'GAN'), ('diffusion', 'Diffusion'),
                      ('ndp_point', '独立噪声NDP'), ('ndp_GP_point', 'GP噪声NDP')):
        values = []
        for fold in (0, 1):
            row = lookup.get((fold, arm, 'baseline' if arm == 'gaussian' else 'joint'))
            values.append('未完成' if row is None else f"{row['group_ES_mm']:.6f}")
        joint_table.append(f'| {name} | {values[0]} | {values[1]} |')
    report = '''# 新几何底座复训结果

作业3688243已停止，单张A800累计3416秒（56分56秒），当前没有本批A800作业排队或运行。完成7/8项；最后一项第1折GP-NDP在第14/1440步保存后退出。Slurm记录FAILED，直接原因是预超时保护置位PAUSE后按保护逻辑保存退出，并非非有限数值错误。未追加服务器预算或启动新训练。

以下均为B2018已暴露的增量开发诊断：每折12个固定外层月度小时、128成员。第0折204站/2385站时刻，第1折244站/2812站时刻。两折分别报告，不能将折预测拼成单个可部署模型的联合分布。残差头尚未做全年C2019评价，D2020保持封存；此前几何底座全年C2019的9.296—9.357mm与本表不是同一评分总体。

## 点预测产品

''' + '\n'.join(table) + '''

点产品由内层站宏RMSE选定。第0折GAN与Diffusion均选回第0步零修正模型；第1折GAN选择72步、Diffusion选择36步。第0折独立噪声NDP改善0.0544mm，第1折增加0.0197mm。GAN/DF此次没有形成稳定的新增点精度收益。

## 联合概率产品

''' + '\n'.join(joint_table) + '''

联合产品独立按内层站组ES选模，不能把其点RMSE替换到点产品排名。Diffusion的组ES第0折降低0.003736mm、第1折增加0.002092mm；GAN第0折降低0.000970mm、第1折增加0.000418mm。这些变化小且未呈两折一致方向。

独立噪声NDP两折95%覆盖率从90.53%/89.55%增加到92.77%/91.42%，相应95%区间宽度从29.36/28.52mm增加到31.22/31.81mm。两折VS均增加，不能据此声称空间依赖改善。第0折GP-NDP的95%覆盖率为93.01%、宽度32.19mm，VS从1.3563降至1.3442；第1折尚无可比较结果。NDP当前是固定站点集合分布，不是连续高分辨率场。

## 曲线与运行判断

GAN/DF各72步只遍历576训练小时一次，尚不足以判定收敛或潜力上限。两者学到的外层均值修正RMS约0.04—0.095mm，主要保持在新底座附近。点NDP内层ES随训练下降，但外层收益存在折差异，下一步应先核查泛化及采样误差。

短基准含1.5倍余量估计3034.8秒，仍低估了完整队列的墙钟时间。例：第0折独立噪声NDP的训练循环含内层评价耗时720.1秒，而4步短测外推其更新耗时约84.2秒；第1折该循环为312.0秒。此差异需要对数据访问、同步、日志/检查点及评价分项计时，现有记录不足以将其归因于单个算子。

已从下载预测重新计算全部保存产品的站宏RMSE，与结果JSON误差小于1e-7mm；预测、选模权重及归档哈希均核验通过。原始记录保留，暂无新增训练。

下一步优先完成第1折GP-NDP，随后检查残差头修正幅度、学习率与更新量，再决定GAN/DF的配对延长试验。本次不以扩展开发数据重新挑选已报告产品。

![训练与评价](server_retraining_results.png)
'''
    (ROOT / 'SERVER_RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps({'passed': True, 'verified_products': len(rows), 'completed_models': 7,
                      'paused': paused, 'report': str(ROOT / 'SERVER_RESULTS.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
