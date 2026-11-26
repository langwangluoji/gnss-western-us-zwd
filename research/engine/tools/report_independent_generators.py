"""CPU-only postprocessing after the registered three-model queue finishes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261123_enhanced_encoder_generative_v03'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    result = read(ROOT/'generators_results.json')
    if result['status'] != 'COMPLETE':
        raise RuntimeError('Final report requires all three completed families')
    plots = ROOT/'figures'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    lines = ['# 独立GAN、Diffusion与VAE本机训练结果', '',
             '当前批次使用863个合格训练小时，三模型分别完整更新；不是服务器全量小时实验。D2020未读取。', '',
             '|模型|产品|选中步|C48h站宏RMSE|CRPS微平均|站组ES|90%覆盖率|90%宽度|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for family, run in result['families'].items():
        curve = read(ROOT/family/'curve.json')
        values = [x for x in curve if 'validation' in x]
        axes[0].plot([x['step'] for x in values], [x['validation']['macro_rmse'] for x in values], label=family)
        axes[1].plot([x['step'] for x in values], [x['validation']['group_ES_mm'] for x in values], label=family)
        loss = [x for x in curve if x.get('losses')]
        axes[2].plot([x['step'] for x in loss], [x['losses']['loss'] for x in loss], label=family)
        for product, record in run['products'].items():
            c = record['scores']['C2019_development']
            lines.append(f'|{family}|{product}|{record["step"]}|{c["macro_rmse"]:.4f}|{c["CRPS_mm"]:.4f}|'
                         f'{c["group_ES_mm"]:.4f}|{c["coverage90"]:.3f}|{c["width90_mm"]:.3f}|')
    for ax, title, ylabel in zip(axes, ['Internal point validation', 'Internal joint validation', 'Training objectives (different scales)'],
                                  ['Station-macro RMSE (mm)', 'Group-macro ES (mm)', 'Registered loss']):
        ax.set(title=title, xlabel='Updates', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(plots/'training_curves.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    nominal = np.array([80, 90, 95])
    axes[0].plot([0, 100], [0, 100], '--', color='grey', label='ideal')
    for family, run in result['families'].items():
        c = run['products']['joint']['scores']['C2019_development']
        axes[0].plot(nominal, [100*c[f'coverage{n}'] for n in nominal], 'o-', label=family)
        axes[1].plot(nominal, [c[f'width{n}_mm'] for n in nominal], 'o-', label=family)
    axes[0].set(xlabel='Nominal coverage (%)', ylabel='Observed station-macro coverage (%)', xlim=(0, 100), ylim=(0, 100))
    axes[1].set(xlabel='Nominal coverage (%)', ylabel='Station-macro interval width (mm)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(plots/'calibration_and_width.png', dpi=180)
    plt.close(fig)
    lines += ['', '评价解释：点产品与联合概率产品分别使用自己的检查点整行报告，不跨产品拼指标。',
              'CRPS为站时刻微平均；覆盖率与宽度为站宏平均；ES同时给出全站小时平均及固定200 km站组宏平均。',
              'VAE报告自由先验的解码场成员，不是包含固定3 mm似然噪声的完整观测预测分布。',
              '扩散使用6步Heun的完整场采样与真实站ES/CRPS训练，不具有稠密干净场DSM目标。',
              'C2019仅48个开发时刻，非全年、非盲测。K32集合尚未达到服务器K64/128稳定性复核规模。',
              '详细逐人口结果、成员打乱VS、各级覆盖与宽度见generators_results.json与各模型NPZ。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT/'report_complete.json').write_text(json.dumps(dict(status='COMPLETE', figures=2, all_families=list(result['families']))), encoding='utf-8')
    print('INDEPENDENT_GENERATORS_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
