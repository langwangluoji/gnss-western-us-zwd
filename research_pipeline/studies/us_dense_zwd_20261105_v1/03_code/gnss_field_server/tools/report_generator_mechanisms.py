"""Fixed-endpoint paired evidence, using only fitting/inner-B predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261123_generator_mechanism_validation_v01'
FAMILIES = ('vae', 'gan', 'diffusion')
PANELS = ('training_fitting', 'training_withheld', 'withheld_weather_fitting', 'withheld_weather_withheld')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def evidence_paths(family, arm, members=16, seed=42):
    if seed != 42:
        return ROOT/'replication'/family/f'{arm}_k{members}'
    if members == 16:
        return ROOT/'endpoints'/family/f'{arm}_k16'
    return ROOT/f'endpoints{members}'/family/arm


def paired_bootstrap(control_path, intervention_path, repetitions=2000):
    control, intervention = np.load(control_path.with_suffix('.npz')), np.load(intervention_path.with_suffix('.npz'))
    for key in ('station_ids', 'withheld_weather_target', 'withheld_weather_mask', 'withheld_weather_hours'):
        np.testing.assert_array_equal(control[key], intervention[key])
    # The locked archive orders fitting357 then inner77. Read the actual split size.
    fitting_count = len(np.load(PROJECT/'results/20261123_simple_weather_unet_v02/data.npz')['split_fit'])
    target = control['withheld_weather_target'][:, fitting_count:]
    mask = control['withheld_weather_mask'][:, fitting_count:]
    hours = control['withheld_weather_hours']
    months = (np.datetime64('1970-01-01T00')+hours.astype('timedelta64[h]')).astype('datetime64[M]')
    groups = [np.flatnonzero(months == month) for month in np.unique(months)]
    if len(groups) != 12:
        raise ValueError('Expected12 registered monthly weather blocks')
    arrays = []
    for archive in (control, intervention):
        samples = archive['withheld_weather_samples'][:, :, fitting_count:].astype('f8')
        clean = np.where(mask, target, 0.)
        squared = np.where(mask, samples.mean(1)-target, 0.)**2
        k = samples.shape[1]
        crps = abs(samples-clean[:, None]).mean(1)-(np.sort(samples, axis=1)*(2*np.arange(1, k+1)-k-1)[None, :, None]).sum(1)/(k*(k-1))
        arrays.append((squared, crps*mask))
    rng = np.random.default_rng(20261123)
    changes = []
    for _ in range(repetitions):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        counts = mask[rows].sum(0)
        active = counts > 0
        values = [(np.sqrt(squared[rows].sum(0)[active]/counts[active]).mean(),
                   (crps[rows].sum(0)[active]/counts[active]).mean()) for squared, crps in arrays]
        changes.append(np.asarray(values[1])-values[0])
    interval = np.quantile(changes, [.025, .975], axis=0)
    return dict(delta_macro_rmse_ci95=interval[:, 0].tolist(), delta_macro_crps_ci95=interval[:, 1].tolist(),
                resamples=repetitions, blocks=len(groups),
                interpretation='Exploratory paired monthly-weather-block interval; not full-year7-day bootstrap and not MC-seed uncertainty')


def screen(control, intervention):
    c, i = control['scores']['withheld_weather_withheld'], intervention['scores']['withheld_weather_withheld']
    delta = i['macro_rmse']-c['macro_rmse']
    es = i['group_ES_mm']/c['group_ES_mm']-1
    crps = i['macro_CRPS_mm']/c['macro_CRPS_mm']-1
    coverage = i['coverage90']-c['coverage90']
    point = delta <= -.10 and es <= .01 and crps <= .01
    probability = (es <= -.01 or crps <= -.01) and es <= .01 and crps <= .01 and delta <= .10 and coverage >= -.02
    return dict(delta_macro_rmse_mm=delta, relative_group_ES_change=es, relative_macro_CRPS_change=crps,
                delta_coverage90=coverage, point_screen=bool(point), probability_screen=bool(probability))


def confirmed_point_direction(primary, high_k, replication, independent_noise):
    return bool(all(r['point_screen'] for r in (primary, high_k, replication, independent_noise))
                and primary['delta_macro_rmse_ci95'][1] < 0)


def report(members=16):
    if read(ROOT/'paired_results.json')['status'] != 'COMPLETE':
        raise ValueError('Six arms must complete before reporting')
    out = ROOT/f'report_k{members}'
    out.mkdir(exist_ok=True)
    config = read(ROOT/'experiment.lock.json')
    results = {}
    lines = ['# 三类模型本机配对验证', '',
             f'每臂新增{config["updates"]}次更新，batch4小时，训练K4，端点评价K{members}。只使用拟合B与内部B，不查询C/D评分标签。', '',
             'VAE干预仅移除后期ELBO；GAN仅关闭生成器对抗项，判别器路径保留；Diffusion仅增加无偏均值MSE。',
             'G学习率5e-5→1e-5；GAN判别器保留源设置：学习率2e-4、weight decay0.01。', '',
             '|模型|源检查点RMSE|对照末尾RMSE|干预末尾RMSE|干预−对照|对照宏CRPS|干预宏CRPS|对照站组ES|干预站组ES|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = ['Fit hours / fit sites', 'Fit hours / withheld sites', 'Withheld weather / fit sites', 'Both withheld']
    for col, family in enumerate(FAMILIES):
        records = {arm: read(evidence_paths(family, arm, members).with_suffix('.json')) for arm in ('source', 'control', 'intervention')}
        result = screen(records['control'], records['intervention'])
        result.update(paired_bootstrap(evidence_paths(family, 'control', members), evidence_paths(family, 'intervention', members)))
        result['source_macro_rmse'] = records['source']['scores']['withheld_weather_withheld']['macro_rmse']
        result['intervention_vs_source_rmse'] = records['intervention']['scores']['withheld_weather_withheld']['macro_rmse']-result['source_macro_rmse']
        result['records'] = records
        results[family] = result
        s, c, i = [records[a]['scores']['withheld_weather_withheld'] for a in ('source', 'control', 'intervention')]
        lines.append(f'|{family}|{s["macro_rmse"]:.4f}|{c["macro_rmse"]:.4f}|{i["macro_rmse"]:.4f}|{result["delta_macro_rmse_mm"]:+.4f}|'
                     f'{c["macro_CRPS_mm"]:.4f}|{i["macro_CRPS_mm"]:.4f}|{c["group_ES_mm"]:.4f}|{i["group_ES_mm"]:.4f}|')
        for row, metric in enumerate(('macro_rmse', 'macro_CRPS_mm')):
            for offset, arm, color in [(-.25, 'source', '#888888'), (0., 'control', '#4878b0'), (.25, 'intervention', '#d78838')]:
                axes[row, col].bar(np.arange(4)+offset, [records[arm]['scores'][p][metric] for p in PANELS], .24, color=color, label=arm)
            axes[row, col].set(title=family.upper(), ylabel='Station-macro RMSE (mm)' if row == 0 else 'Station-macro CRPS (mm)')
            axes[row, col].set_xticks(np.arange(4), labels, rotation=18, ha='right', fontsize=8)
            axes[row, col].grid(axis='y', alpha=.2)
            axes[row, col].set_axisbelow(True)
        axes[0, col].legend(fontsize=8)
    fig.savefig(out/'paired_four_panel.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, family in zip(axes, FAMILIES):
        for arm in ('control', 'intervention'):
            curve = read(ROOT/'seed42'/family/arm/'curve.json')
            steps = np.array([v['step'] for v in curve])
            values = np.array([v['es_mm'] for v in curve])
            complete = len(values)//25*25
            ax.plot(steps[:complete].reshape(-1, 25).mean(1), values[:complete].reshape(-1, 25).mean(1), label=arm)
        ax.set(title=family.upper(), xlabel='New-stage updates', ylabel='Training ES (mm),25-update mean')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out/'paired_training_es.png', dpi=170)
    plt.close(fig)
    lines += ['', '配对变化为干预−对照，负误差差值表示改善。源检查点单列，防止把“比继续恶化的对照好”直接当作超越已有结果。',
              '宏CRPS与宏RMSE使用同一人口；原微CRPS、覆盖率、宽度、VS、跨200km格联合ES保留于JSON。',
              '置信区间使用12个月份天气块同步重采样2000次，只作小试诊断；不能代替全年7日块bootstrap，也不估计采样噪声。', '',
              '|模型|RMSE变化95%区间|宏CRPS变化95%区间|点方向门槛|概率方向门槛|',
              '|---|---|---|---|---|']
    for family, r in results.items():
        lines.append(f'|{family}|{r["delta_macro_rmse_ci95"]}|{r["delta_macro_crps_ci95"]}|{r["point_screen"]}|{r["probability_screen"]}|')
    lines += ['', '门槛通过属于方向筛查，不能单独作为方法已成立的结论。仍需高成员数、独立采样噪声与适配种子复核；没有通过时报告本配置未显示足够收益。',
              '本实验不证明整类GAN/VAE/Diffusion可行或不可行，也不检验尚未实现的显式双解码器架构。', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'paired_analysis.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('PAIRED_REPORT_COMPLETE', members, {f: {k: v for k, v in r.items() if k in ('delta_macro_rmse_mm', 'relative_group_ES_change', 'relative_macro_CRPS_change', 'point_screen', 'probability_screen')} for f, r in results.items()})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', type=int, default=16)
    report(parser.parse_args().members)
