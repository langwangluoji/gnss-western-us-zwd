"""Scientific figures from verified outer-pilot and independent-pilot diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    root = Path(__file__).resolve().parents[1]/'results/20261122_generation_postrun_research_v01'
    source = root/'downloaded'
    data = json.loads((root/'outer_diagnostics.json').read_text(encoding='utf-8'))
    pilot = json.loads((root/'independent_pilot_diagnostics.json').read_text(encoding='utf-8'))
    bootstrap = json.loads((root/'paired_weather_week_bootstrap.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    arms = ['stochastic', 'diffusion', 'gan']; names = ['随机残差', 'Diffusion', 'GAN']
    colors = ['#287b8e', '#d17b31', '#7662a4']

    def save(fig, name):
        fig.savefig(root/f'{name}.png', dpi=180)
        fig.savefig(root/f'{name}.svg')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for f, ax in enumerate(axes):
        r = data['folds'][str(f)]; base = r['gaussian']['station_macro_rmse_mm']
        for i, (arm, name, color) in enumerate(zip(arms, names, colors)):
            record = bootstrap['folds'][str(f)][arm]
            value = record['difference_mm']; lo, hi = record['interval95_mm']
            ax.errorbar(i, value, yerr=[[value-lo], [hi-value]], fmt='o', color=color, capsize=5, markersize=8)
            ax.text(i, value+.025, f'+{value:.3f}', ha='center', color=color)
        ax.axhline(0, color='#52616b', linewidth=1)
        ax.set_xticks(range(3), names); ax.set_xlim(-.5, 2.5); ax.set_ylim(-.03, .76)
        ax.set_title(f'第{f}折 · 基础场 RMSE={base:.4f} mm')
        ax.set_ylabel('留出站宏RMSE差值 / mm（候选－基础场）')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('内层改善未迁移到空间留出站：六组模型均出现误差增加', fontsize=13)
    fig.supxlabel('2000次配对抽样天气周bootstrap；12簇×每簇8小时，未作多重比较调整', fontsize=9)
    save(fig, 'outer_rmse_paired_difference')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    labels = ['原生成集合\nK32', '同批移回均值\nK32·诊断', '独立先导中心化\nK16', '先导缩放高斯\nK16']
    for f, ax in enumerate(axes):
        r = data['folds'][str(f)]
        for j, (arm, name, color) in enumerate(zip(arms, names, colors)):
            q = r['arms'][arm]['joint']; g = r['gaussian']['group_ES_mm']
            independent = pilot['folds'][str(f)][arm]
            value = [q['original']['group_ES_mm']-g, q['generated_at_baseline_mean']['group_ES_mm']-g]
            value += [np.mean([x[key]['group_ES_mm']-x['baseline']['group_ES_mm'] for x in independent])
                      for key in ('independently_centered', 'pilot_scaled_gaussian')]
            ax.bar(np.arange(4)+(j-1)*.23, value, width=.22, color=color, label=name)
        ax.axhline(0, color='#52616b', linewidth=1); ax.grid(axis='y', alpha=.2)
        ax.set_xticks(range(4), labels); ax.set_title(f'第{f}折')
        ax.set_ylabel('站组ES差值 / mm（相对各自同K高斯）')
    axes[0].legend(frameon=False)
    fig.suptitle('均值漂移掩盖概率收益；简单幅度缩放已能解释本轮小幅改善', fontsize=13)
    fig.supxlabel('两方向16+16先导/评分成员互换；缩放参数来自预测集合，不使用目标标签', fontsize=9)
    save(fig, 'probability_mean_shape_controls')

    calibrated = json.loads((root/'gaussian_scale_control.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
    for f, ax in enumerate(axes):
        r = data['folds'][str(f)]
        baseline = r['gaussian']['group_ES_mm']
        values = [baseline, calibrated['folds'][str(f)]['field_only']['metrics']['group_ES_mm']]
        values += [r['arms'][arm]['joint']['original']['group_ES_mm'] for arm in arms]
        ax.scatter(range(5), values, c=['#8b969e', '#3b8d64', *colors], s=75, zorder=3)
        for i, value in enumerate(values):
            ax.text(i, value+.012, f'{value:.4f}', ha='center', fontsize=9)
        ax.set_xlim(-.5, 4.5)
        ax.set_xticks(range(5), ['原高斯', '单参数\n校准高斯', *names])
        ax.set_ylim(min(values)-.08, max(values)+.08); ax.grid(axis='y', alpha=.2)
        ax.set_ylabel('外层站组ES / mm（K32，越低越好）'); ax.set_title(f'第{f}折')
    fig.suptitle('本轮B2018试验：训练资料拟合的单参数高斯对照表现更好', fontsize=13)
    fig.supxlabel('仅缩放潜在空间场，基础均值、天气幅度和站端噪声固定；两折系数0.5990 / 0.6014', fontsize=9)
    save(fig, 'calibrated_gaussian_control')

    stations = pd.read_csv(source/'metadata/stations.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1), layout='constrained')
    for f, ax in enumerate(axes):
        g = dict(np.load(source/f'fold_{f}/gaussian_outer.npz'))
        d = dict(np.load(source/f'fold_{f}/diffusion/point_outer_predictions.npz'))
        mask = g['population']; count = mask.sum(0); use = count > 0
        errors = []
        for prediction in (g['mean'], d['mean']):
            errors.append(np.sqrt(np.where(mask, prediction-g['truth'], 0.)[:, use].__pow__(2).sum(0)/count[use]))
        xy = stations.iloc[g['station_ids'][use]][['x', 'y']].to_numpy()/1000
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=errors[1]-errors[0], cmap='RdBu_r',
                             norm=TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1.), s=22, edgecolors='none')
        ax.set_aspect('equal'); ax.set_title(f'第{f}折 · {use.sum()}个实际评分站')
        ax.set_xlabel('Lambert X / km'); ax.set_ylabel('Lambert Y / km')
    fig.colorbar(scatter, ax=axes, label='Diffusion－基础场：逐站RMSE差值 / mm', shrink=.8, extend='both')
    fig.suptitle('Diffusion外层误差的空间分布：蓝色改善，红色增加', fontsize=13)
    save(fig, 'diffusion_station_rmse_map')

    cases = [dict(np.load(root/f'publication_cases/fold_0/{arm}/joint_publication_case.npz')) for arm in arms]
    land = cases[0]['land'].astype(bool)
    means = np.concatenate([x['mean_Z0_control_variate'][land] for x in cases])
    stds = np.concatenate([x['std_Z0'][land] for x in cases])
    lo, hi = np.quantile(means, [.01, .99]); shi = np.quantile(stds, .99)
    fig, axes = plt.subplots(2, 3, figsize=(11, 8), layout='constrained')
    for j, (case, name) in enumerate(zip(cases, names)):
        x, y = case['x']/1000, case['y']/1000
        for row, key in enumerate(('mean_Z0_control_variate', 'std_Z0')):
            z = np.where(land, case[key], np.nan)
            im = axes[row, j].pcolormesh(x, y, z, shading='auto', cmap='viridis' if row == 0 else 'magma',
                vmin=lo if row == 0 else 0, vmax=hi if row == 0 else shi)
            axes[row, j].set_aspect('equal'); axes[row, j].set_title(name + (' · 发布均值' if row == 0 else ' · 场标准差'))
            axes[row, j].set_xlabel('X / km'); axes[row, j].set_ylabel('Y / km')
            fig.colorbar(im, ax=axes[row, j], shrink=.8, label='Z0 / mm')
    fig.suptitle('固定代表时刻 2018-01-18 10:00 · 第0折 · 25 km输出间距', fontsize=13)
    fig.supxlabel('均值已恢复控制变量估计；标准差来自32成员潜在场，不含站端测量噪声。非整幅独立真值验证。', fontsize=9)
    save(fig, 'representative_publication_fields')


if __name__ == '__main__':
    main()
