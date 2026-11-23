"""Reproducible local-method figures and robustness checks from saved outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_probe_diagnostics import PROJECT, ProbeData, emit, save_npz
from local_shared_field import LocalFieldFeatures
from local_joint_field import local_bases
from gnss_field.artifacts import read_json, sha256, write_json
from gnss_field.evaluation.bootstrap import _blocks, _draw_blocks

ROOT = PROJECT/'results/20261119_local_method_research_v02'
M0 = PROJECT/'results/20261119_local_joint_probability_v01/full2019'
FUSED = PROJECT/'results/20261119_local_fused_probability_v02/full2019'


def crossed_scores(root):
    z = dict(np.load(root/'joint_scores.npz'))
    rng = np.random.default_rng(20261117)
    results = {}
    for baseline in ('diagonal', 'shuffled'):
        model, base = z['correlated'], z[baseline]
        difference = np.nanmean(np.nanmean(model, axis=0), axis=0)-np.nanmean(np.nanmean(base, axis=0), axis=0)
        output = {}
        for days in (7, 14):
            blocks = _blocks(z['hours'], days)
            values = []
            for _ in range(2000):
                times = _draw_blocks(blocks, rng)
                groups = rng.integers(0, model.shape[1], model.shape[1])
                # Preserve each named group's time average, then resample groups.
                a = np.nanmean(model[times], axis=0)[groups].mean(0)
                b = np.nanmean(base[times], axis=0)[groups].mean(0)
                values.append(a-b)
            values = np.array(values)
            output[str(days)] = dict(difference=difference, ci95=np.quantile(values, [.025, .975], axis=0),
                                     block_days=days, n_time_blocks=len(blocks), n_groups=model.shape[1],
                                     columns=['ES_mm', 'VS_mm'], replicates=2000)
        results[baseline] = output
    write_json(root/'crossed_robustness.json', dict(source=sha256(__file__), input=sha256(root/'joint_scores.npz'), results=results))
    return results


def plots():
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    blend = read_json(ROOT/'full2019_blend/results.json')
    m0, fused = read_json(M0/'results.json'), read_json(FUSED/'results.json')
    labels = ['冻结M0', 'A站IDW', '输入锚定混合场']
    keys = ['M0', 'A_IDW', 'blend']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    values = [blend['metrics'][k]['station_macro_rmse_mm'] for k in keys]
    bars = ax[0].bar(labels, values, color=['#89929a', '#c18b44', '#337b99'])
    ax[0].bar_label(bars, fmt='%.3f', padding=4)
    ax[0].set(ylabel='站宏RMSE（mm）', title='完整2019开发：76站 / 642,258站时', ylim=(0, 12))
    values = [blend['increments_0_50km'][k]['group_macro_rmse_mm'] for k in keys]
    bars = ax[1].bar(labels, values, color=['#89929a', '#c18b44', '#337b99'])
    ax[1].bar_label(bars, fmt='%.3f', padding=4)
    ax[1].set(ylabel='Z0站间差值组宏RMSE（mm）', title='0—50 km：8组 / 100站对', ylim=(0, 13))
    fig.savefig(ROOT/'annual_accuracy.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    names, cols = ['相关场', '独立高斯', '成员置乱'], ['correlated', 'diagonal', 'shuffled']
    for j, metric in enumerate(['ES_mm', 'VS_mm']):
        x = np.arange(3)
        for delta, r, label, color in [(-.17, m0, '冻结M0均值', '#89929a'), (.17, fused, '混合均值', '#337b99')]:
            bars = axes[0, j].bar(x+delta, [r['joint'][k][metric] for k in cols], width=.32, label=label, color=color)
            axes[0, j].bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
        axes[0, j].set(xticks=x, xticklabels=names, ylabel=metric, title='2019联合评分：6组 / 49,908组时')
        axes[0, j].legend(fontsize=9)
    for r, label, color in [(m0, 'M0均值', '#89929a'), (fused, '混合均值', '#337b99')]:
        primary = r['analytic_common_marginal']['primary']
        nominal = np.array([.8, .9, .95])
        coverage = [primary['intervals'][str(int(v*100))]['coverage'] for v in nominal]
        width = [primary['intervals'][str(int(v*100))]['mean_width_mm'] for v in nominal]
        axes[1, 0].plot(nominal, coverage, 'o-', color=color, label=label)
        axes[1, 1].plot(nominal, width, 'o-', color=color, label=label)
    axes[1, 0].plot([.75, 1.], [.75, 1.], '--', color='#666666', lw=1)
    axes[1, 0].set(xlabel='名义覆盖率', ylabel='实际站宏覆盖率', title='三种依赖对照共用相同解析边缘')
    axes[1, 1].set(xlabel='名义覆盖率', ylabel='区间宽度（mm）', title='区间宽度与覆盖同时报告')
    for ax in axes[1]:
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.savefig(ROOT/'joint_scores_and_calibration.png', dpi=170)
    plt.close(fig)


def field_example():
    torch.set_num_threads(2)
    data = ProbeData(device='cpu')
    c = data.arrays(2019)
    dates = pd.to_datetime(c['hours'], unit='h')
    index = np.flatnonzero((dates.month == 7) & (dates.hour == 12))[0]
    hour = int(c['hours'][index])
    field = c['mean_fields'][index].reshape(data.store.shape)
    store = data.development
    features = LocalFieldFeatures(store, 'cpu')
    _, idw = features.fields(hour, field, c['mean_A'][index], store.context_values([hour])[0],
                             [store.context_values([hour-offset])[0] for offset in (6, 12, 18)])
    weight = read_json(ROOT/'B_only_parameters.lock.json')['blend']['weights'][0]
    mean = field+weight*(idw.numpy()-field)
    parameters = read_json(FUSED.parent/'B_only_parameters.lock.json')['model']['std']
    bases = local_bases(store)
    realizations = []
    for member in range(3):
        rng = np.random.default_rng(np.random.SeedSequence([42, hour, member, 982476]))
        residual = sum(s*(basis@rng.standard_normal(basis.shape[1])) for s, basis in zip(parameters, bases))
        realizations.append(mean+residual.reshape(store.shape))
    grid = store.grid
    land = grid['region'].astype(bool)
    extent = [grid['x'].min()/1000, grid['x'].max()/1000, grid['y'].min()/1000, grid['y'].max()/1000]
    lo, hi = np.quantile(np.r_[field[land], idw.numpy()[land], mean[land]], [.01, .99])
    fig, axes = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
    values = [field, idw.numpy(), mean, *realizations]
    titles = ['冻结M0基础场', 'A站IDW格网', '锁定混合均值', '联合潜在场成员1', '联合潜在场成员2', '联合潜在场成员3']
    for ax, value, title in zip(axes.flat, values, titles):
        im = ax.imshow(np.where(land, value, np.nan), origin='lower', extent=extent, cmap='viridis', vmin=lo, vmax=hi)
        ax.set(title=title, xlabel='Lambert X（km）', ylabel='Lambert Y（km）')
    fig.colorbar(im, ax=list(axes.flat), shrink=.75, label='公共高度Z0（mm）')
    fig.suptitle(f'{dates[index]} 名义小时；25 km输出间距；成员为模型样本，未加入观测nugget')
    fig.savefig(ROOT/'field_example.png', dpi=155)
    plt.close(fig)
    save_npz(ROOT/'field_example.npz', hour=np.array(hour), mean=mean, M0=field, IDW=idw.numpy(),
             latent_members=np.array(realizations), land=land, x=grid['x'], y=grid['y'])


def main():
    for root in (M0, FUSED):
        emit(f'Crossed time/group uncertainty: {root.parent.name}')
        crossed_scores(root)
    plots()
    field_example()
    emit('Figures and paired robustness saved')


if __name__ == '__main__':
    main()
