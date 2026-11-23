"""Display the final two-parameter conditional-scale experiment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_probe_diagnostics import PROJECT
from gnss_field.artifacts import read_json


def main():
    root = PROJECT/'results/20261119_local_method_research_v02'
    a = read_json(PROJECT/'results/20261119_local_fused_probability_v02/full2019/results.json')
    b = read_json(PROJECT/'results/20261119_local_conditional_probability_v01/results.json')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    values = [[a['joint']['correlated']['ES_mm'], b['joint']['conditional']['ES_mm']],
              [a['joint']['correlated']['VS_mm'], b['joint']['conditional']['VS_mm']],
              [a['analytic_common_marginal']['primary']['station_macro_crps_mm'], b['marginal']['primary']['station_macro_crps_mm']]]
    for ax, value, label in zip(axes, values, ['组宏ES', '组宏VS', '站宏CRPS（mm）']):
        bars = ax.bar(['常数尺度', 'A条件尺度'], value, color=['#89929a', '#337b99'])
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set(ylabel=label, ylim=(0, max(value)*1.15))
    fig.suptitle('完整2019开发；混合均值固定；条件尺度仅用B2018拟合')
    fig.savefig(root/'conditional_probability.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
