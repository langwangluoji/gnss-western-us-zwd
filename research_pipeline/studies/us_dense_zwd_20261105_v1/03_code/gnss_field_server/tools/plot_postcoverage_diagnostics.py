"""Render the completed, frozen diagnostics without recomputing or selecting models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]/'results/20261120_postcoverage_analysis_v01'


def main():
    result = json.loads((ROOT/'annual_results.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for arm, color in [('mean', '#176b87'), ('edm', '#bd5b29')]:
        rows = [json.loads(line) for line in (ROOT/'artifacts'/arm/'development.jsonl').read_text().splitlines()]
        axes[0].plot([r['step'] for r in rows], [r['metrics']['joint']['value'] for r in rows], label=arm.upper(), color=color)
    axes[0].set(title='B2018 development: EMA', xlabel='Updates', ylabel='Energy score (mm)')
    axes[0].legend(frameon=False)
    estimates = [result[k]['station_macro_rmse_mm'] for k in
        ('mean_analytic_point', 'EDM_integrated_mean_point', 'EDM_observation_ensemble_mean_point')]
    axes[1].bar(['Mean\nanalytic', 'EDM\nlatent-integrated', 'EDM\nnoisy finite128'], estimates,
        color=['#176b87', '#bd5b29', '#bac6cb'])
    axes[1].set(title='C2019 point estimators', ylabel='Station-macro RMSE (mm)', ylim=(9.90, 10.01))
    for i, value in enumerate(estimates):
        axes[1].text(i, value+.002, f'{value:.4f}', ha='center', fontsize=9)
    keys = ['EDM_minus_mean', 'EDM_minus_Gaussian_plus_EDM_shift', 'shape_only_EDM_minus_Gaussian']
    labels = ['EDM - mean', 'EDM - Gaussian + shift', 'Shape-only - Gaussian']
    for i, key in enumerate(keys):
        value = result['paired'][key]['difference_mm']['ES']
        lo, hi = result['paired'][key]['7']['intervals_mm']['ES']
        axes[2].errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', color='#176b87', capsize=4)
    axes[2].axvline(0, color='#8b9499', lw=1)
    axes[2].set(yticks=range(3), yticklabels=labels, xlabel='Energy score difference (mm)',
        title='C2019 paired 95% intervals')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Frozen products: optimization, Monte Carlo error and distribution attribution', fontsize=14)
    fig.savefig(ROOT/'diagnostics.png', dpi=160)
    plt.close(fig)
    print(ROOT/'diagnostics.png')


if __name__ == '__main__':
    main()
