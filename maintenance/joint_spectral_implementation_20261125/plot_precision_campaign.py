"""Plot saved measurements, distinguishing varying batches and fixed evaluation."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from precision_analysis import ENGINE


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ENGINE/'results'
    out = root/'20261125_warm_joint_precision/combined_precision'
    comparison = read(out/'comparison.json'); sources = [out/'comparison.json']; figures = []
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for family, folder, color in [('S raw-history', '20261125_station_query_precision', '#357c9b'),
                                   ('A anomaly-history', '20261125_station_anomaly_precision', '#bd7545'),
                                   ('D distance-balanced', '20261125_station_distance_precision', '#35836c')]:
        for seed, style in [(42, '-'), (17, '--')]:
            path = root/folder/f'trajectory_s{seed}/curve.json'
            if not path.exists(): continue
            rows = read(path); sources.append(path)
            ax.plot([r['step'] for r in rows], [r['batch_rmse_mm'] for r in rows], style,
                    color=color, label=f'{family} seed{seed}', linewidth=1.3, alpha=.85)
    ax.set(xlabel='Updates (chronological target blocks)', ylabel='Training batch RMSE (mm)',
           title='Native station branches: batches change with season and station layout')
    ax.grid(alpha=.2); ax.legend(fontsize=9); ax.spines[['right', 'top']].set_visible(False)
    path = out/'native_training_curves.png'; fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)
    rows = comparison['two_seed_means']
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows)*.45+1.6)), layout='constrained')
    for i, row in enumerate(rows):
        lo, hi = row['common_month_two_seed_paired95_mm']; delta = row['delta_macro_rmse_mm']
        color = '#317b69' if row['improves_both_seeds'] and hi < 0 else '#627b8d'
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(delta, i, 'o', color=color)
        ax.annotate(f" {row['macro_rmse_mm']:.6f} mm", (max(hi, delta), i), fontsize=8, va='center')
    ax.axvline(0, color='#555', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(rows)), [r['case'] for r in rows]); ax.invert_yaxis()
    ax.margins(x=.35)
    ax.set(xlabel='Macro RMSE change vs matched GRU+p14 (mm); common-month 95% interval',
           title='Final 3000-step paired development evaluation; complete seed pairs only')
    ax.grid(axis='x', alpha=.2); ax.spines[['right', 'top']].set_visible(False)
    path = out/'matched_precision_changes.png'; fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)
    capacity = root/'20261125_native_capacity_probe/results.json'
    if capacity.exists():
        c = read(capacity); sources.append(capacity)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        ax.plot([r['step'] for r in c['trace']], [r['fixed_window_rmse_mm'] for r in c['trace']],
                color='#357c9b', marker='.', label='Added branch fitting')
        ax.axhline(c['reference_fixed_window_rmse_mm'], color='#888', linestyle='--', label='Frozen reference')
        ax.axhline(c['unattainable_pointwise_oracle_5mm_bound_floor'], color='#bd7545',
                   linestyle=':', label='Label oracle under +/-5 mm bound (diagnostic)')
        ax.set(xlabel='Updates on the same continuous 2-hour training window', ylabel='Fixed-window RMSE (mm)',
               title='Capacity probe: fitting ability, not development accuracy')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.spines[['right', 'top']].set_visible(False)
        path = out/'native_capacity_curve.png'; fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)
    support = root/'20261125_support_distribution_audit/results.json'
    if support.exists():
        s = read(support); sources.append(support)
        fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained'); x = np.arange(4)
        for offset, (regime, label, color) in enumerate([
                ('random', 'Random training layouts', '#62899b'),
                ('quarter_hole100', '25% hole100 training layouts', '#bd7545')]):
            fractions = np.mean([r['record_fractions'] for r in s['results'] if r['regime'] == regime], axis=0)
            ax.bar(x+(offset-1)*.25, fractions*100, width=.25, label=label, color=color)
        ax.bar(x+.25, np.array(s['distance_weight_derivation']['target_fractions'])*100,
               width=.25, label='Development geometry only', color='#35836c')
        ax.set_xticks(x, ['<50', '50-100', '100-200', '>=200'])
        ax.set(xlabel='Nearest current condition distance (km)', ylabel='Target fraction (%)',
               title='Training distance distribution changes little with 25% hole100 layouts')
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8); ax.spines[['right', 'top']].set_visible(False)
        path = out/'support_distribution.png'; fig.savefig(path, dpi=170); plt.close(fig); figures.append(path)
    manifest = dict(script_sha256=sha(Path(__file__)),
        sources={str(p.relative_to(ENGINE)): sha(p) for p in sources},
        figures={p.name: sha(p) for p in figures},
        note='Intervals resample paired months, not new spatial populations or random seeds')
    (out/'campaign_figure_manifest.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode('utf-8'))
    print([p.name for p in figures])


if __name__ == '__main__': main()
