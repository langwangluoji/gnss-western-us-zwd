"""Diagnose saved matched predictions without fitting or selecting a candidate."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from precision_analysis import ENGINE, statistics
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    out = args.warm / 'combined_precision'
    protocol = json.loads((args.warm / 'protocol.json').read_text(encoding='utf-8'))
    source = Path(protocol['config']['source'])
    baseline = {s: dict(np.load(source / f'gru_s{s}/evaluation.npz')) for s in (42, 17)}
    records = []
    for root in [args.warm] + args.runs:
        receipt = root / 'summary.json'
        if not receipt.exists():
            continue
        for row in json.loads(receipt.read_text(encoding='utf-8')):
            if row['case'] == 'reference':
                continue
            path = root / f"{row['case']}_s{row['seed']}/evaluation.npz"
            data = dict(np.load(path)); ref = baseline[row['seed']]
            np.testing.assert_array_equal(data['truth'], ref['truth'])
            correction = data['prediction'].astype('float64') - ref['prediction'].astype('float64')
            target = data['truth'].astype('float64')[None] - ref['prediction'].astype('float64')
            valid = np.isfinite(target)
            c = correction[valid]; t = target[valid]
            denominator = float(np.dot(c, c))
            # Development labels here support diagnostics only, never a deployed scale.
            oracle = float(np.dot(c, t) / denominator) if denominator > 0 else None
            monthly = []
            for m in range(12):
                old = statistics(ref['prediction'][:, m:m+1], data['truth'][m:m+1])
                new = statistics(data['prediction'][:, m:m+1], data['truth'][m:m+1])
                monthly.append(dict(month=m+1, reference=old['macro_rmse_mm'],
                                    candidate=new['macro_rmse_mm'],
                                    delta=new['macro_rmse_mm']-old['macro_rmse_mm']))
            records.append(dict(case=row['case'], seed=row['seed'],
                correction_rms_mm=float(np.sqrt(np.mean(c*c))),
                correction_std_mm=float(np.std(c)), correction_mean_mm=float(np.mean(c)),
                correction_q01_q50_q99_mm=np.quantile(c, [.01, .5, .99]).tolist(),
                residual_correlation=float(np.corrcoef(c, t)[0, 1]) if np.std(c) > 0 else None,
                mse_reduction_mm2=float(np.mean(t*t)-np.mean((t-c)**2)),
                oracle_development_scale_diagnostic_only=oracle, monthly=monthly,
                prediction_sha256=sha(path)))
    write_json(out / 'correction_diagnostics.json', dict(
        purpose='Posthoc amplitude/alignment diagnosis, no training or model selection',
        oracle_warning='Uses exposed development truth; oracle scale is never a model result',
        records=records))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    labels = [f"{r['case']}\ns{r['seed']}" for r in records]
    axes[0].bar(np.arange(len(records)), [r['correction_rms_mm'] for r in records], color='#397a9d')
    axes[0].set_xticks(np.arange(len(records)), labels, rotation=65, ha='right', fontsize=7)
    axes[0].set_ylabel('Correction RMS (mm)'); axes[0].set_title('Size of learned correction')
    for r in records:
        axes[1].plot(np.arange(1, 13), [m['delta'] for m in r['monthly']],
                     label=f"{r['case']} s{r['seed']}", linewidth=1, alpha=.75)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_xlabel('2020 month'); axes[1].set_ylabel('Macro RMSE change (mm)')
    axes[1].set_title('Final-weight monthly paired changes')
    geo = json.loads((out / 'geographic_and_pair_metrics.json').read_text(encoding='utf-8'))
    distance = np.mean([[x['macro_rmse_mm'] for x in geo[f'reference_s{s}']['nearest_current_condition_km']]
                        for s in (42, 17)], axis=0)
    axes[2].bar(['<50', '50-100', '100-200', '>=200'], distance, color='#a36c45')
    axes[2].set_xlabel('Nearest current condition distance (km)')
    axes[2].set_ylabel('Reference macro RMSE (mm)'); axes[2].set_title('Spatial gaps are the priority')
    for axis in axes:
        axis.grid(axis='y', alpha=.2); axis.spines[['top', 'right']].set_visible(False)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle('Matched ~10 mm foundation: correction and spatial-support diagnostics')
    figure = out / 'precision_diagnostics.png'; fig.savefig(figure, dpi=170); plt.close(fig)
    write_json(out / 'diagnostic_figure_manifest.json', dict(
        script_sha256=sha(Path(__file__)), figure_sha256=sha(figure),
        source_sha256=sha(out / 'correction_diagnostics.json')))
    print(json.dumps([dict(case=r['case'], seed=r['seed'], rms=r['correction_rms_mm'],
                          correlation=r['residual_correlation']) for r in records]))


if __name__ == '__main__':
    main()
