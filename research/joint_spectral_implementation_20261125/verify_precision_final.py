"""Read-only verification of the frozen fit and both evaluation calendars."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

from precision_analysis import ROOT, ENGINE, statistics
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.innovation_experiment import CASES
from tools.terrain_diffusion.innovation_refinement import solve_ridge


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    results = ENGINE / 'results'
    fit = results / '20261125_observed_innovation_precision'
    extra = results / '20261125_innovation_calendar15'
    combined = results / '20261125_warm_joint_precision/combined_precision'
    protocol = read(fit / 'protocol.json')
    lock = read(fit / 'frozen_selection.json')
    additional = read(extra / 'protocol.json')
    assert lock['primary_candidate'] == 'I1_history'
    assert sha(fit / 'protocol.json') == lock['training_protocol_sha256'] == additional['training_protocol_sha256']
    assert sha(fit / 'summary.json') == lock['training_summary_sha256']
    assert sha(combined / 'comparison.json') == lock['primary_comparison_sha256']
    assert sha(fit / 'frozen_selection.json') == read(extra / 'primary_selection.json')['selection_lock_sha256']
    assert sha(Path(protocol['config']['data']) / 'contract.json') == protocol['data_sha256'] == additional['data_sha256']
    for name, expected in protocol['code_sha256'].items():
        assert sha(ENGINE / 'tools/terrain_diffusion' / name) == expected, name
    assert sha(ROOT / 'maintenance/joint_spectral_implementation_20261125/innovation_additional_calendar.py') == additional['code_sha256']
    coefficient_checks = []
    metric_checks = []
    metrics = ['macro_rmse_mm', 'micro_rmse_mm', 'mae_mm', 'bias_mm', 'increment_macro_rmse_mm']
    primary_records = read(combined / 'comparison.json')['records']
    extra_records = read(extra / 'results.json')['records']
    for seed in (42, 17):
        source = Path(protocol['config']['source']) / f'gru_s{seed}'
        assert sha(source / 'last.pt') == protocol['source_weights'][str(seed)] == additional['source_weights'][str(seed)]
        state_path = fit / f'trajectory_s{seed}/statistics_3000.pt'
        state = torch.load(state_path, weights_only=True, map_location='cpu')
        assert state['step'] == 3000 and state['seed'] == seed
        assert state['data_sha256'] == protocol['data_sha256']
        assert state['source_sha256'] == protocol['source_weights'][str(seed)]
        for name, dimension in CASES.items():
            path = fit / f'{name}_s{seed}/coefficients.pt'
            expected = lock['coefficient_sha256'][f'{name}_s{seed}']
            assert sha(path) == expected == additional['coefficient_sha256'][f'{name}_s{seed}']
            assert sha(state_path) == next(c['statistics_sha256'] for c in lock['checks'] if c['case'] == name and c['seed'] == seed)
            actual = torch.load(path, weights_only=True, map_location='cpu')
            derived = solve_ridge(state['matrix'][:dimension, :dimension], state['rhs'][:dimension])
            assert torch.equal(actual, derived)
            coefficient_checks.append(dict(case=name, seed=seed, sha256=expected, exact=True))
        for calendar, records in [('primary', primary_records), ('supplementary', extra_records)]:
            reference_path = source / 'evaluation.npz' if calendar == 'primary' else extra / f'reference_s{seed}/evaluation.npz'
            with np.load(reference_path) as reference:
                reference_truth = reference['truth']
            for name in ['reference', *CASES]:
                path = reference_path if name == 'reference' else (fit if calendar == 'primary' else extra) / f'{name}_s{seed}/evaluation.npz'
                with np.load(path) as arrays:
                    assert np.array_equal(arrays['truth'], reference_truth, equal_nan=True)
                    measured = statistics(arrays['prediction'], arrays['truth'])
                recorded = next(r for r in records if r['case'] == name and r['seed'] == seed)
                difference = max(abs(measured[k] - recorded[k]) for k in metrics)
                assert difference < 2e-6, (calendar, name, seed, difference)
                assert measured['valid_count'] == recorded['valid_count']
                metric_checks.append(dict(calendar=calendar, case=name, seed=seed,
                    prediction_sha256=sha(path), five_metric_max_difference_mm=difference,
                    valid_count=measured['valid_count'], truth_exact=True))
    report = ROOT / 'docs/research_notes/2026-11-25-precision-final.md'
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report.read_text(encoding='utf-8'))
    for target in links:
        if '://' not in target:
            assert (report.parent / target.split('#')[0]).exists(), target
    manifest = read(combined / 'final_figure_manifest.json')
    assert sha(report) == manifest['report_sha256']
    assert sha(combined / 'station_changes.csv') == manifest['station_csv_sha256']
    for name, expected in manifest['figures'].items():
        assert sha(combined / name) == expected
    receipt = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), frozen_selection_sha256=sha(fit / 'frozen_selection.json'),
        frozen_source_code_and_data_hashes_match=True, coefficient_checks=coefficient_checks,
        metric_checks=metric_checks, report_relative_links_checked=len(links),
        figure_manifest_matches=True, no_training_or_refitting=True,
        evaluation_scope='2020 exposed development calendars; current-time spatial reconstruction')
    write_json(combined / 'final_verification.json', receipt)
    print(json.dumps(dict(exact_coefficients=len(coefficient_checks), evaluations=len(metric_checks),
        relative_links=len(links), max_metric_difference_mm=max(r['five_metric_max_difference_mm'] for r in metric_checks))))


if __name__ == '__main__':
    main()
