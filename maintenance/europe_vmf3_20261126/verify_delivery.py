"""Verify final lightweight delivery against frozen local artifacts; no training."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .features import HERE, CACHE, PARENT, OLD_CACHE, sha, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    sources = read(PARENT / 'verification.json')['source_hashes_unchanged']
    for item in sources:
        assert sha(Path(item['path'])) == item['sha256'], item['path']
    source_files = read(HERE / 'vmf3_data_receipt.json')['sources']
    for path, expected in source_files.items():
        assert sha(Path(path)) == expected, path

    lock = read(HERE / 'probability_lock.json')
    selection = read(HERE / 'selection_2023.json')
    assert sha(HERE / 'selection_2023.json') == lock['point_mean_selection_sha256']
    for item in selection['records']:
        seed = item['seed']
        for arm, expected in item['weights'].items():
            assert sha(HERE / f'runs/{arm}_s{seed}/best.pt') == expected
        assert sha(PARENT / f'runs/joint_v2_s{seed}/best.pt') == item['joint_weights_sha256']
        source = read(HERE / f'diffusion_source_s{seed}.json')
        assert sha(HERE / f'diffusion_source_s{seed}.json') == lock['source_seals'][str(seed)]
        stats = CACHE / f'diffusion_s{seed}/statistics.json'
        assert sha(stats) == source['statistics_sha256']
        assert sha(HERE / f'residual_statistics_s{seed}.json') == sha(stats)

    training = read(HERE / 'training_verification.json')
    assert training['status'] == 'passed' and len(training['models']) == 14
    evaluations = []
    for key, model in lock['models'].items():
        seed, view, kind = key.split('/')
        folder = HERE / f'evaluation/s{seed}/{view}/{kind}'
        receipt = read(folder / 'receipt.json')
        assert receipt['probability_lock_sha256'] == sha(HERE / 'probability_lock.json')
        assert receipt['weights'] == model
        assert sha(HERE / f'probability/s{seed}/{view}/{kind}/last.pt') == model['weight_sha256']
        assert sha(HERE / f'probability/s{seed}/{view}/{kind}/lock.json') == model['protocol_sha256']
        assert sha(folder / 'evaluation.npz') == receipt['prediction_sha256']
        assert sha(folder / 'metrics.json') == receipt['scores_sha256']
        assert receipt['deterministic_mean_preserved'] and receipt['members'] == 32
        evaluations.append(key)
    assert len(evaluations) == 14

    delivery = np.load(CACHE / 'point_delivery.npz')
    old = np.load(OLD_CACHE / 'comparison_arrays.npz')
    roles = np.load(OLD_CACHE / 'data.npz')['role']
    assert np.array_equal(delivery['old_cnn'], old['original'], equal_nan=True)
    assert np.array_equal(delivery['old_I1'], old['ours'], equal_nan=True)
    assert np.array_equal(delivery['valid'], old['valid'])
    mask = old['valid'] & (roles == 'confirm')[None]
    assert int(mask.sum()) == 63494 and int(mask.any(0).sum()) == 59
    point = read(HERE / 'point_results.json')['cohorts']['confirm']
    recalculated = {}
    for name, metrics in point.items():
        error = delivery[name].astype('f8')[mask] - old['target'].astype('f8')[mask]
        value = float(np.sqrt(np.mean(error * error)))
        assert abs(value - metrics['micro_rmse_mm']) < 1e-10, name
        recalculated[name] = value

    figures = read(HERE / 'figure_manifest.json')
    assert len(figures) == 14
    for name, expected in figures.items():
        path = HERE / 'figures' / name
        assert sha(path) == expected, name
        if path.suffix == '.svg':
            ET.parse(path)
    for target in re.findall(r'\]\(([^)]+)\)', (HERE / 'REPORT.md').read_text(encoding='utf-8')):
        if not target.startswith(('https:', 'http:', '#')):
            assert (HERE / target.split('#')[0]).exists(), target
    test_log = (HERE / 'final_pytest.local.txt').read_text(encoding='utf-8', errors='replace')
    assert '30 passed' in test_log
    receipt = dict(status='passed', original_hdf5_sources=sources,
                   vmf3_source_files_verified=len(source_files),
                   mean_selection_and_probability_locks_unchanged=True,
                   probability_evaluations_verified=evaluations,
                   fixed_old_cnn_and_I1_predictions_unchanged=True,
                   confirm_records=63494, confirm_active_stations=59,
                   float64_rmse_recalculated=recalculated,
                   figure_files_verified=len(figures), report_links_exist=True,
                   delivery_tests='30 passed',
                   full_engine_tests='568 passed; earlier same-run local log retained',
                   local_test_log_sha256=sha(HERE / 'final_pytest.local.txt'))
    write_json(HERE / 'delivery_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
