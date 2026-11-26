"""Audit completed probe outputs and publish F0/S0-separated evaluation tables."""
import csv
import json
import shutil

import numpy as np
import torch

from local_probe_diagnostics import ProbeData, OUTPUT
from gnss_field.artifacts import read_json, write_json
from gnss_field.evaluation import evaluate_arrays
from gnss_field.training.engine import HourCursor
from gnss_field.training.prediction import safe_fine_sites


def main():
    data = ProbeData()
    summary = read_json(OUTPUT/'summary.json')
    lock = read_json(OUTPUT/'probe.lock.json')
    arrays = data.arrays(2018)
    development = data.arrays(2019)
    ids = development['station_ids']
    metadata = data.development.stations.iloc[ids].reset_index(drop=True)
    f0 = np.isfinite(development['truth']) & data.development.query_valid[ids]
    archive = OUTPUT/'evaluation_v01_support_only'
    archive.mkdir(exist_ok=True)
    if not (archive/'summary.json').exists():
        shutil.copyfile(OUTPUT/'summary.json', archive/'summary.json')
    counts = []
    for row in summary['folds']:
        fold = row['fold']
        folder = OUTPUT/f'fold_{fold}'
        with np.load(folder/'forecast_2019.npz') as z:
            metrics = {}
            for name, prediction, coarse in [('model', z['prediction'], z['coarsened_prediction']),
                                             ('M0', z['baseline'], development['coarse']),
                                             ('A_IDW', z['idw'], None)]:
                value = evaluate_arrays(z['hours'], ids, z['truth'], prediction, f0, z['population'],
                                        metadata=metadata, coarsened_prediction=coarse,
                                        safe_T=safe_fine_sites(data.development, ids))
                assert value['primary'] == row['results']['2019'][name]['primary']
                metrics[name] = value
        if not (archive/f'fold_{fold}_results.json').exists():
            shutil.copyfile(folder/'results.json', archive/f'fold_{fold}_results.json')
        row['results']['2019'] = metrics
        row['evaluation_revision'] = 'F0=finite label and query geometry; S0=locked A support intersection'
        write_json(folder/'results.json', row)
        checkpoint = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        cursor = HourCursor(np.array(lock['training_hours']), 42)
        valid = arrays['valid'] & (data.assignment != fold)
        cumulative, per_step = 0, {}
        for step in range(1, checkpoint['step']+1):
            batch = cursor.take(32)
            n = int(valid[np.searchsorted(arrays['hours'], batch)].sum())
            cumulative += n
            per_step[step] = (n, cumulative)
        np.testing.assert_array_equal(cursor.order, checkpoint['cursor']['order'])
        assert cursor.position == checkpoint['cursor']['position']
        records = []
        for line in (folder/'training.jsonl').read_text().splitlines():
            record = json.loads(line)
            current, total = per_step[record['step']]
            record['supervised_station_hours'] = current
            record['cumulative_supervised_station_hours'] = total
            record['supervision_count_source'] = 'replayed locked HourCursor and fitted-B qualification masks'
            records.append(record)
        (folder/'training_telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf8')
        counts.append(dict(fold=fold, updates=checkpoint['step'], sampled_hours=32*checkpoint['step'],
                           supervised_station_hours=cumulative, cursor_verified=True))
    summary['evaluation_revision'] = '2019 full F0 and supported S0 separated; primary scores independently matched'
    summary['supervision_audit'] = counts
    write_json(OUTPUT/'summary.json', summary)
    write_json(OUTPUT/'supervision_audit.json', counts)
    with (OUTPUT/'comparison_2019.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fold', 'selected_step', 'arm', 'population', 'stations', 'station_hours', 'RMSE_mm', 'MAE_mm', 'bias_mm'])
        for row in summary['folds']:
            for name, bundle in row['results']['2019'].items():
                for population in ('full', 'primary', 'unsupported'):
                    m = bundle[population]
                    writer.writerow([row['fold'], row['selection']['selected']['step'], name, population,
                                     m['n_stations'], m['n_station_hours'], m['station_macro_rmse_mm'],
                                     m['station_macro_mae_mm'], m['station_macro_bias_mm']])
    print(json.dumps(dict(verification='PASS', F0_stations=metrics['M0']['full']['n_stations'],
                          S0_stations=metrics['M0']['primary']['n_stations'], supervision=counts), indent=2))


if __name__ == '__main__':
    main()
