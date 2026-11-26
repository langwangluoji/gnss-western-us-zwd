"""Fixed-date training/development residual probe; not checkpoint selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from precision_analysis import ENGINE
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.query_precision import QueryResidual
from tools.terrain_diffusion.station_query import StationData, StationResidual
from tools.terrain_diffusion.warm_joint_experiment import WarmData


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--family', choices=['query', 'station'], required=True)
    a = p.parse_args()
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.18)
    protocol = json.loads((a.run / 'protocol.json').read_text(encoding='utf-8'))
    data_path = Path(protocol['config']['data'])
    source_path = Path(protocol['config']['source']) / f'gru_s{a.seed}/last.pt'
    if sha(source_path) != protocol['source_weights'][str(a.seed)]:
        raise ValueError('source weight hash changed')
    if sha(data_path / 'contract.json') != protocol['data_sha256']:
        raise ValueError('data contract changed')
    source = torch.load(source_path, weights_only=True)
    if a.family == 'station':
        if protocol['config'].get('history_mode') == 'anomaly':
            from tools.terrain_diffusion.station_anomaly import AnomalyData
            data = AnomalyData(data_path)
        else:
            data = StationData(data_path)
        model = StationResidual(source, protocol['config']['kind']).cuda().eval()
    else:
        data = WarmData(data_path)
        model = QueryResidual(source).cuda().eval()
    weight = a.run / f'{a.case}_s{a.seed}/last.pt'
    model.load_state_dict(torch.load(weight, weights_only=True))
    rows = []
    for year in (2016, 2017, 2018, 2019, 2020):
        for month in (1, 7):
            date = f'{year}-{month:02d}-08'
            start = int(np.searchsorted(data.events.hours, np.datetime64(date, 'h').astype('int64')))
            pieces = []
            for offset in range(0, 24, 2):
                b = data.batch(np.arange(start+offset, start+offset+2), data.events.layouts[0], year == 2020)
                reference = model.reference(b); prediction = model(b)
                error = (b['target']-reference)[b['mask']] * data.std[0]
                correction = (prediction-reference)[b['mask']] * data.std[0]
                pieces.append((error.cpu().numpy().astype('float64'), correction.cpu().numpy().astype('float64')))
            e = np.concatenate([x[0] for x in pieces]); c = np.concatenate([x[1] for x in pieces])
            rows.append(dict(date=date, role='exposed_2020_development' if year == 2020 else 'training_station_diagnostic',
                valid_count=len(e), reference_micro_rmse_mm=float(np.sqrt(np.mean(e*e))),
                candidate_micro_rmse_mm=float(np.sqrt(np.mean((e-c)**2))),
                correction_rms_mm=float(np.sqrt(np.mean(c*c))),
                correction_std_mm=float(np.std(c)),
                correction_residual_correlation=float(np.corrcoef(e, c)[0, 1]) if np.std(c) else None))
    write_json(a.run / f'{a.case}_s{a.seed}/fit_probe.json', dict(
        role='Posthoc fixed-date diagnostics, never checkpoint selection or independent validation',
        code_sha256=sha(Path(__file__)), weight_sha256=sha(weight), source_sha256=sha(source_path),
        data_sha256=sha(data_path / 'contract.json'), layout=0, hours_per_date=24, batch_hours=2,
        training_targets_exclude_condition_physical_groups=True, results=rows))
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
