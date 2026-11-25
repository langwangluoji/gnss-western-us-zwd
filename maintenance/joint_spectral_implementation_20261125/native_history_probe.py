"""Isolate sensitivity to the added station-time branch, holding the source fixed."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

from precision_analysis import ENGINE
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.station_query import StationData, StationResidual
from tools.terrain_diffusion.station_anomaly import AnomalyData


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', required=True); p.add_argument('--seed', type=int, required=True)
    a = p.parse_args(); torch.set_num_threads(2); torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.18)
    protocol = json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    source = Path(protocol['config']['source']) / f'gru_s{a.seed}/last.pt'
    if sha(source) != protocol['source_weights'][str(a.seed)]: raise ValueError('source changed')
    cls = AnomalyData if protocol['config'].get('history_mode') == 'anomaly' else StationData
    data = cls(Path(protocol['config']['data']))
    model = StationResidual(torch.load(source, weights_only=True), protocol['config']['kind']).cuda().eval()
    weight = a.run/f'{a.case}_s{a.seed}/last.pt'
    model.load_state_dict(torch.load(weight, weights_only=True)); rows = []
    for date in ('2020-01-08', '2020-07-08'):
        start = int(np.searchsorted(data.events.hours, np.datetime64(date, 'h').astype('int64')))
        b = data.batch(np.arange(start, start+2), data.events.layouts[0], True)
        ref = model.reference(b); normal = model(b)
        history = b['station_history'].clone(); history[:, :-1, :, :7] = 0
        missing_past = model(dict(b, station_history=history))
        hook = model.time.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
        no_state = model(b); hook.remove()
        mask = b['mask']; scale = float(data.std[0])
        rms = lambda x: float(x[mask].square().mean().sqrt())*scale
        rows.append(dict(date=date, evaluated_hours=2, valid_count=int(mask.sum()),
            source_micro_rmse_mm=rms(ref-b['target']), candidate_micro_rmse_mm=rms(normal-b['target']),
            correction_rms_mm=rms(normal-ref),
            added_branch_past_removal_change_rms_mm=rms(normal-missing_past),
            added_branch_state_zero_change_rms_mm=rms(normal-no_state)))
    write_json(a.run/f'{a.case}_s{a.seed}/history_sensitivity.json', dict(
        purpose='Posthoc input sensitivity on two fixed development windows; not a retrained temporal ablation',
        source_and_legacy_temporal_inputs_held_fixed=True, code_sha256=sha(Path(__file__)),
        weight_sha256=sha(weight), results=rows))
    print(json.dumps(rows))


if __name__ == '__main__': main()
