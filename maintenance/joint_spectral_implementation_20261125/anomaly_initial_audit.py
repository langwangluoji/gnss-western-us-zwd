"""Real GNSS input-contract and exact reference audit before anomaly training."""
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
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.30)
    source = ENGINE / 'results/20261125_p14_continuous_fusion'
    data_path = ENGINE / 'results/20261125_temporal_continuous_data'
    data = AnomalyData(data_path); records = []
    state = torch.load(source / 'gru_s42/last.pt', weights_only=True)
    model = StationResidual(state).cuda().eval()
    for date, development in [('2016-01-08', False), ('2020-01-08', True)]:
        start = int(np.searchsorted(data.events.hours, np.datetime64(date, 'h').astype('int64')))
        anchors = np.arange(start, start+12); context = data.events.layouts[0]
        raw = StationData.batch(data, anchors, context, development)
        b = data.batch(anchors, context, development)
        for key in ('legacy_history', 'point', 'support', 'edges', 'target', 'mask'):
            if not torch.equal(raw[key], b[key]):
                raise ValueError(f'anomaly altered preserved reference input {key}')
        prediction = model(b); reference = model.reference(raw)
        if not torch.equal(prediction, reference):
            raise ValueError('anomaly zero initialization changed original reference')
        if not torch.isfinite(b['station_history']).all():
            raise ValueError('nonfinite anomaly tokens')
        difference = None
        if development:
            saved = np.load(source / 'gru_s42/evaluation.npz')['prediction'][0, 0, :12]
            actual = (prediction*data.std[0]+data.mean[0]).cpu().numpy()
            difference = float(np.max(np.abs(actual-saved)))
            if difference > 1e-4:
                raise ValueError(f'original saved reference not reproduced: max_difference_mm={difference}; first={actual[0,:5]}; saved={saved[0,:5]}')
        records.append(dict(date=date, development=development, exact_initialized_reference=True,
            preserved_inputs_exact=True, history_shape=list(b['station_history'].shape),
            anomaly_rms=float(b['station_history'][..., 0].square().mean().sqrt()),
            valid_anomaly_fraction=float(b['station_history'][..., 3].mean()),
            saved_original_prediction_max_difference_mm=difference))
    out = ENGINE / 'results/20261125_station_anomaly_audit'
    out.mkdir(exist_ok=True)
    write_json(out / 'verification.json', dict(code_sha256=sha(Path(__file__)),
        anomaly_code_sha256=sha(ENGINE / 'tools/terrain_diffusion/station_anomaly.py'),
        source_sha256=sha(source / 'gru_s42/last.pt'), data_sha256=sha(data_path / 'contract.json'),
        records=records, confirmation_C_D_labels_used=False))
    print(records)


if __name__ == '__main__':
    main()
