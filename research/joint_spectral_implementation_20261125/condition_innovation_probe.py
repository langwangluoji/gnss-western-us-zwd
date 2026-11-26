"""Test information in visible-station model errors, without fitting a correction."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

from precision_analysis import ENGINE
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.p14_fusion import FusionData, P14Fusion


def main():
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.25)
    path = ENGINE/'results/20261125_temporal_continuous_data'
    source = ENGINE/'results/20261125_p14_continuous_fusion'
    data = FusionData(path); geo = data.events.geo; rows = []
    for seed in (42, 17):
        model = P14Fusion('gru').cuda().eval()
        model.load_state_dict(torch.load(source/f'gru_s{seed}/last.pt', weights_only=True))
        for year in (2016, 2019, 2020):
            for month in (1, 7):
                date = f'{year}-{month:02d}-08'
                start = int(np.searchsorted(data.events.hours, np.datetime64(date, 'h').astype('int64')))
                anchors = np.arange(start, start+12); c = data.events.layouts[0]
                full = data.batch(anchors, c, False); full_prediction = model(full)
                index = torch.as_tensor(c, device='cuda')
                observed = data.values[torch.as_tensor(anchors, device='cuda')[:, None], index[None], 0]
                valid = torch.isfinite(observed)
                innovation = torch.where(valid, observed-(full_prediction[:, index]*data.std[0]+data.mean[0]), 0)
                development = year == 2020
                b = data.batch(anchors, c, True) if development else full
                prediction = model(b) if development else full_prediction
                if development and month == 1:
                    old = np.load(source/f'gru_s{seed}/evaluation.npz')['prediction'][0, 0, :12]
                    discrepancy = float(np.max(np.abs((prediction*data.std[0]+data.mean[0]).cpu().numpy()-old)))
                    if discrepancy > 1e-4: raise ValueError(f'original reference differs by {discrepancy} mm')
                qxy = torch.as_tensor(geo['dev_xy' if development else 'xy'], device='cuda', dtype=torch.float32)
                qheight = torch.as_tensor(geo['dev_height' if development else 'height'], device='cuda', dtype=torch.float32)
                cxy = torch.as_tensor(geo['xy'][c], device='cuda', dtype=torch.float32)
                cheight = torch.as_tensor(geo['height'][c], device='cuda', dtype=torch.float32)
                distance = torch.cdist(qxy, cxy, compute_mode='donot_use_mm_for_euclid_dist')
                error = ((b['target']-prediction)*data.std[0])[b['mask']].cpu().numpy().astype('float64')
                features = []
                for radius in (50, 150, 300):
                    weights = torch.exp(-.5*(distance/radius).square())[None]*valid[:, None]
                    numerator = (weights*innovation[:, None]*torch.exp(cheight[None, None]/3000)).sum(-1)
                    field = numerator/weights.sum(-1).clamp_min(1e-8)*torch.exp(-qheight[None]/3000)
                    feature = field[b['mask']].cpu().numpy().astype('float64')
                    features.append(dict(radius_km=radius, field_rms_mm=float(np.sqrt(np.mean(feature**2))),
                        residual_correlation=float(np.corrcoef(error, feature)[0, 1]) if np.std(feature) else None,
                        oracle_scale_diagnostic_only=float(np.dot(error, feature)/np.dot(feature, feature)) if np.dot(feature, feature) else None))
                row = dict(seed=seed, date=date, role='exposed_development' if development else 'training_window_diagnostic',
                    valid_targets=len(error), reference_micro_rmse_mm=float(np.sqrt(np.mean(error**2))),
                    visible_innovation_rms_mm=float(innovation[valid].square().mean().sqrt()), features=features)
                rows.append(row); print(json.dumps(row), flush=True)
        del model; torch.cuda.empty_cache()
    out = ENGINE/'results/20261125_condition_innovation_probe'; out.mkdir(exist_ok=True)
    write_json(out/'results.json', dict(code_sha256=sha(Path(__file__)), data_sha256=sha(path/'contract.json'),
        source_weights={str(s): sha(source/f'gru_s{s}/last.pt') for s in (42, 17)},
        mechanism='visible ZWD minus frozen GRU+p14 prediction at the SAME visible station; transport errors to query height',
        diagnostic_only=True, no_corrector_fitted=True, no_development_coefficients_used_for_predictions=True,
        confirmation_C_D_labels_used=False, results=rows))


if __name__ == '__main__': main()
