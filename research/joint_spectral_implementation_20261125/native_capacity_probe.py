"""Can the added branch fit one continuous training window? CPU-only optimization."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from precision_analysis import ENGINE
sys.path.insert(0, str(ENGINE))
from tools.terrain_diffusion.data import sha, write_json
from tools.terrain_diffusion.station_query import StationData, StationResidual


def main():
    torch.set_num_threads(3); torch.cuda.set_per_process_memory_fraction(.16)
    data_path = ENGINE/'results/20261125_temporal_continuous_data'
    source = ENGINE/'results/20261125_p14_continuous_fusion/gru_s42/last.pt'
    data = StationData(data_path)
    start = int(np.searchsorted(data.events.hours, np.datetime64('2016-07-08', 'h').astype('int64')))
    device_batch = data.batch(np.arange(start, start+2), data.events.layouts[0], False)
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in device_batch.items()}
    std = float(data.std[0]); del device_batch, data; torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = StationResidual(torch.load(source, weights_only=True, map_location='cpu')).train()
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-3, weight_decay=1e-4)
    out = ENGINE/'results/20261125_native_capacity_probe'; out.mkdir(exist_ok=False)
    write_json(out/'protocol.json', dict(purpose='Fixed-window capacity probe, never a formal precision score',
        date='2016-07-08', consecutive_hours=2, condition_layout=0, seed=42, steps=200, lr=1e-3,
        device='cpu', source_sha256=sha(source), data_sha256=sha(data_path/'contract.json'),
        code_sha256=sha(Path(__file__)), reference_frozen=True, trust_region_mm=5,
        formal_training_contract=False, no_development_or_confirmation_labels=True))
    mask = batch['mask']; pair = mask[1:] & mask[:-1]
    with torch.no_grad():
        reference = model.reference(batch)
        if not torch.equal(model(batch), reference): raise ValueError('initial prediction changed')
        errors = (batch['target']-reference)[mask]*std
        reference_rmse = float(errors.square().mean().sqrt())
        bound_floor = float((errors.abs()-5).clamp_min(0).square().mean().sqrt())
    rows = []; begin = time.monotonic()
    for step in range(201):
        prediction = model(batch); loss = (prediction-batch['target'])[mask].square().mean()
        loss = loss+.1*((prediction.diff(dim=0)-batch['target'].diff(dim=0))[pair]).square().mean()
        if step % 20 == 0:
            with torch.no_grad():
                row = dict(step=step, seconds=time.monotonic()-begin,
                    fixed_window_rmse_mm=float((prediction-batch['target'])[mask].square().mean().sqrt())*std,
                    correction_rms_mm=float((prediction-reference)[mask].square().mean().sqrt())*std)
                rows.append(row); write_json(out/'curve.json', rows); print(json.dumps(row), flush=True)
        if step == 200: break
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        if not torch.isfinite(grad): raise ValueError('nonfinite probe gradient')
        optimizer.step()
    write_json(out/'results.json', dict(reference_fixed_window_rmse_mm=reference_rmse,
        unattainable_pointwise_oracle_5mm_bound_floor=bound_floor,
        final_fixed_window_rmse_mm=rows[-1]['fixed_window_rmse_mm'],
        reduction_mm=reference_rmse-rows[-1]['fixed_window_rmse_mm'], trace=rows,
        interpretation='Training capacity only; one repeated continuous2h window, no generalization conclusion'))


if __name__ == '__main__': main()
