"""Trace deployed residual updates and test learning on one fitting hour, locally only."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from retrain_common import ROOT
from retrain_data import Data
from train_new_base import configure_determinism
from pilot_data import schedule
from pilot_models import make_generator
from gnss_field.objectives.scores import energy_score


def main():
    configure_determinism()
    received = Path(json.loads((ROOT / 'collection_complete.json').read_text())['directory'])
    report = {'scope': 'local diagnostic only;one predetermined fitting B hour;no validation selection;no server jobs',
              'device': torch.cuda.get_device_name(), 'products': [], 'gradient_checks': []}
    for fold in (0, 1):
        baseline = np.load(received / f'fold_{fold}/gaussian/predictions.npz')
        for arm in ('gan', 'diffusion'):
            folder = received / f'fold_{fold}' / arm
            result = json.loads((folder / 'result.json').read_text())
            torch.manual_seed(42)
            initial = make_generator(arm).state_dict()
            final = torch.load(folder / result['products']['joint']['selected']['checkpoint'],
                               map_location='cpu', weights_only=True)
            changes = {}
            for prefix in ('net.encoders', 'net.decoders', 'net.output'):
                names = [n for n in initial if n.startswith(prefix) and initial[n].is_floating_point()]
                values = [float((final[n]-initial[n]).abs().max()) for n in names]
                changes[prefix] = dict(changed=sum(v > 0 for v in values), tensors=len(values), maximum=max(values))
            history = [json.loads(v) for v in (folder / 'training.jsonl').read_text().splitlines()]
            for product in ('point', 'joint'):
                arrays = np.load(folder / f'{product}_predictions.npz');mask = arrays['population']
                dm = arrays['mean']-baseline['mean'];dz = arrays['samples']-baseline['samples']
                random_change = dz-dz.mean(1, keepdims=True)
                row = dict(fold=fold, arm=arm, product=product,
                           selected_step=result['products'][product]['selected']['step'],
                           baseline_RMSE_mm=json.loads((received / f'fold_{fold}/gaussian/result.json').read_text())['metrics']['station_macro_rmse_mm'],
                           RMSE_mm=result['products'][product]['metrics']['station_macro_rmse_mm'],
                           mean_max_change_mm=float(abs(dm[mask]).max()),
                           mean_rms_change_mm=float(np.sqrt(np.mean(dm[mask]**2))),
                           random_change_rms_mm=float(np.sqrt(np.mean(random_change.transpose(1,0,2)[:,mask]**2))),
                           weight_changes=changes,
                           gradient_clipped_fraction=float(np.mean([r['gradient_norm'] > 1 for r in history])))
                report['products'].append(row)
    data = Data(device='cuda');fold = 0;data.set_fold(fold)
    hour = schedule(data.lock['splits'][fold]['train_hours'], 72, batch=8)[0, :1]
    ix = data.indices(hour);mask = data.valid[ix] & data.role(fold, 'fit')[None]
    for arm in ('gan', 'diffusion'):
        torch.manual_seed(42);model = make_generator(arm, recompute=True).cuda()
        folder = received / f'fold_{fold}' / arm
        model.load_state_dict(torch.load(folder / 'candidate_0072.pt', map_location='cpu', weights_only=True))
        def objective():
            row = data.sample(model, hour, fold, 4, 800001)
            loss = energy_score(row['samples'], data.truth[ix], mask)
            mean = row['base'] + row['delta'].mean(1)
            mse = ((mean-data.truth[ix])[mask]/10).square().mean()
            return loss, mse, mean
        model.zero_grad(set_to_none=True)
        loss, mse, _ = objective();before_loss = float(loss.detach());before_mse = float(mse.detach())
        loss.backward();gradient_groups = {}
        for prefix in ('net.encoders', 'net.decoders', 'net.output'):
            parameters = [(name, p) for name, p in model.named_parameters() if name.startswith(prefix)]
            norms = {name:float(p.grad.norm()) if p.grad is not None else None for name,p in parameters}
            gradient_groups[prefix] = dict(nonzero=sum(v is not None and v > 0 for v in norms.values()),
                                          parameters=len(parameters), maximum=max(v or 0 for v in norms.values()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        curve = [];started = time.monotonic()
        for step in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss, mse, _ = objective();loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step();curve.append(dict(step=step+1, ES_mm=float(loss.detach()), MSE_standardized=float(mse.detach()), gradient_norm=norm))
        with torch.no_grad():
            loss, mse, mean = objective()
            change = (mean - torch.tensor(np.array(data.new_means[data.indices(hour), fold]), device=data.device))[mask]
        record = dict(arm=arm, fitting_hour=int(hour[0]), fitting_stations=int(mask.sum()),
                      objective='ES only;fixed K4 noise;diagnostic copy of trained step72;8 updates at1e-4',
                      gradients=gradient_groups, before_ES_mm=before_loss, after_ES_mm=float(loss),
                      before_MSE_standardized=before_mse, after_MSE_standardized=float(mse),
                      after_mean_correction_rms_mm=float(change.square().mean().sqrt()),
                      seconds=time.monotonic()-started, curve=curve)
        report['gradient_checks'].append(record)
        print('DIAGNOSTIC', json.dumps({k:v for k,v in record.items() if k not in ('curve','gradients')}), flush=True)
        del model, optimizer;torch.cuda.empty_cache()
    (ROOT / 'near_identity_diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
