"""Bounded local screen for the fixed-support encoder prototype."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .fixed_support import FixedSupportModel, masked_multiscale_summary
from .unet_foundation import Experiment, ROOT
from .residual_experiment import load_pack, save_evaluation


def support_for(ex: Experiment, event: dict, validation: bool) -> torch.Tensor:
    row, hour, c = event['row'], event['hour'], event['context']
    t = 12 + hour
    raw = ex.raw['val_context' if validation else 'train'][row, t - 11:t + 1, c]
    # Validation layouts keep the 410-site training support index space;
    # only the queried targets switch to the 137-site validation geometry.
    geo = ex.train_features.geo
    g = ex.grid.geo
    summary = masked_multiscale_summary(g[:, :2], g[:, 2], geo[c, :2], geo[c, 2],
                                        raw[:, :, 0], torch.isfinite(raw[:, :, 0]))
    return summary.T.reshape(1, summary.shape[1], *ex.shape)


def uv_for(ex: Experiment, geo: torch.Tensor) -> torch.Tensor:
    h, w = ex.shape
    uv = (geo[:, :2] - ex.origin) / 25
    uv = torch.stack([2 * uv[:, 0] / (w - 1) - 1, 2 * uv[:, 1] / (h - 1) - 1], -1)
    return uv.reshape(1, 1, -1, 2)


def predict(ex, model, event, validation=False):
    point = ex.get_features(event, 'mlp', validation)
    geo = ex.val_features.geo if validation else ex.train_features.geo
    return model(point, support_for(ex, event, validation), uv_for(ex, geo))


def fit(ex, seed: int, seconds: float, out: Path):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FixedSupportModel().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); order = rng.permutation(len(ex.fit)); start = time.monotonic(); step = 0; best = float('inf'); best_step = 0
    while time.monotonic() - start < seconds or step == 0:
        e = ex.fit[order[step % len(order)]]; model.train(); opt.zero_grad(set_to_none=True)
        pred = predict(ex, model, e); y = (ex.raw['train'][e['row'], 12 + e['hour'], :, 0] - ex.data.mean[0]) / ex.data.std[0]
        mask = torch.as_tensor(e['target'], device='cuda') & torch.isfinite(y)
        loss = (pred[mask] - y[mask]).square().mean(); loss.backward(); grad = nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step(); step += 1
        if step == 1 or step % 25 == 0:
            model.eval(); vals = []
            with torch.no_grad():
                for m in ex.monitor[:8]:
                    yy = (ex.raw['train'][m['row'], 12 + m['hour'], :, 0] - ex.data.mean[0]) / ex.data.std[0]
                    mm = torch.as_tensor(m['target'], device='cuda') & torch.isfinite(yy)
                    vals.append(float(((predict(ex, model, m)[mm] - yy[mm]).square().mean().sqrt() * ex.data.std[0]).cpu()))
            score = float(np.mean(vals))
            if score < best: best, best_step = score, step
            print(json.dumps({'seed': seed, 'step': step, 'train_rmse_mm': float((loss.sqrt() * ex.data.std[0]).detach()), 'monitor_rmse_mm': score, 'grad': float(grad)}), flush=True)
    out.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), out / 'last.pt')
    with torch.no_grad():
        e = dict(row=0, hour=0, context=ex.data.layouts()[0].tolist()); pred = predict(ex, model.eval(), e, True)
        finite = bool(torch.isfinite(pred).all()); query_rms = float(torch.sqrt(torch.mean((pred[:10] - pred[:10]) ** 2)))
    predictions = []
    with torch.no_grad():
        for layout in ex.data.layouts():
            for row in range(12):
                for hour in range(24):
                    ev = dict(row=row, hour=hour, context=layout.tolist())
                    predictions.append((predict(ex, model, ev, True) * ex.data.std[0] + ex.data.mean[0]).cpu().numpy())
    pack = load_pack(ROOT.parent / '20261124_residual_field_v12/cache/validation.npz')
    save_evaluation(out / 'expanded', np.stack(predictions)[None], pack, ex.data)
    metrics = json.loads((out / 'expanded/metrics.json').read_text(encoding='utf-8'))
    result = {'seed': seed, 'steps': step, 'best_monitor_rmse_mm': best, 'best_step': best_step, 'finite_validation': finite, 'query_self_rms': query_rms, 'expanded_metrics': metrics}
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); return result


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--seconds', type=float, default=60); p.add_argument('--seed', type=int, action='append', default=None); p.add_argument('--out', type=Path, default=ROOT.parent / '20261124_fixed_support_encoder_p14')
    a = p.parse_args(); a.seed = a.seed or [42, 17]; torch.set_num_threads(4); ex = Experiment(); a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'protocol.json').write_text(json.dumps({'kind': 'fixed_support_p14', 'seconds': a.seconds, 'seeds': a.seed, 'confirmation_C_D_read': False}, indent=2), encoding='utf-8')
    results = [fit(ex, s, a.seconds, a.out / f's{s}') for s in a.seed]
    (a.out / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
