"""Continue P15 support-sampling fits and evaluate with the p14 expanded contract."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .optimization_campaign import Campaign
from .fixed_support import FixedSupportModel
from .fixed_support_experiment import predict
from .unet_foundation import ROOT
from .residual_experiment import load_pack, save_evaluation


def continue_fit(c: Campaign, seed: int, seconds: float, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = FixedSupportModel().cuda()
    previous = out / 'best.pt'
    if previous.exists():
        model.load_state_dict(torch.load(previous, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); order = rng.permutation(len(c.fit_events))
    start = time.monotonic(); step = 0; best = float('inf'); history = []
    while time.monotonic() - start < seconds or step == 0:
        e = c.fit_events[order[step % len(order)]]; model.train(); opt.zero_grad(set_to_none=True)
        pred = c.predict(model, e); y = (c.ex.raw['train'][e['row'], 12 + e['hour'], :, 0] - c.ex.data.mean[0]) / c.ex.data.std[0]
        mask = torch.as_tensor(e['target'], device='cuda') & torch.isfinite(y)
        loss = (pred[mask] - y[mask]).square().mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step(); step += 1
        if step == 1 or step % 50 == 0:
            model.eval(); vals = []
            with torch.no_grad():
                for q in c.monitor[:8]:
                    yy = (c.ex.raw['train'][q['row'], 12 + q['hour'], :, 0] - c.ex.data.mean[0]) / c.ex.data.std[0]
                    mm = torch.as_tensor(q['target'], device='cuda') & torch.isfinite(yy)
                    vals.append(float(((c.predict(model, q)[mm] - yy[mm]).square().mean().sqrt() * c.ex.data.std[0]).cpu()))
            score = float(np.mean(vals)); history.append({'step': step, 'monitor_rmse_mm': score, 'train_rmse_mm': float((loss.sqrt() * c.ex.data.std[0]).detach())})
            if score < best:
                best = score; torch.save(model.state_dict(), out / 'best_continued.pt')
            print(json.dumps({'sampling': c.sampling, 'seed': seed, 'step': step, 'monitor_rmse_mm': score}), flush=True)
    model.load_state_dict(torch.load(out / 'best_continued.pt', weights_only=True)); model.eval()
    predictions = []
    with torch.no_grad():
        for layout in c.ex.data.layouts():
            for row in range(12):
                for hour in range(24):
                    ev = dict(row=row, hour=hour, context=layout.tolist())
                    predictions.append((predict(c.ex, model, ev, True)).cpu().numpy() * c.ex.data.std[0].item() + c.ex.data.mean[0].item())
    pack = load_pack(ROOT.parent / '20261124_residual_field_v12/cache/validation.npz')
    save_evaluation(out / 'expanded_continued', np.stack(predictions)[None], pack, c.ex.data)
    metrics = json.loads((out / 'expanded_continued/metrics.json').read_text(encoding='utf-8'))
    result = {'sampling': c.sampling, 'seed': seed, 'continued_seconds': seconds, 'continued_steps': step, 'best_monitor_rmse_mm': best, 'expanded_metrics': metrics}
    (out / 'continued_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--sampling', choices=['random', 'hole100', 'hole150', 'distance_stratified'], required=True); p.add_argument('--seconds', type=float, default=600); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); c = Campaign(a.sampling, 'fixed3000'); rows = [continue_fit(c, s, a.seconds, a.out / f's{s}') for s in (42, 17)]
    (a.out / 'formal_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
