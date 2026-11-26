"""Matched encoder-width experiment on the previously audited local data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from simple_weather_unet_pilot import (PROJECT, ROOT as DATA_ROOT, PilotData, score, predict, metrics)
from gnss_field.models.conditioned_unet import ConditionedUNet
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256

ROOT = PROJECT/'results/20261123_enhanced_encoder_generative_v01'


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(data, seed, steps):
    out = ROOT/f'encoder_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        return read_json(out/'complete.json')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ConditionedUNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rows = data.rows('train')
    inner = data.rows('inner')
    fixed = rows[np.rint(np.linspace(0, len(rows)-1, 48)).astype(int)]
    fit, val = data.a['split_fit'], data.a['split_inner']
    rng = np.random.default_rng(seed)
    schedule = np.concatenate([rng.permutation(rows) for _ in range(math.ceil(steps*8/len(rows)))])[:steps*8].reshape(steps, 8)
    start, history, best = 0, [], float('inf')
    if (out/'last.pt').exists():
        saved = torch.load(out/'last.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, history, best = saved['step'], saved['history'], saved['best']
    else:
        initial = score(model, data, inner, True, val)
        best = initial['macro_rmse']
        history.append(dict(step=0, inner=initial, train=score(model, data, fixed, True, fit)))
        atomic_checkpoint(out/'best.pt', dict(model=model.state_dict(), step=0))
    started = time.monotonic()
    losses = []
    for step in range(start+1, steps+1):
        model.train()
        batch = schedule[step-1]
        x, base = data.batch(batch, True)
        pred = data.query((base+10*model(x))[:, 0], fit)
        mask = data.mask[batch][:, fit]
        loss = (((pred-data.labels[batch][:, fit])/10).square()*mask).sum()/mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError('Invalid encoder optimization')
        lr = .00002+.00098*.5*(1+math.cos(math.pi*(step-1)/(steps-1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        losses.append(float(loss.detach())*100)
        if step % 100 == 0 or step == steps or (ROOT/'PAUSE').exists():
            record = dict(step=step, inner=score(model, data, inner, True, val),
                          train=score(model, data, fixed, True, fit), loss_mm2=float(np.mean(losses)),
                          lr=lr, seconds=time.monotonic()-started)
            if record['inner']['macro_rmse'] < best:
                best = record['inner']['macro_rmse']
                atomic_checkpoint(out/'best.pt', dict(model=model.state_dict(), step=step))
            history.append(record)
            losses = []
            atomic_checkpoint(out/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              step=step, best=best, history=history, schedule=schedule))
            write_json(out/'curve.json', history)
            write_json(ROOT/'status.json', dict(stage='ENCODER_TRAINING', seed=seed, updated=time.time(), **record))
            print(json.dumps(dict(stage='encoder', seed=seed, **record)), flush=True)
            if (ROOT/'PAUSE').exists():
                raise RuntimeError('Paused after committed encoder update')
    selected = torch.load(out/'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(selected['model'])
    result = {}
    for cohort, indices, ids in [('spatial', fixed, data.a['split_outer']),
                                ('temporal', data.rows('outer'), fit),
                                ('spatial_temporal', data.rows('outer'), data.a['split_outer']),
                                ('C2019_development', data.rows('development'), data.a['C_ids'])]:
        prediction = predict(model, data, indices, True, ids)
        y = data.a['labels'][np.ix_(indices, ids)]
        mask = data.a['mask'][np.ix_(indices, ids)]
        result[cohort] = metrics(prediction, y, mask)
        np.savez(out/f'{cohort}_predictions.npz', prediction=prediction, target=y, mask=mask,
                 hour_ids=data.hours[indices], station_ids=ids)
    receipt = dict(seed=seed, selected_step=selected['step'], best_inner_rmse=best,
                   parameters=sum(p.numel() for p in model.parameters()), results=result,
                   checkpoint_sha256=sha256(out/'best.pt'), seconds=time.monotonic()-started)
    write_json(out/'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    source = read_json(DATA_ROOT/'data_complete.json')
    if source['sha256'] != sha256(DATA_ROOT/'data.npz'):
        raise ValueError('Audited input cache changed')
    immutable_json(ROOT/'encoder.lock.json', dict(data=source['sha256'], steps=args.steps,
            encoder_widths=[32, 64, 128], decoder_widths=[32, 16], seeds=[42, 17],
            script=sha256(__file__), model=sha256(PROJECT/'src/gnss_field/models/conditioned_unet.py'),
            input_source='user confirmed existing A GNSS / ERA5 sampled variables / terrain',
            evaluation='previously exposed development populations; D2020 sealed'))
    data = PilotData()
    results = [train(data, seed, args.steps) for seed in (42, 17)]
    write_json(ROOT/'encoder_results.json', dict(status='COMPLETE', results=results))
    print('ENCODER_PAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
