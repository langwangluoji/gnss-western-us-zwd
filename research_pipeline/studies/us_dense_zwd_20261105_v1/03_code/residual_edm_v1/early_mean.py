"""Train the stage-2 RAW_CURRENT anchor on 2000--2014 only, with full resume.

This reproduces the no-GAN/no-latent branch of architecture_search.train:
16 paired times per update, grid surface loss + .25 station surface loss,
the original two AdamW groups, clipping at 5, and four complete passes.
No validation or retrospective scorer is called here.
"""
from pathlib import Path
import hashlib
import json
import time

import h5py
import numpy as np
import torch

from .data import STUDY, PAIRED, SOURCE, sha256, write_json
from .training import atomic_checkpoint, rng_state, restore_rng

CONFIG = dict(generator='query_film', critic='none', latent=False)


def early_indices(years, hours, usable, anchor):
    years, hours = np.asarray(years), np.asarray(hours)
    ids = np.arange(19, len(years), dtype=np.int64)
    return ids[(years[ids]>=2000) & (years[ids]<=2014) &
               (years[ids]==years[ids-19]) & np.asarray(usable)[ids] &
               (hours[ids-1]>=anchor)]


def paired_indices(current):
    current = np.asarray(current, dtype=np.int64)
    return np.stack([current, current-1], axis=1).ravel()


def validate_early_checkpoint(saved, lineage=None):
    expected = dict(train_years=[2000,2014], normalization_years=[2000,2014],
                    completed_passes=4, seed=17, config=CONFIG)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f'Early mean checkpoint has incompatible {key}')
    if lineage is not None and saved.get('lineage') != lineage:
        raise ValueError('Early mean lineage changed')
    for key in ('model','mean','std','lineage'):
        if key not in saved:
            raise ValueError(f'Early mean checkpoint missing {key}')
    if not torch.isfinite(saved['mean']).all() or not torch.isfinite(saved['std']).all() or not (saved['std']>0).all():
        raise ValueError('Invalid early-only normalization')


def train_early_mean(root: Path, device='cuda'):
    """Return root/early_mean.pt; resume an interrupted update stream exactly."""
    from architecture_search import SearchGenerator, initialize_shared, state_fingerprint
    from component_ablation import canonical_initialization
    from expanded_architecture_search import complete_pass_orders
    from raw_current_ablation import CurrentObservations
    from moment_superresolution import inputs
    from train_us_gan import load, batch, station_arrays, masked_mse

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root/'early_mean_training'
    directory.mkdir(exist_ok=True)
    output = root/'early_mean.pt'
    device = torch.device(device)
    torch.set_num_threads(2)
    # Only the anchor timestamp is read: no MAE representation or pretrained weights.
    with h5py.File(STUDY/'02_data/processed/expanded_mae_cutoff2018_features.h5','r') as handle:
        anchor = int(handle['hour_id'][0])
    with h5py.File(PAIRED,'r') as handle:
        years, hours = handle['year'][:], handle['hour_id'][:]
    usable = np.load(PAIRED.parent/'usable_pair_hours.npy')
    indices = early_indices(years, hours, usable, anchor)
    if len(indices) != 83232:
        raise ValueError(f'Expected 83232 early eligible times, found {len(indices)}')
    orders = complete_pass_orders(indices, 17, passes=4)
    full_order = np.concatenate(orders)
    unique, counts = np.unique(full_order, return_counts=True)
    if not np.array_equal(unique, np.sort(indices)) or not np.all(counts==4):
        raise ValueError('Early mean sample order is not four complete passes')
    lineage = dict(data_sha256=sha256(PAIRED), source_sha256=sha256(SOURCE),
                   indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                   order_sha256=hashlib.sha256(full_order.tobytes()).hexdigest(),
                   anchor_nominal_hour=anchor, training_hours=len(indices),
                   train_years=[2000,2014], normalization_years=[2000,2014],
                   initialization='canonical_initialization(seed17), no pretrained encoder',
                   time_basis='original nominal hours', passes=4)
    if output.exists():
        validate_early_checkpoint(torch.load(output,map_location='cpu',weights_only=False), lineage)
        return output
    write_json(directory/'protocol.json',dict(lineage=lineage, seed=17, config=CONFIG,
               paired_times_per_update=16, clipping=5., epochs=len(orders),
               optimizer={'core_lr':3e-4,'adapter_lr':1e-4,'weight_decay':.01},
               loss='surface grid MSE + 0.25 surface non-context station MSE; internal Z0/20',
               resume='model, optimizer, RNG, stage, next offset, running metrics every200 updates and stage end'))
    data, _, _, _, _ = load(train_years=tuple(range(2000,2015)), data_path=PAIRED,
                            validation_years=(), chunked_normalization=True)
    archive = CurrentObservations(data, anchor)
    torch.manual_seed(17)
    fine_height = -3000*torch.log(data['fine_height_factor'])
    model = SearchGenerator('query_film', False, fine_height).to(device)
    initial = canonical_initialization(17)
    initialize_shared(model, initial)
    initial_sha = state_fingerprint(initial)
    core_ids = {id(p) for p in model.core.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=model.core.parameters(),lr=3e-4),
        dict(params=[p for p in model.parameters() if id(p) not in core_ids],lr=1e-4)])
    stage, offset, updates, elapsed = 0, 0, 0, 0.
    squared, count, records = 0., 0, []
    resume = directory/'last.pt'
    if resume.exists():
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        if saved['lineage'] != lineage or saved['initial_sha256'] != initial_sha:
            raise ValueError('Early mean resume contract changed')
        torch.testing.assert_close(saved['mean'], data['mean'], rtol=0, atol=0)
        torch.testing.assert_close(saved['std'], data['std'], rtol=0, atol=0)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        stage, offset, updates, elapsed = saved['stage'], saved['offset'], saved['updates'], saved['seconds']
        squared, count, records = saved['squared'], saved['count'], saved['records']
    factor = data['fine_height_factor'].float().to(device)[None,None]
    started = time.monotonic()

    def snapshot():
        atomic_checkpoint(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          rng=rng_state(), lineage=lineage, initial_sha256=initial_sha,
                          mean=data['mean'], std=data['std'], stage=stage, offset=offset,
                          updates=updates, seconds=elapsed+time.monotonic()-started,
                          squared=squared, count=count, records=records))

    while stage < len(orders):
        order = orders[stage]
        if offset == 0:
            torch.manual_seed(17+10000*stage)
        model.train()
        while offset < len(order):
            current = order[offset:offset+16]
            ids = paired_indices(current)
            normalized, baseline, mask = batch(data, ids, device)
            tokens, padding, queries = inputs(archive, data, ids, hours, device)
            result = model(normalized, baseline, tokens, padding, queries)
            prediction = result['field_mm']
            labels = data['target_h0'][ids].to(device)/20
            grid_loss = masked_mse(prediction*factor, labels*factor, mask)
            sp, _, sl, sm = station_arrays(data, result, mask, ids, device)
            loss = grid_loss+.25*masked_mse(sp, sl, sm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Early mean nonfinite loss at stage{stage} offset{offset}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            squared += float(((prediction.detach()-labels)*factor*20).square()[mask].sum())
            count += int(mask.sum())
            offset += len(current)
            updates += 1
            if updates % 200 == 0:
                snapshot()
                print(json.dumps(dict(kind='early_mean',stage=stage+1,offset=offset,
                                      updates=updates,grid_rmse_mm=(squared/max(count,1))**.5)),flush=True)
        records.append(dict(stage=stage+1,updates=updates,grid_rmse_mm=(squared/max(count,1))**.5))
        stage += 1
        offset, squared, count = 0, 0., 0
        snapshot()
        write_json(directory/'training.json',records)
    saved = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 mean=data['mean'], std=data['std'], config=CONFIG, seed=17,
                 train_years=[2000,2014], normalization_years=[2000,2014], completed_passes=4,
                 lineage=lineage, data_sha256=lineage['data_sha256'], initial_sha256=initial_sha,
                 updates=updates, seconds=elapsed+time.monotonic()-started)
    validate_early_checkpoint(saved,lineage)
    atomic_checkpoint(output,saved)
    write_json(directory/'completion.json',dict(checkpoint=str(output),sha256=sha256(output),
               updates=updates,training_hours=len(indices),completed_passes=4,
               normalization_years=[2000,2014],evaluation_labels_used=False))
    return output
