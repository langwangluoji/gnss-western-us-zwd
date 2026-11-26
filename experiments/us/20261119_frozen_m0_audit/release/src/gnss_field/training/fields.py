"""Full-contributor field prediction and explicitly built monthly mean caches."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint

from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json
from gnss_field.geometry import mosaic, patch_dictionary


def predict_patches(model, context, store, *, chunk=8, residual=False, scale=1.):
    origins = store.grid['origins']
    outputs, features = [], []
    b = len(context.hours)
    for i in range(0, len(origins), chunk):
        selected = origins[i:i+chunk]
        batch = patch_dictionary(context, selected, store.grid['origin'], scale)
        if model.training and torch.is_grad_enabled():
            result = checkpoint(lambda _, values=batch: model(values), batch['coarse'], use_reentrant=False)
        else:
            result = model(batch)
        r, f = result
        outputs.append(r.reshape(len(selected), b, *r.shape[1:]).transpose(0, 1))
        if f is not None:
            features.append(f.reshape(len(selected), b, *f.shape[1:]).transpose(0, 1))
    field = mosaic(torch.cat(outputs, 1), origins, store.shape)
    feat = mosaic(torch.cat(features, 1), origins, store.shape) if features else None
    if residual:
        if context.mean is None:
            raise ValueError('Pixel residual model needs a frozen full mean field')
        field = context.mean + scale*field
    return field, feat


class MeanCache:
    def __init__(self, store, directory, model_hash, scenario_hash=None, context_policy='four_frame'):
        if scenario_hash is None:
            from gnss_field.data import InputScenario
            scenario_hash = InputScenario().identity
        self.store = store
        context_hash = fingerprint({name: meta for name, meta in store.manifest['files'].items() if name.startswith('A/')})
        self.contract = fingerprint(dict(context=context_hash, stats=store.stats, grid=store.grid_id,
                                         mean=model_hash, scenario=scenario_hash, context_policy=context_policy,
                                         schema='mean_cache_v2'))
        self.root = Path(directory)/self.contract
        self.model_hash = model_hash
        self.scenario_hash = scenario_hash
        self.arrays = {}

    def get(self, hours, *, device='cpu'):
        result = []
        for hour in np.asarray(hours, dtype='i8'):
            month = str(pd.to_datetime(hour, unit='h').strftime('%Y-%m'))
            path = self.root/f'{month}.npy'
            meta = read_json(path.with_suffix('.json'))
            if not meta['complete'] or meta['contract'] != self.contract:
                raise ValueError('Mean cache is incomplete or mismatched')
            if month not in self.arrays:
                if sha256(path) != meta['sha256']:
                    raise ValueError('Mean cache checksum mismatch')
                self.arrays[month] = np.load(path, mmap_mode='r')
            result.append(self.arrays[month][int(hour)-meta['start']])
        return torch.as_tensor(np.array(result), device=device, dtype=torch.float32)

    @torch.no_grad()
    def build(self, model, builder, hours, *, device='cuda', chunk=8, pause_check=lambda: None, scenario=None):
        from gnss_field.data import InputScenario
        scenario = InputScenario() if scenario is None else scenario
        if scenario.identity != self.scenario_hash:
            raise ValueError('Mean cache scenario differs from actual input masks')
        self.root.mkdir(parents=True, exist_ok=True)
        immutable_json(self.root/'contract.json', dict(contract=self.contract, model_hash=self.model_hash))
        model.eval()
        hours = np.asarray(hours, dtype='i8')
        months = pd.to_datetime(hours, unit='h').strftime('%Y-%m')
        for month in sorted(set(months)):
            bounds = self.store.manifest['months'][month]
            first, stop = bounds['start'], bounds['stop']
            # Complete calendar-month cache so arbitrary development hours share one artifact.
            month_hours = np.arange(max(first, self.store.manifest['start']+18), stop)
            path = self.root/f'{month}.npy'
            if path.exists() and path.with_suffix('.json').exists():
                if read_json(path.with_suffix('.json'))['contract'] != self.contract:
                    raise ValueError('Mean cache metadata mismatch')
                continue
            partial = path.with_suffix('.building.npy')
            cursor_file = path.with_suffix('.cursor.json')
            if partial.exists() and cursor_file.exists():
                cursor = read_json(cursor_file)
                if cursor['contract'] != self.contract:
                    raise ValueError('Partial cache contract mismatch')
                array = np.load(partial, mmap_mode='r+')
                done = cursor['done']
            else:
                array = np.lib.format.open_memmap(partial, mode='w+', dtype='f4', shape=(stop-first, 1, *self.store.shape))
                array[:] = np.nan
                done = 0
            for i in range(done, len(month_hours)):
                pause_check()
                ctx = builder.build_context([month_hours[i]], scenario, device=device)
                value = predict_patches(model, ctx, self.store, chunk=chunk)[0]
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite frozen mean field')
                array[int(month_hours[i])-first] = value[0].cpu().numpy()
                if (i+1) % 24 == 0 or i+1 == len(month_hours):
                    array.flush()
                    write_json(cursor_file, dict(contract=self.contract, done=i+1))
            array.flush()
            del array
            partial.replace(path)
            write_json(path.with_suffix('.json'), dict(contract=self.contract, complete=True, start=first,
                       stop=stop, sha256=sha256(path)))
            cursor_file.unlink(missing_ok=True)
