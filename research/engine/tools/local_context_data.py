"""Bounded B2018-only sparse M0 cache for the registered local-context study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time
import numpy as np
import torch

from gnss_field.artifacts import immutable_json, read_json, write_json, sha256

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'results/20261120_architecture_research_v01'
RUN = PROJECT/'results/20261120_local_context_three_arm_v01'
DESIGN = RESEARCH/'training_design_v01.json'


class SparseMeanCache:
    def __init__(self, root, hours, shape, dependencies):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hours = np.asarray(hours, dtype='i8')
        if np.any(np.diff(self.hours) <= 0):
            raise ValueError('Cache hours must be sorted and unique')
        self.lookup = {int(hour): i for i, hour in enumerate(self.hours)}
        immutable_json(self.root/'contract.json', dict(hours=self.hours, shape=shape, dependencies=dependencies))
        path = self.root/'mean.npy'
        mode = 'r+' if path.exists() else 'w+'
        self.array = np.lib.format.open_memmap(path, mode=mode, dtype='f4', shape=(len(hours), *shape))
        cursor = self.root/'cursor.json'
        self.done = np.array(read_json(cursor)['done'], bool) if cursor.exists() else np.zeros(len(hours), bool)
        if (self.root/'complete.json').exists():
            meta = read_json(self.root/'complete.json')
            if not self.done.all() or meta['sha256'] != sha256(path):
                raise ValueError('Completed M0 cache differs from its checksum')

    def contains(self, hour):
        return int(hour) in self.lookup and self.done[self.lookup[int(hour)]]

    def put(self, hour, field):
        if int(hour) not in self.lookup:
            raise ValueError('Hour is outside the locked cache')
        index = self.lookup[int(hour)]
        value = np.asarray(field, dtype='f4').reshape(self.array.shape[1:])
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite M0 field')
        if self.done[index]:
            if not np.array_equal(value, self.array[index]):
                raise ValueError('Completed hour cannot be replaced')
            return
        self.array[index] = value
        self.array.flush()
        self.done[index] = True
        write_json(self.root/'cursor.json', dict(done=self.done))

    def get(self, hour):
        if not self.contains(hour):
            raise ValueError('M0 hour is not complete')
        value = np.array(self.array[self.lookup[int(hour)]])
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite cached M0 field')
        return value

    def finish(self):
        if not self.done.all():
            raise ValueError('Incomplete M0 cache')
        write_json(self.root/'complete.json', dict(hours=len(self.hours), sha256=sha256(self.root/'mean.npy')))


def open_cache(data):
    design = read_json(DESIGN)
    hours = np.sort(np.r_[design['core_hours'], design['inner_hours'], design['outer_hours']])
    return SparseMeanCache(RUN/'M0_cache', hours, data.store.shape, dict(
        design=sha256(DESIGN), foundation=data.contract,
        old_mean=sha256(data.old_mean.filename), extra_mean=sha256(data.extra_mean.filename)))


def build_cache():
    from local_probe_diagnostics import ProbeData, MEAN
    from gnss_field.data import ContextBuilder
    from gnss_field.models.mean import MeanField
    from gnss_field.training.fields import predict_patches
    from gnss_field.runtime import configure
    configure(type('Runtime', (), {'cpus': 2})(), 42)
    RUN.mkdir(parents=True, exist_ok=True)
    data = ProbeData()
    cache = open_cache(data)
    net = None
    builder = ContextBuilder(data.store)
    started = time.monotonic()
    generated = reused = 0
    for j, hour in enumerate(cache.hours):
        if (RUN/'PAUSE').exists():
            write_json(RUN/'cache_status.json', dict(status='PAUSED', completed=int(cache.done.sum())))
            return
        if cache.contains(hour):
            continue
        if hour in data.hours:
            field = data.frozen_field(hour)[0]
            reused += 1
        else:
            if net is None:
                net = MeanField().cuda().eval().requires_grad_(False)
                net.load_state_dict(torch.load(MEAN, map_location='cuda', weights_only=False)['ema'])
            with torch.inference_mode():
                ctx = builder.build_context([hour], device='cuda')
                field = predict_patches(net, ctx, data.store, chunk=4)[0][0, 0].cpu().numpy()
            generated += 1
        cache.put(hour, field)
        if (j+1) % 12 == 0 or j+1 == len(cache.hours):
            status = dict(status='BUILDING', completed=int(cache.done.sum()), total=len(cache.hours),
                generated_this_session=generated, reused_this_session=reused,
                elapsed_seconds=time.monotonic()-started, updated=time.time())
            write_json(RUN/'cache_status.json', status)
            print(status, flush=True)
    cache.finish()
    assert sha256(MEAN) == data.mean_hash
    write_json(RUN/'cache_status.json', dict(status='COMPLETE', completed=len(cache.hours),
        generated_this_session=generated, reused_this_session=reused,
        elapsed_seconds=time.monotonic()-started, updated=time.time()))
    print('M0_CACHE_COMPLETE', flush=True)


if __name__ == '__main__':
    build_cache()
