"""Rebuild one preregistered A-only context and test downloaded models at K128."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from grid_pair_data import GridStudy, local_runtime, feature69
from local_context_features import LocalFeatureBuilder, VertexQuery
from gnss_field.artifacts import sha256
from gnss_field.data import PackedStore
from gnss_field.data.context import cycles
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.models.grid_generation import GenerationSystem, gaussian_prior_field
from gnss_field.models.diffusion import keyed_noise
from gnss_field.runtime import configure


def main(hour=None):
    project = Path(__file__).resolve().parents[1]
    root = project/'results/20261122_generation_postrun_research_v01'
    source = root/'downloaded'
    configure(SimpleNamespace(cpus=2), 42)
    runtime = local_runtime()
    store = PackedStore(runtime['data_root'], purpose='context')
    assert sha256(runtime['m0']) == '605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0'
    manifest = json.loads((source/'metadata/bundle_manifest.json').read_text())
    bundle = Path(runtime['bundle'])
    assert sha256(bundle/'spatial.npz') == manifest['spatial_sha256']
    assert sha256(bundle/'data.npz') == manifest['data_sha256']
    data, spatial = dict(np.load(bundle/'data.npz')), dict(np.load(bundle/'spatial.npz'))
    first = dict(np.load(source/'fold_0/gaussian_outer.npz'))
    hour = int(first['hours'][0]) if hour is None else int(hour)
    ti = int(np.flatnonzero(first['hours'] == hour)[0])
    ids = first['station_ids']
    scratch = object.__new__(GridStudy)
    scratch.runtime, scratch.store, scratch.device, scratch.net = runtime, store, 'cuda', None
    m0 = scratch.frozen_m0(hour)
    aid = store.ids('A'); aq, _ = safe_query_geometry(store, aid)
    grid = store.grid; nodes = np.arange(np.prod(store.shape))
    builder = LocalFeatureBuilder(np.c_[grid['x'].ravel(), grid['y'].ravel()], store.xy[aid],
        store.stations.height.to_numpy()[aid], grid['height'], grid['region'], nodes,
        store.stats, aq, store.query_valid[aid])
    frames = np.array([store.context_values([hour-offset])[0] for offset in (0, 6, 12, 18)])
    x = feature69(builder, frames, m0, cycles([hour])[0], store.shape)
    weight = json.loads(Path(runtime['parameters']).read_text())['weight']
    base = m0 + weight*(x[59]*float(store.stats['z0_std'])-m0)
    qxy, height = safe_query_geometry(store, ids)
    query = VertexQuery(store.shape, qxy, height, nodes).to('cuda')
    ix = int(np.searchsorted(data['hours'], hour)); assert data['hours'][ix] == hour
    amplitude = float(data['amplitude'][ix])
    phi, v = torch.as_tensor(spatial['phi'], device='cuda'), torch.as_tensor(spatial['v'], device='cuda')
    prior = gaussian_prior_field(phi, v,
        keyed_noise([hour], 128, 20261124, shape=(v.shape[0],), device='cuda'),
        keyed_noise([hour], 128, 20262124, shape=(v.shape[1],), device='cuda'), store.shape)[0]
    white = keyed_noise([hour], 128, 20264124, shape=(1, *store.shape), device='cuda')[0]
    errors = keyed_noise([hour], 128, 20263124, shape=(905,), device='cuda')[0, :, ids]
    conditions = torch.tensor(x[None], device='cuda')
    base_tensor = torch.tensor(base[None, None], device='cuda')
    with torch.inference_mode():
        base_query = query(base_tensor.flatten(2)).flatten().cpu().numpy()
    baseline_error = float(abs(base_query-first['mean'][ti]).max())
    assert baseline_error < .001, baseline_error
    results = dict(hour=hour, device=torch.cuda.get_device_name(), base_query_error_mm=baseline_error,
                   context_source='rebuilt from A only; no full feature-cache copy', runs={})
    for fold in (0, 1):
        for arm in ('stochastic', 'diffusion', 'gan'):
            directory = source/f'fold_{fold}'/arm
            reference = json.loads((directory/'joint_weights_reference.json').read_text())
            state = torch.load(source/reference['path'], map_location='cpu', weights_only=False)
            system = GenerationSystem(arm, recompute=False).to('cuda').eval()
            system.load_state_dict(state['ema'])
            fields = []; start = time.monotonic()
            with torch.inference_mode():
                for i in range(0, 128, 4):
                    fields.append(system.generator.sample_residual(conditions.expand(4, -1, -1, -1),
                        prior[i:i+4], white[i:i+4]).cpu())
                generated = torch.cat(fields).to('cuda')
                correction = query((generated-prior).flatten(2)).squeeze(1)*amplitude
                samples = base_query[None] + amplitude*(query(generated.flatten(2)).squeeze(1) + manifest['tau_surface_mm']*errors).cpu().numpy()
                correction = correction.cpu().numpy()
            archive = dict(np.load(directory/'joint_outer_predictions.npz'))
            error = float(abs(samples[:32]-archive['samples'][ti]).max())
            assert error < .001, (arm, fold, error)
            out = root/'member_stability_probe'/str(hour)/f'fold_{fold}'/arm; out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out/'samples128.npz', hour=hour, samples=samples, correction=correction,
                base_mean=base_query, station_ids=ids, truth=archive['truth'][ti], population=archive['population'][ti])
            results['runs'][f'fold_{fold}/{arm}'] = dict(members=128, first32_max_error_mm=error,
                inference_seconds=time.monotonic()-start,
                mean_change32_to128_rms_mm=float(np.sqrt(np.mean((correction[:32].mean(0)-correction.mean(0))**2))))
            (root/f'member_probe_status_{hour}.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(fold, arm, results['runs'][f'fold_{fold}/{arm}'], flush=True)
            del system, generated; torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hour', type=int)
    main(parser.parse_args().hour)
