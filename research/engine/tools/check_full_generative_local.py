"""Small real-data CUDA engineering acceptance for all complete model families."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch

from gnss_field.artifacts import fingerprint, sha256, write_json
from gnss_field.data import PackedStore
from gnss_field.generative.background import FusionBackgroundAssets, FrozenFusionBackground
from gnss_field.generative.data import FieldDataConfig, FullFieldData, VerifiedBackgroundProvider
from gnss_field.generative.training import FullTrainingConfig, run_stage, load_stage_model
from grid_pair_data import local_runtime


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'results/20261122_full_generative_design_v01'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    work = project/'build/full_generative_real_check_v03'
    if os.name == 'nt':
        work = Path('\\\\?\\'+str(work))
    work.mkdir(parents=True, exist_ok=True)
    assets_spec = json.loads((root/'background_assets.local.json').read_text(encoding='utf-8'))
    assets = FusionBackgroundAssets(**assets_spec)
    runtime = local_runtime();store = PackedStore(runtime['data_root'], purpose='train')
    callback = FrozenFusionBackground(store, assets, device='cuda')
    hashes = {**assets.expected_sha256, str(project/'src/gnss_field/generative/background.py'):
              sha256(project/'src/gnss_field/generative/background.py')}
    background = VerifiedBackgroundProvider(callback, hashes, identity=callback.identity)
    config = FieldDataConfig(packed_root=runtime['data_root'], cache_root=str(work/'cache'),
                             background_assets=hashes, prepare_batch_hours=2)
    full = FullFieldData(config, purpose='train', store=store, background_provider=background)
    source = np.load(project/'results/20261122_local_coverage_expansion_v01/data.npz')
    hours = source['hours'][[0, 6, 30, 60]]
    full.prepare(hours, role='B')
    class Panel:
        train_hours = hours
        development_hours = []
        metadata = {**full.metadata, 'engineering_panel': hours.tolist(),
                    'label_release': full._label_identity('B')}
        def batch(self, hour_ids, role='B', device='cpu'):
            return full.batch(hour_ids, role=role, device=device)
    data = Panel();results = {};stage_configs = {};started = time.monotonic()
    base = FullTrainingConfig(output_dir=str(work/'control'), stage='deterministic_control',
        model_family='vae', device='cuda', epochs=1, min_epochs=1, batch_size=1,
        latent_dim=64, warmup_updates=0, checkpoint_every_updates=2,
        development_hours=[], reference_count=442., sigma_eff_mm=6.36)
    def execute(name, stage, family='vae', dependencies=None, epochs=1):
        stage_config = replace(base, output_dir=str(work/name), stage=stage, model_family=family,
                               epochs=epochs, min_epochs=epochs, dependency_paths=dependencies or {})
        stage_configs[name] = stage_config
        result = run_stage(stage_config, data=data)
        if result['status'] != 'complete':
            raise ValueError('Local engineering stage did not complete: '+name)
        results[name] = dict(status=result['status'], step=result['step'], checkpoint=result['checkpoint'])
        print('REAL_CUDA_STAGE', name, result['step'], flush=True)
        torch.cuda.empty_cache()
        return result['checkpoint']
    control = execute('control', 'deterministic_control')
    representation = execute('representation', 'vae_representation', dependencies={'representation':control})
    cache_checkpoint = execute('posterior', 'posterior_cache', dependencies={'representation':representation})
    cache = str(Path(cache_checkpoint).parent/'posterior_cache.pt')
    for family in ('vae', 'gan', 'diffusion'):
        prior = representation
        if family != 'vae':
            prior = execute(family+'_pretrain', family+'_pretrain', family,
                {'representation':representation, 'posterior_cache':cache})
        shared = execute(family+'_shared', 'prior_es', family,
                         {'representation':representation, 'posterior_cache':cache, 'prior':prior})
        complete = execute(family+'_complete', 'complete_model_es', family,
                           {'representation':representation, 'prior':shared})
        model, _ = load_stage_model(complete, device='cuda')
        batch = data.batch(hours[:1], device='cuda')
        with torch.no_grad():
            fields = model.sample_fields(batch, 4, 20261122)
        if not torch.isfinite(fields).all() or tuple(fields.shape) != (1, 4, 1, 144, 96):
            raise ValueError('Invalid real full-field sample')
        del model, fields
    resume_checks = {}
    for name in ('representation', 'gan_pretrain', 'diffusion_pretrain'):
        continuous_config = replace(stage_configs[name], output_dir=str(work/(name+'_continuous200')),
                                    epochs=50, min_epochs=50, evaluation_check_updates=())
        continuous = run_stage(continuous_config, data=data)
        stage_config = replace(continuous_config, output_dir=str(work/(name+'_split')), max_updates=100)
        paused = run_stage(stage_config, data=data)
        assert paused['status'] == 'paused' and paused['step'] == 100
        resumed = run_stage(replace(stage_config, max_updates=None), data=data)
        expected = torch.load(continuous['checkpoint'], map_location='cpu', weights_only=False)
        actual = torch.load(resumed['checkpoint'], map_location='cpu', weights_only=False)
        maximum = 0.
        for source_name in ('raw', 'ema'):
            for key, value in expected[source_name].items():
                other = actual[source_name][key]
                if value.is_floating_point():
                    torch.testing.assert_close(value, other, atol=1e-7, rtol=0)
                    maximum = max(maximum, float((value-other).abs().max()))
                else:
                    assert torch.equal(value, other)
        assert torch.equal(expected['rng']['torch'], actual['rng']['torch'])
        resume_checks[name] = dict(passed=True, continuous_updates=200, resumed_updates=[100, 100],
                                  maximum_float_state_error=maximum, CPU_RNG_exact=True)
        print('REAL_CUDA_RESUME', name, maximum, flush=True)
    record = dict(passed=True, device=torch.cuda.get_device_name(), results=results,
                  hours=hours.tolist(), B_only=True, C_accessed=False, D_accessed=False,
                  scope='engineering4realtraininghours;notprecisionorconvergenceevidence',
                  elapsed_seconds=time.monotonic()-started,
                  resume_checks=resume_checks,
                  data_identity=fingerprint(data.metadata), peak_memory_bytes=torch.cuda.max_memory_allocated())
    write_json(root/'real_cuda_acceptance.json', record)
    print('REAL_CUDA_ACCEPTED', record['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
