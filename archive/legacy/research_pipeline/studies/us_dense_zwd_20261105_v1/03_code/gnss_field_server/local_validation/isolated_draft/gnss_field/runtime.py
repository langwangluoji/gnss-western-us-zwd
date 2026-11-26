"""Local engineering checks and explicitly allocated Tongji benchmark probes."""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import fingerprint, model_digest, read_json, write_json
from gnss_field.contracts import ScienceConfig, RuntimeConfig


def configure(runtime, seed=42):
    torch.set_num_threads(runtime.cpus)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _command(args):
    if shutil.which(args[0]) is None:
        return dict(ok=False, command=args, output='command unavailable')
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return dict(ok=r.returncode == 0, command=args, output=(r.stdout+r.stderr)[-12000:])
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(ok=False, command=args, output=str(e))


def _send_step_usr1():
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    if not job or not step or not job.isdigit() or not step.isdigit():
        raise RuntimeError('A numbered real Slurm srun step is required for the signal probe')
    target = f'{job}.{step}'
    result = subprocess.run(['scancel', '--signal=USR1', target], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'Slurm step signal failed: {result.stderr}')
    return target


def doctor(runtime, *, output=None, live_tongji=False, hashes=None):
    """Static checks on a submit host; GPU proof is collected only in an allocation."""
    result = dict(verified=False, site='local', profile='local_unverified', gpu='unverified',
                  hostname=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
                  torch=torch.__version__, torch_cuda=torch.version.cuda, cpus=runtime.cpus,
                  memory_gb=runtime.memory_gb, partition=runtime.partition, gres=runtime.gres,
                  driver_ok=False, account_ok=False, hashes=hashes or {}, commands={},
                  updated=datetime.now(timezone.utc).isoformat())
    if live_tongji and os.name == 'posix':
        partitions = _command(['sinfo', '-h', '-p', runtime.partition, '-o', '%P|%G'])
        account = _command(['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', f'user={getpass.getuser()}', 'format=Account,Partition'])
        result['commands'] = dict(partitions=partitions, account=account)
        account_ok = account['ok'] and bool(account['output'].strip())
        partition_ok = partitions['ok'] and runtime.partition.lower() in partitions['output'].lower() and 'l40' in partitions['output'].lower()
        result.update(site='Tongji', profile='tongji_l40', gpu='L40', account_ok=bool(account_ok),
                      verified=bool(partition_ok and account_ok and runtime.cpus <= 7 and shutil.which('sbatch')))
    allocated = bool(os.environ.get('SLURM_JOB_ID') and
                     (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES')))
    if runtime.device.startswith('cuda') and torch.cuda.is_available():
        result['actual_gpu'] = torch.cuda.get_device_name()
        result['actual_memory_gb'] = torch.cuda.get_device_properties(0).total_memory/2**30
        result['driver_ok'] = bool(allocated and 'L40' in result['actual_gpu'] and result['actual_memory_gb'] >= 40)
        result['commands']['gpu'] = _command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    if output is not None:
        write_json(output, result)
    return result


def engineering_smoke(science, runtime, output, *, cases=2, resume_updates=0,
                      slurm_signal_test=False, profile=False):
    """A few true training observations exercise the actual global paths; no model selection."""
    from gnss_field.data import PackedStore, ContextBuilder
    from gnss_field.models.mean import MeanField
    from gnss_field.models.basis import SpatialBasis
    from gnss_field.models.diffusion import EDMHead
    from gnss_field.training.fields import predict_patches
    from gnss_field.training.system import FunctionalModel, FrozenDecoder
    from gnss_field.objectives import fit_latent, CoefficientTransform, energy_score
    from gnss_field.geometry import surface_query, basis_observation_matrix
    from gnss_field.training.engine import TrainEngine, StageSpec
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure(runtime)
    device = runtime.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    store = PackedStore(runtime.data_root, purpose='train')
    builder = ContextBuilder(store, use_history=science.use_history)
    available = np.asarray(store.hours)[pd.to_datetime(store.hours, unit='h').year == 2018]
    chosen = []
    for hour in available[:168]:
        mask = builder.support(hour)[1][store.ids('B')] & np.isfinite(store.targets('B', [hour])[0])
        if mask.sum() >= science.rank+8:
            chosen.append(int(hour))
            if len(chosen) == cases:
                break
    if len(chosen) != cases:
        raise ValueError('Insufficient qualified engineering hours')
    ids = store.ids('B')
    q = torch.as_tensor(store.qxy[ids], dtype=torch.float32, device=device)
    height = torch.as_tensor(store.stations.height.to_numpy()[ids], dtype=torch.float32, device=device)
    truth = torch.as_tensor(store.targets('B', chosen), dtype=torch.float32, device=device)
    valid = torch.as_tensor(np.stack([builder.support(t)[1][ids] for t in chosen]), device=device) & torch.isfinite(truth)
    context = builder.build_context(chosen, device=device)
    model = MeanField().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model_digest(model)
    started = time.perf_counter()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    model.train()
    field, _ = predict_patches(model, context, store, chunk=runtime.patch_chunk)
    mean = surface_query(field, q, height).squeeze(1)
    error = (mean-torch.where(valid, truth, 0))/store.stats['surface_std']
    loss = error[valid].square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    mean_seconds = time.perf_counter()-started
    changed = before != model_digest(model)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        frozen_mean = predict_patches(model, context, store, chunk=runtime.patch_chunk)[0]
    context.mean = frozen_mean
    basis = SpatialBasis(science.rank).to(device)
    xyz = torch.as_tensor(np.c_[store.grid['x'].ravel(), store.grid['y'].ravel(), store.grid['height'].ravel()], dtype=torch.float32, device=device)
    land = torch.as_tensor(store.grid['region'].ravel(), dtype=torch.bool, device=device)
    phi = basis(xyz, land)
    h = basis_observation_matrix(phi, store.shape, q, height)
    residual = (truth-surface_query(frozen_mean, q, height).squeeze(1))/store.stats['z0_std']
    coefficient = []
    start_ridge = time.perf_counter()
    for i in range(cases):
        c, _ = fit_latent(h[valid[i]], residual[i, valid[i]])
        coefficient.append(c.float())
    coefficients = torch.stack(coefficient)
    ridge_loss = coefficients.square().mean()
    ridge_loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in basis.parameters()):
        raise ValueError('Nonfinite basis/FP64 inverse gradient')
    ridge_seconds = time.perf_counter()-start_ridge
    transform = CoefficientTransform.fit(coefficients.detach()).to(device)
    decoder = FrozenDecoder(phi.detach(), store.shape, transform, store.stats['z0_std'])
    conditional = FunctionalModel('edm', science.rank, frozen_encoder=True).to(device)
    features = conditional.condition(context, decoder.scale).detach()
    head = conditional.head
    frozen_hash = model_digest(model)
    basis_hash = model_digest(basis)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    optimizer.zero_grad()
    calls, intermediate = [], []
    original = head.denoise
    def track(x, sigma, condition):
        value = original(x, sigma, condition)
        value.retain_grad()
        calls.append(float(torch.as_tensor(sigma).flatten()[0]))
        intermediate.append(value)
        return value
    head.denoise = track
    started = time.perf_counter()
    normalized = head.sample(features, chosen, 4, 29)
    head.denoise = original
    base_station = surface_query(frozen_mean, q, height).squeeze(1)
    predicted = decoder.station(normalized, base_station, matrix=h.detach())
    es = energy_score(predicted, truth, valid, decoder.scale)
    es.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    es_seconds = time.perf_counter()-started
    with torch.no_grad():
        normalized = head.sample(features, chosen, 4, 31)
        whole = decoder.field(normalized, frozen_mean)
        exact = surface_query(whole.squeeze(2), q, height)
        fast = decoder.station(normalized, base_station, matrix=h.detach())
        discrepancy = float((exact-fast).abs().max())
        split = torch.cat([head.sample(features[i:i+1], chosen[i:i+1], 4, 31) for i in range(cases)])
        torch.testing.assert_close(normalized, split, atol=1e-5, rtol=1e-5)
    if discrepancy > .001 or len(calls) != 35 or not changed:
        raise ValueError('Global field/EDM smoke acceptance failed')
    resume_report = None
    if resume_updates:
        initial = copy.deepcopy(head.state_dict())
        def create(folder, updates=resume_updates):
            fresh = EDMHead(science.rank).to(device)
            fresh.load_state_dict(initial)
            spec = StageSpec('edm_refine', updates, updates, cases, 1, 42,
                             {'kind': 'real_B_engineering', 'data': store.manifest['training_hash']})
            return TrainEngine(fresh, output/folder, np.arange(cases), spec, science)
        def objective(indices, network, step):
            tensor_idx = torch.as_tensor(indices, device=device)
            z = network.sample(features[tensor_idx], np.array(chosen)[indices], 4, 1000003+step)
            forecasts = decoder.station(z, base_station[tensor_idx], matrix=h.detach())
            return {'loss': energy_score(forecasts, truth[tensor_idx], valid[tensor_idx], decoder.scale)}
        a = create('resume_continuous')
        a.run(objective)
        b = create('resume_split')
        b.run(objective, stop_after=resume_updates//2)
        b = create('resume_split')
        b.run(objective)
        equal = all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items())
        if not equal:
            raise ValueError('Actual observation-space EDM exact recovery failed')
        resume_report = dict(updates=resume_updates, split=resume_updates//2, bitwise_equal=True,
                             stage='edm_real_B_ES_35NFE')
        if slurm_signal_test:
            reference = create('signal_reference', 4)
            reference.run(objective)
            interrupted = create('signal_interrupted', 4)
            signal_record = {}
            def signaled_objective(indices, network, step):
                if not signal_record:
                    signal_record['target'] = _send_step_usr1()
                    deadline = time.monotonic()+15
                    while not interrupted.pause and time.monotonic() < deadline:
                        time.sleep(.05)
                    if not interrupted.pause:
                        raise RuntimeError('USR1 did not reach the Python training process')
                return objective(indices, network, step)
            stopped = interrupted.run(signaled_objective)
            if stopped['status'] != 'PAUSED' or stopped['step'] != 1:
                raise RuntimeError('Signal did not checkpoint after a complete optimization update')
            restored = create('signal_interrupted', 4)
            restored.run(objective)
            if not all(torch.equal(v, restored.model.state_dict()[k]) for k, v in reference.model.state_dict().items()):
                raise RuntimeError('Slurm signal/checkpoint/resume changed the training result')
            resume_report['slurm_step_signal'] = {**signal_record, 'passed': True,
                                                 'pause_step': 1, 'resume_bitwise_equal': True}
    timings = {}
    if profile:
        from gnss_field.objectives import edm_loss, gaussian_nll, gradient_penalty
        def measured(name, operation):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            begin = time.perf_counter()
            operation()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            timings[name] = time.perf_counter()-begin
        targets = transform.encode(coefficients.detach())
        for arm in ('det', 'gaussian', 'edm', 'gan'):
            network = FunctionalModel(arm, science.rank, frozen_encoder=True).to(device)
            opt = torch.optim.AdamW(network.head.parameters(), lr=1e-4)
            def pretrain():
                opt.zero_grad()
                if arm == 'det':
                    value = (network.head(features)-targets).square().mean()
                elif arm == 'gaussian':
                    mu, logstd = network.head(features)
                    value = gaussian_nll(mu, logstd, targets)
                elif arm == 'edm':
                    value = edm_loss(network.head, targets, features)
                else:
                    critic_opt = torch.optim.Adam(network.critic.parameters(), lr=1e-4)
                    noise = torch.randn_like(targets)
                    fake = network.head(noise, features)
                    critic_loss = network.critic(fake.detach(), features).mean()-network.critic(targets, features).mean()
                    critic_loss = critic_loss+10*gradient_penalty(network.critic, targets, fake.detach(), features)
                    critic_loss.backward()
                    critic_opt.step()
                    network.critic.requires_grad_(False)
                    value = -network.critic(fake, features).mean()
                value.backward()
                opt.step()
            measured(arm+'_pretrain_batch', pretrain)
            def refinement():
                opt.zero_grad()
                c = network.coefficients(features, chosen, 4, 20261117, science=science)
                forecasts = decoder.station(c, base_station, matrix=h.detach())
                value = ((forecasts[:, 0]-torch.where(valid, truth, 0))[valid]/decoder.scale).square().mean() if arm == 'det' else energy_score(forecasts, truth, valid, decoder.scale)
                value.backward()
                opt.step()
            measured(arm+'_observation_batch', refinement)
            if arm != 'det':
                def evaluate():
                    with torch.no_grad():
                        c = network.coefficients(features, chosen, 64, 20261117, science=science)
                        forecasts = decoder.station(c, base_station, matrix=h.detach())
                        energy_score(forecasts, truth, valid, decoder.scale)
                measured(arm+'_K64_evaluation_batch', evaluate)
        from gnss_field.models.pixel import ResidualField, gaussian_losses
        from gnss_field.geometry import query_grid
        pixel = ResidualField().to(device).train()
        pixel_opt = torch.optim.AdamW(pixel.parameters(), lr=1e-4)
        def pixel_step():
            pixel_opt.zero_grad()
            output_field, output_features = predict_patches(pixel, context, store,
                chunk=runtime.patch_chunk, residual=True, scale=decoder.scale)
            mu = surface_query(output_field, q, height).squeeze(1)
            feat = query_grid(output_features, q).transpose(1, 2)
            geometry = feat.new_zeros(*feat.shape[:2], 3)
            std = pixel.standard_deviation(feat, geometry, decoder.scale, faithful=True)
            mse, nll = gaussian_losses(mu[valid], truth[valid], std[valid].square(), decoder.scale, faithful=True)
            (mse+nll).backward()
            pixel_opt.step()
        measured('m2_full_field_observation_batch', pixel_step)
        def cache_io():
            from gnss_field.artifacts import atomic_numpy
            path = output/'cache_timing.npy'
            atomic_numpy(path, frozen_mean.detach().cpu().numpy())
            np.load(path).sum()
            path.unlink()
        measured('mean_cache_write_reload_batch', cache_io)
        # Actual full-field effective-eight-hour updates, cycling the two audit hours only for timing.
        timings['mean_microbatch_sweep'] = []
        timing_mean = copy.deepcopy(model).requires_grad_(True)
        for micro in (1, 2):
            for chunk in sorted(set((4, runtime.patch_chunk))):
                timing_mean.train()
                optimizer = torch.optim.AdamW(timing_mean.parameters(), lr=1e-4)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                optimizer.zero_grad()
                for offset in range(0, 8, micro):
                    subset = [(offset+i) % cases for i in range(micro)]
                    ctx = builder.build_context(np.asarray(chosen)[subset], device=device)
                    field = predict_patches(timing_mean, ctx, store, chunk=chunk)[0]
                    pred = surface_query(field, q, height).squeeze(1)
                    (((pred-torch.where(valid[subset], truth[subset], 0))/store.stats['surface_std'])[valid[subset]].square().mean()*micro/8).backward()
                optimizer.step()
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                timings['mean_microbatch_sweep'].append(dict(micro_hours=micro, patch_chunk=chunk,
                    effective_hours=8, seconds=time.perf_counter()-begin, repeated_hours_for_timing=True))
    report = dict(kind='engineering_only_not_accuracy_result', passed=True, data=store.manifest['training_hash'],
                  device=torch.cuda.get_device_name() if device.startswith('cuda') else 'cpu',
                  hours=chosen, targets=int(valid.sum()), full_patch_count=len(store.grid['origins']),
                  mean_updated=changed, mean_seconds=mean_seconds, ridge_seconds=ridge_seconds,
                  es_seconds=es_seconds, denoiser_calls=len(calls),
                  all_denoiser_gradients_finite=all(v.grad is not None and torch.isfinite(v.grad).all() for v in intermediate),
                  query_equivalence_max_mm=discrepancy, foundation_unchanged=(frozen_hash == model_digest(model) and basis_hash == model_digest(basis)),
                  peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
                  resume=resume_report, verified_for_tongji_budget=False)
    report['profile'] = timings
    write_json(output/'smoke.json', report)
    return report


def benchmark(science, runtime, output, *, hashes, static_preflight, max_card_hours=4):
    if not 0 < max_card_hours <= 4 or runtime.gres.lower() != 'gpu:l40:1':
        raise ValueError('Initial benchmark is limited to four single-L40 card hours')
    if not os.environ.get('SLURM_JOB_ID') or os.name != 'posix':
        raise RuntimeError('A Tongji Slurm GPU allocation is required; use smoke for local engineering')
    if not static_preflight.get('verified') or static_preflight.get('hashes') != hashes:
        raise ValueError('A matching static Tongji preflight is required')
    probe = doctor(runtime, hashes=hashes)
    if not probe['driver_ok']:
        raise RuntimeError('Allocated GPU is not the verified L40 CUDA profile')
    begin = time.monotonic()
    report = engineering_smoke(science, runtime, output, cases=2, resume_updates=200,
                               slurm_signal_test=True, profile=True)
    elapsed = time.monotonic()-begin
    if elapsed > max_card_hours*3600:
        raise RuntimeError('Benchmark budget exhausted')
    if not report['resume'].get('slurm_step_signal', {}).get('passed'):
        raise RuntimeError('A successful real Slurm step signal/resume probe is required')
    report.update(verified_for_tongji_budget=True, hashes=hashes, allocated_profile='tongji_l40',
                  elapsed_seconds=elapsed, allocated_card_hours=elapsed/3600,
                  checkpoint_signal_test='passed_real_srun_step',
                  estimate_scope='measured kernels, two qualified hours; reservations require explicit review; not an end-to-end wall-time guarantee',
                  estimated_update_seconds={'mean_base': report['mean_seconds']/2,
                    'mean_all': report['mean_seconds']/2, 'basis': report['ridge_seconds']/2,
                    'edm_refine': report['es_seconds']/2})
    write_json(Path(output)/'benchmark.json', report)
    merged = {**static_preflight, 'driver_ok': True, 'gpu_probe_hostname': socket.gethostname(),
              'actual_gpu': probe['actual_gpu'], 'actual_memory_gb': probe['actual_memory_gb']}
    write_json(Path(output)/'verified_preflight.json', merged)
    return report
