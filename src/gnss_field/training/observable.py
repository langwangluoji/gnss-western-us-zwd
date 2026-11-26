"""Portable frozen-bundle training for the observation-sensitive coverage study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import atexit
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch

from gnss_field.artifacts import read_json, write_json, sha256, immutable_json
from gnss_field.contracts import ScienceConfig
from gnss_field.models.subspace import SubspaceEDM, station_members
from gnss_field.models.mean_shift import SubspaceMeanShift
from gnss_field.models.diffusion import keyed_noise
from gnss_field.objectives.scores import energy_score
from gnss_field.training.engine import TrainEngine, StageSpec, atomic_torch
from gnss_field.training.release import verify_release
from gnss_field.runtime import configure, _send_step_usr1


def acquire_study_lease(root):
    """One live GPU process in this study; a hard kill leaves an inspectable lock."""
    path = Path(root)/'.observable_gpu.lock'
    token = f'{os.getpid()}:{os.environ.get("SLURM_JOB_ID", "local")}:{time.time_ns()}'
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(token)
    def release():
        if path.exists() and path.read_text() == token:
            path.unlink()
    atexit.register(release)
    return path


def coverage_split(hours, qualified, development_anchors):
    hours = np.asarray(hours)
    training = np.asarray(qualified, bool).copy()
    for block in np.unique(np.asarray(development_anchors)//168):
        training &= ~((hours >= block*168-18) & (hours < (block+1)*168+18))
    return training


def load_bundle(directory):
    root = Path(directory)
    manifest = read_json(root/'manifest.json')
    if manifest.get('role') != 'B' or manifest.get('year') != 2018:
        raise PermissionError('Coverage training only accepts the declared B2018 bundle')
    if sha256(root/'data.npz') != manifest['data_sha256']:
        raise ValueError('Bundle data differs from locked release')
    arrays = dict(np.load(root/'data.npz'))
    if np.any(np.diff(arrays['hours']) <= 0):
        raise ValueError('Training hours must be strictly increasing and unique')
    years = pd.to_datetime(arrays['hours'], unit='h').year
    if not (years == 2018).all():
        raise ValueError('Unexpected year in training bundle')
    for name in ('training_indices', 'development_indices'):
        indices = arrays[name]
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(years)):
            raise ValueError(f'Invalid or duplicate {name}')
    expected = np.flatnonzero(coverage_split(arrays['hours'], arrays['valid'].sum(-1) >= 4,
        arrays['hours'][arrays['development_indices']]))
    if not np.array_equal(expected, arrays['training_indices']):
        raise ValueError('Training indices differ from registered weather-block exclusion')
    return arrays, manifest


def _same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            _same(first, second)
    else:
        assert a == b


def restore_initial_checkpoint(engine):
    """Keep the registered zero candidate if interruption precedes the first last.pt."""
    initial = engine.directory/'initial.pt'
    if engine.step > 0 and not initial.is_file() and engine.spec.observation_early_stopping:
        raise ValueError('A trained formal run is missing initial.pt; cannot relabel current EMA as step zero')
    if initial.is_file() and not (engine.directory/'last.pt').exists():
        engine._load(initial)


def study_schedule(manifest, training_hours):
    """A new explicit protocol cannot silently extend a frozen two-pass run."""
    protocol = manifest.get('convergence_protocol')
    if protocol not in (None, 'mean_convergence_v1'):
        raise ValueError('Unknown coverage protocol')
    maximum = 5000 if protocol else 2*int(np.ceil(training_hours/8))
    if manifest['maximum_updates_per_arm'] != maximum:
        raise ValueError('Manifest differs from registered schedule')
    science = ScienceConfig(rank=64, condition_dim=32, sigma_data=1., learning_rate=1e-4,
        refinement_lr=1e-4, warmup=100, evaluate_every=100, startup_evaluation=0, log_every=100,
        minimum_refinement_steps=1600 if protocol else 300, patience=10 if protocol else 3)
    return maximum, science


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['mean', 'edm'], default='mean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--budget-lock')
    args = parser.parse_args()
    release_hash = verify_release(Path(__file__).resolve().parents[3], required=not args.benchmark)
    configure(type('Runtime', (), {'cpus': 2})(), args.seed)
    root, output = Path(args.bundle), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.benchmark and any(output.iterdir()):
        raise ValueError('Benchmark requires a fresh engineering-only output directory')
    if not args.benchmark and (output/'benchmark.json').exists():
        raise ValueError('An engineering checkpoint cannot initialize formal scientific step zero')
    arrays, manifest = load_bundle(root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name() if device == 'cuda' else 'CPU'
    training = arrays['training_indices']
    maximum, science = study_schedule(manifest, len(training))
    model_file = Path(__file__).parent.parent/'models'/('subspace.py' if args.arm == 'edm' else 'mean_shift.py')
    if not args.benchmark:
        if not args.budget_lock:
            raise ValueError('Formal A800 training requires a post-benchmark budget lock')
        budget = read_json(args.budget_lock)
        if (platform.system() != 'Linux' or 'A800' not in gpu_name or
            budget['bundle_manifest_sha256'] != sha256(root/'manifest.json') or
            budget['runner_sha256'] != sha256(__file__) or
            budget['release_sha256'] != release_hash or
            budget['model_sha256'][args.arm] != sha256(model_file) or
            args.arm not in budget['allowed_arms'] or args.seed != budget['registered_seed'] or
            output.resolve().parent != Path(budget['registered_run_root']) or
            output.name != f'{args.arm}_seed{args.seed}' or
            budget['maximum_updates_per_arm'] != maximum or not budget['A800_benchmark_verified']):
            raise ValueError('Formal hardware, data or budget lock differs')
        cap_seconds = budget['maximum_card_hours_per_arm']*3600
    else:
        cap_seconds = 300.
    acquire_study_lease(output.parent)
    tensors = {key: torch.tensor(arrays[key], device=device) for key in
               ('condition', 'mean', 'truth', 'valid', 'amplitude', 'projection', 'remainder')}
    model = (SubspaceEDM() if args.arm == 'edm' else SubspaceMeanShift()).to(device)
    stage = f'observable_{args.arm}_benchmark' if args.benchmark else f'observable_{args.arm}_refine'
    spec = StageSpec(stage, maximum, maximum, 8, 8, args.seed,
        dict(bundle_manifest=sha256(root/'manifest.json'), runner=sha256(__file__),
             release_sha256=release_hash,
             model_source=sha256(model_file), model_type=type(model).__name__))
    engine = TrainEngine(model, output, arrays['hours'][training], spec, science)
    restore_initial_checkpoint(engine)
    engine.max_seconds = max(0., cap_seconds-engine.elapsed)
    if not args.benchmark and engine.phase != 'COMPLETE' and (
        engine.max_seconds <= 0 or (output/'PAUSE').exists() or (output.parent/'PAUSE').exists()):
        engine.phase = 'BUDGET_EXHAUSTED' if engine.max_seconds <= 0 else 'PAUSED'
        engine.save()
        raise SystemExit(75)

    def sample(current, indices, members, seed=None, gaussian=False):
        batch = len(indices)
        if seed is None:
            initial = torch.randn(batch, members, 64, device=device)
            remainder = torch.randn(batch, members, tensors['remainder'].shape[1], device=device)
            error = torch.randn(batch, members, len(arrays['station_ids']), device=device)
        else:
            hours = arrays['hours'][indices]
            initial = keyed_noise(hours, members, seed, (64,), device=device)
            remainder = keyed_noise(hours, members, seed+1000, (tensors['remainder'].shape[1],), device=device)
            error = keyed_noise(hours, members, seed+2000, (905,), device=device)[..., arrays['station_ids']]
        latent = initial if gaussian else current.sample_noise(tensors['condition'][indices], initial)
        return station_members(tensors['mean'][indices], latent, remainder, error,
            tensors['projection'], tensors['remainder'], tensors['amplitude'][indices], manifest['tau_surface_mm'])

    def objective(hours, current, step):
        indices = np.searchsorted(arrays['hours'], hours)
        loss = energy_score(sample(current, indices, 4), tensors['truth'][indices], tensors['valid'][indices])
        return dict(loss=loss, observation_ES=loss)

    def development_score(current):
        indices = arrays['development_indices']
        values = []
        for start in range(0, len(indices), 8):
            selected = indices[start:start+8]
            samples = sample(current, selected, 16, seed=20261119)
            # Equal hour weights, matching the registered U-ES development objective.
            values.append(float(energy_score(samples, tensors['truth'][selected], tensors['valid'][selected]))*len(selected))
        return sum(values)/len(indices)

    def evaluate(current, step):
        result = dict(value=development_score(current), members=16, population='B incremental weather development')
        if manifest.get('convergence_protocol'):
            result['raw_ES_mm'] = development_score(engine.model)
            result['EMA_minus_raw_ES_mm'] = result['value']-result['raw_ES_mm']
        return dict(joint=result)

    def telemetry(hours):
        seen = arrays['hours'][training] if engine.step*8 >= len(training) else engine.cursor.order[:engine.cursor.position]
        dates = pd.to_datetime(seen, unit='h')
        return dict(supervised_station_hours=int(tensors['valid'][np.searchsorted(arrays['hours'], hours)].sum()),
                    unique_training_hours_seen=len(seen), observed_months=sorted(set(int(v) for v in dates.month)),
                    observed_nominal_7day_blocks=len(np.unique(seen//168)))
    engine.batch_telemetry = telemetry
    if args.benchmark:
        started = time.monotonic()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            idx = arrays['development_indices'][:8]
            error = float(abs(sample(engine.ema, idx, 8, seed=42)-sample(engine.ema, idx, 8, seed=42, gaussian=True)).max())
            assert error <= .001
        engine.run(objective, stop_after=20)
        assert engine.step == 20
        if device == 'cuda':
            torch.cuda.synchronize()
        update_seconds = (time.monotonic()-started)/max(engine.step, 1)
        def fresh(directory):
            torch.manual_seed(args.seed)
            current = (SubspaceEDM() if args.arm == 'edm' else SubspaceMeanShift()).to(device)
            return TrainEngine(current, directory, arrays['hours'][training], spec, science, max_seconds=300)
        continuous = fresh(output/'resume_continuous')
        continuous.run(objective, stop_after=200)
        assert continuous.step == 200
        expected = continuous.state()
        first = fresh(output/'resume_split')
        first.run(objective, stop_after=100)
        assert first.step == 100
        resumed = fresh(output/'resume_split')
        resumed.run(objective, stop_after=200)
        assert resumed.step == 200
        actual = resumed.state()
        for key in ('model', 'ema', 'optimizers', 'rng', 'cursor', 'step'):
            _same(expected[key], actual[key])
        signal_probe = dict(verified=False, engineering_updates=0)
        if platform.system() == 'Linux' and 'A800' in gpu_name:
            probe = fresh(output/'slurm_signal')
            def send_real_signal(hours, current, step):
                if step == 5:
                    signal_probe['target'] = _send_step_usr1()
            probe.run(objective, stop_after=20, before_update=send_real_signal)
            if probe.phase != 'PAUSED' or not 5 <= probe.step < 20:
                raise ValueError('Real Slurm signal did not pause at an update boundary')
            signal_probe['paused_step'] = probe.step
            resumed_probe = fresh(output/'slurm_signal')
            resumed_probe.run(objective, stop_after=probe.step+5)
            if resumed_probe.step != probe.step+5:
                raise ValueError('Real Slurm signal checkpoint did not resume')
            signal_probe.update(verified=True, resumed_step=resumed_probe.step,
                                engineering_updates=resumed_probe.step)
        eval_start = time.monotonic()
        with torch.no_grad():
            metrics = evaluate(engine.ema, engine.step)
        if device == 'cuda':
            torch.cuda.synchronize()
        development_seconds = time.monotonic()-eval_start
        checkpoint_start = time.monotonic()
        atomic_torch(output/'serialization_probe.pt', engine.state())
        checkpoint_seconds = time.monotonic()-checkpoint_start
        peak_rss_mb = None
        if platform.system() == 'Linux':
            import resource
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        write_json(output/'benchmark.json', dict(gpu=gpu_name, platform=platform.system(), arm=args.arm,
            release_sha256=release_hash,
            bundle_manifest_sha256=sha256(root/'manifest.json'), maximum_initial_Gaussian_error_mm=error,
            runner_sha256=sha256(__file__), model_sha256=sha256(model_file),
            model_sources_sha256={arm: sha256(Path(__file__).parent.parent/'models'/name)
                for arm, name in [('mean', 'mean_shift.py'), ('edm', 'subspace.py')]},
            real_100_plus100_equals200=True,
            real_slurm_signal_resume=signal_probe['verified'], slurm_signal_probe=signal_probe,
            total_engineering_updates=420+signal_probe['engineering_updates'],
            updates=engine.step, seconds_per_update=update_seconds, development_seconds=development_seconds,
            checkpoint_seconds=checkpoint_seconds, peak_cpu_rss_mb=peak_rss_mb,
            environment=dict(executable=sys.executable, python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, numpy=np.__version__, pandas=pd.__version__, runner=str(Path(__file__).resolve())),
            metrics=metrics, peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else 0,
            formal_verification=platform.system() == 'Linux' and 'A800' in gpu_name,
            engineering_only=True, no_scientific_selection=True))
        return
    if not (output/'initial.pt').exists():
        with torch.no_grad():
            initial = evaluate(engine.ema, 0)
        checkpoint = output/'initial.pt'
        engine.products['joint'] = [dict(**initial['joint'], step=0, checkpoint=str(checkpoint.resolve()))]
        atomic_torch(checkpoint, engine.state())
    engine.run(objective, evaluator=evaluate)
    if engine.phase != 'COMPLETE':
        raise SystemExit(75)  # Saved pause/budget status; afterok must not advance.
    best = min(engine.products['joint'], key=lambda row: (row['value'], row['step']))
    immutable_json(output/'selection.lock.json', dict(**best, checkpoint_sha256=sha256(best['checkpoint'])))


if __name__ == '__main__':
    main()
