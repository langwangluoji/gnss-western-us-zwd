"""Measured cold-month/full-population cost model, separate from scientific runs."""
from dataclasses import replace
from pathlib import Path
import math
import os
import signal
import time

import numpy as np
import torch

from gnss_field.artifacts import fingerprint, immutable_json, read_json, sha256, write_json


def benchmark_panel(hours, per_month=8):
    hours = np.unique(np.asarray(hours, 'i8'))
    months = hours.astype('datetime64[h]').astype('datetime64[M]')
    selected = []
    for month in np.unique(months):
        group = hours[months == month]
        selected.extend(group[np.linspace(0, len(group)-1, min(per_month, len(group))).astype(int)])
    return np.random.default_rng(20261122).permutation(np.asarray(selected, 'i8'))


def sustained_timing(warm, sustained, *, cold_wall_seconds):
    """Price updates 9..32 from actual cumulative CPU/data/model telemetry."""
    count = sustained['step']-warm['step']
    if warm['step'] != 8 or sustained['step'] != 32 or count != 24:
        raise ValueError('Sustained benchmark requires warm8 then resumed32 updates')
    a, b = warm['telemetry'], sustained['telemetry']
    values = {name:(b[name]-a[name])/count for name in ('data_seconds','model_seconds')}
    if any(not math.isfinite(v) or v<0 for v in values.values()) or sum(values.values())<=0:
        raise ValueError('Missing or decreasing sustained data/model telemetry')
    rate = sum(values.values())
    return dict(seconds_per_update=rate,
                data_seconds_per_update=values['data_seconds'],
                model_seconds_per_update=values['model_seconds'],
                startup_seconds=max(0.,cold_wall_seconds-a['checkpoint_seconds']-warm['step']*rate),
                checkpoint_seconds=b['checkpoint_seconds'], warmup_updates=8, sustained_updates=24)


def missing_preparation_panel(data, *, months=4, per_month=32):
    """Choose genuinely missing rows, spanning distinct monthly geometry states."""
    hours = np.asarray(data.train_hours,'i8')
    calendar = hours.astype('datetime64[h]').astype('datetime64[M]').astype(str)
    candidates = []
    for month in np.unique(calendar):
        selected = hours[calendar==month]
        path = data.cache_root/'conditions'/f'{month}.npz'
        old = data._load_shard(str(path))['hour_ids'] if data._has_shard(path) else []
        missing = selected[~np.isin(selected,old)]
        if len(missing)>=per_month:
            candidates.append((month,missing))
    if len(candidates)<months:
        raise ValueError('Parallel preparation benchmark needs128 missing rows across4 months')
    chosen = np.linspace(0,len(candidates)-1,months).astype(int)
    return np.concatenate([candidates[i][1][np.linspace(0,len(candidates[i][1])-1,per_month).astype(int)] for i in chosen])


def benchmark_evaluation_runtime(config):
    runtime = config.get('runtime',{})
    evaluation = config.get('evaluation',{})
    batch = int(evaluation.get('evaluation_hours_batch',runtime.get('evaluation_hours_batch',16)))
    chunk = int(evaluation.get('evaluation_chunk',runtime.get('evaluation_chunk',16)))
    if not 1<=batch<=16 or chunk<1:
        raise ValueError('Evaluation runtime requires B1..16 and a positive member chunk')
    return dict(evaluation_hours_batch=batch,evaluation_chunk=min(chunk,256//batch))


def extrapolate_costs(*, training_hours, development_hours, preparation_seconds_per_hour,
                      evaluation_seconds_per_hour, pilot_seconds_per_update,
                      pilot_validation_seconds_per_hour, checkpoint_seconds,
                      representation_check_seconds, final_seconds_per_hour=None, pilot_startup_seconds=0.):
    values = [preparation_seconds_per_hour, pilot_seconds_per_update,
              pilot_validation_seconds_per_hour, checkpoint_seconds, representation_check_seconds,
              *evaluation_seconds_per_hour.values()]
    if any(not math.isfinite(float(v)) or v <= 0 for v in values):
        raise ValueError('Finite positive actual measurements required')
    if not math.isfinite(pilot_startup_seconds) or pilot_startup_seconds<0:
        raise ValueError('Invalid measured diagnostic startup cost')
    final = final_seconds_per_hour or evaluation_seconds_per_hour
    # 12 pilots: 9 sigma64 and 3 latent128. Each validates once, including
    # separate predictive and withheld-posterior scoring, measured together.
    diagnostics = 12*(5*math.ceil(1728/8)*pilot_seconds_per_update +
                       864*pilot_validation_seconds_per_hour + 60*checkpoint_seconds + pilot_startup_seconds)
    return dict(preparation_seconds=(training_hours+development_hours)*preparation_seconds_per_hour,
        evaluation_seconds_per_call={k:576*v for k,v in evaluation_seconds_per_hour.items()},
        diagnostics_seconds=diagnostics,
        # Upper bound: every optimized stage (10) has both point/joint products;
        # use the highest measured K128 cost. Baselines are covered conservatively.
        final_evaluation_seconds=40*development_hours*max(final.values())+representation_check_seconds,
        checkpoint_seconds_per_write=checkpoint_seconds)


def run_benchmark(config, data, output):
    """Time all real objective paths. No benchmark checkpoint is a candidate.

    This is the only run allowed before a measured budget lock. Its allocation
    has a one-card-hour hard wall limit. The panel spans all 36 training months;
    no previous small hot-cache timing is accepted as full-run evidence.
    """
    from .training import FullTrainingConfig, run_stage, load_stage_model
    from .evaluation import evaluate_model
    from .diagnostics import score_B_prior, _posterior_withheld_rmse, physical_group_folds, _station_metadata
    evaluation_runtime = benchmark_evaluation_runtime(config)
    device = config['runtime'].get('device', 'cuda')
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Server budget benchmark requires an actual A800')
    name = torch.cuda.get_device_name()
    if 'A800' not in name:
        raise RuntimeError('Budget cannot be locked from a different GPU: '+name)
    started = time.monotonic()
    deadline = started+min(float(config.get('benchmark', {}).get('budget_seconds', 3300)), 3300)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    root = Path(output).parent/'benchmark_work'; root.mkdir(parents=True, exist_ok=True)
    all_hours = np.asarray(data.train_hours, 'i8')
    hours = benchmark_panel(all_hours)
    if len(hours) < 128:
        raise ValueError('Need a cross-month panel of at least 128 training hours')
    months = np.unique(hours.astype('datetime64[h]').astype('datetime64[M]'))
    if len(months) != 36:
        raise ValueError('Benchmark requires all 36 training months')
    evaluation_hours = benchmark_panel(np.arange(np.datetime64('2019-01-01').astype('datetime64[h]').astype('i8'),
        np.datetime64('2020-01-01').astype('datetime64[h]').astype('i8')), per_month=4)
    from .preparation import prepare_parallel, price_preparation
    # A recheck binds the frozen original A800 record while measuring NEW missing
    # rows in the real shared cache. A hot row is never called cold preparation.
    source_path = config.get('benchmark',{}).get('reuse_preparation_benchmark')
    source_binding = None
    if source_path:
        source = read_json(source_path)
        if source.get('actual_device') != name or source.get('condition_identity') != fingerprint(data.metadata):
            raise ValueError('Original benchmark device or condition identity changed')
        source_binding = dict(path=str(source_path),sha256=sha256(source_path))
        cold_hours = missing_preparation_panel(data)
    else:
        if (Path(data.config.cache_root)/'preparation_B.json').exists():
            raise ValueError('Use explicit reuse_preparation_benchmark for a hot-cache performance recheck')
        cold_hours = hours
    cold_prepare = prepare_parallel(config,data=data,train_hours=cold_hours,
        development_hours=evaluation_hours,timeout_seconds=max(1.,deadline-time.monotonic()-90))
    missing = sum(row['missing_condition_hours'] for row in cold_prepare['monthly'])
    if missing < 128:
        raise ValueError('Cold preparation measurement requires at least128 actually missing rows')
    preparation_costs = price_preparation(cold_prepare,total_hours=len(all_hours)+8760)
    preparation_per_hour = preparation_costs['preparation_seconds']/(len(all_hours)+8760)
    hot_prepare = prepare_parallel(config,data=data,train_hours=cold_hours,
        development_hours=evaluation_hours,timeout_seconds=max(1.,deadline-time.monotonic()-90))
    if any(row['missing_condition_hours'] for row in hot_prepare['monthly']):
        raise RuntimeError('Hot-cache preparation unexpectedly rebuilt conditions')
    # All objective panel rows must exist before timer starts. Existing source
    # rows are retained, and fresh benchmark mode prepares its complete panel.
    data.prepare(hours,role='B')
    dev = data._other_view('development')
    dev.prepare(evaluation_hours,role='C')

    class Panel:
        train_hours = hours
        development_hours = []
        metadata = {**data.metadata, 'benchmark_panel': hours.tolist()}
        def batch(self, ids, role='B', device='cpu'):
            return data.batch(ids, role=role, device=device)
        def _label_identity(self, role):
            return data._label_identity(role)
    panel = Panel(); results = {}; measured = {}; timings = {}; startups = {}; checkpoint_times = []
    base = FullTrainingConfig(**{**config['scientific'], **config.get('training', {}),
        'stage':'deterministic_control', 'model_family':'vae', 'device':device,
        'latent_dim':128,
        'output_dir':str(root/'control'), 'epochs':10, 'min_epochs':10, 'batch_size':8,
        'max_updates':8, 'checkpoint_every_updates':4, 'development_hours':[],
        'reference_count':data.audit('B')['reference_count']})

    def execute(label, stage, family='vae', dependencies=None, batch_size=8, epochs=10, cache=False):
        if time.monotonic() > deadline-60:
            raise TimeoutError('Measured benchmark exceeded one-card-hour preflight envelope')
        cfg = replace(base, stage=stage, model_family=family, output_dir=str(root/label),
            epochs=epochs, min_epochs=epochs, batch_size=batch_size,
            kl_warmup_updates=1 if stage=='vae_representation' else 0,
            dependency_paths=dependencies or {}, max_updates=None if cache else 8,
            budget_seconds=deadline-time.monotonic()-30)
        if (Path(cfg.output_dir)/'last.pt').exists():
            raise ValueError('Benchmark output exists; do not mix old/new measurements')
        t = time.monotonic(); warm = run_stage(replace(cfg,max_updates=8), data=panel)
        torch.cuda.synchronize(); cold_wall = time.monotonic()-t
        if warm['step'] != 8:
            raise RuntimeError('Incomplete warmup objective path: '+label)
        result = run_stage(replace(cfg,max_updates=32),data=panel)
        torch.cuda.synchronize()
        timing = sustained_timing(warm,result,cold_wall_seconds=cold_wall)
        measured[label] = timing['seconds_per_update']
        startups[label] = timing['startup_seconds']
        timings[label] = timing
        # Conservative denominator retains real writes, including the extra
        # pause/resume boundary; it never subtracts checkpoint I/O from data.
        checkpoint_times.append(max(1e-6,warm['telemetry']['checkpoint_seconds']/4,
                                     result['telemetry']['checkpoint_seconds']/8))
        if cache:
            result = run_stage(replace(cfg,max_updates=None),data=panel)
            if result['status'] != 'complete':
                raise RuntimeError('Incomplete posterior cache path')
        results[label] = result
        print('BENCHMARK_STAGE', label, result['step'], timing, flush=True)
        return result['checkpoint']

    control = execute('control', 'deterministic_control')
    representation = execute('vae_representation', 'vae_representation', dependencies={'representation':control})
    # Actual Linux USR1 is delivered during the fourth complete optimizer batch.
    # Both paths cover the same cross-month working set and checkpointed RNG.
    resume_cfg=replace(base,stage='vae_representation',output_dir=str(root/'resume_continuous'),
                       dependency_paths={'representation':control},checkpoint_every_updates=4)
    continuous=run_stage(resume_cfg,panel)
    class SignaledPanel(Panel):
        calls=0
        def batch(self, ids, role='B', device='cpu'):
            values=super().batch(ids,role,device);self.calls+=1
            if self.calls==4: os.kill(os.getpid(),signal.SIGUSR1)
            return values
    split_cfg=replace(resume_cfg,output_dir=str(root/'resume_signaled'))
    interrupted=run_stage(split_cfg,SignaledPanel())
    if interrupted['status']!='paused' or interrupted['step']!=4:
        raise RuntimeError('Actual USR1 did not pause at the completed update boundary')
    resumed=run_stage(split_cfg,panel)
    a=torch.load(continuous['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
    for weights in ('raw','ema'):
        if any(not torch.equal(value,b[weights][key]) for key,value in a[weights].items()):
            raise RuntimeError('A800 signal/resume weights differ from uninterrupted training')
    if not torch.equal(a['rng']['torch'],b['rng']['torch']):
        raise RuntimeError('A800 signal/resume RNG differs')
    posterior = execute('posterior_cache', 'posterior_cache', dependencies={'representation':representation}, cache=True, epochs=1)
    cache = str(Path(posterior).parent/'posterior_cache.pt')
    evaluation_cost = {}; final_cost = {}; example_models = {}
    for family in ('vae', 'gan', 'diffusion'):
        prior = representation
        if family != 'vae':
            prior = execute(family+'_pretrain', family+'_pretrain', family,
                {'representation':representation, 'posterior_cache':cache}, batch_size=128, epochs=20)
        shared = execute('shared_'+family, 'prior_es', family,
            {'representation':representation, 'posterior_cache':cache, 'prior':prior}, epochs=3)
        complete = execute('complete_'+family, 'complete_model_es', family,
            {'representation':representation, 'prior':shared}, epochs=3)
        model, state = load_stage_model(complete, device)
        evaluation_config = replace(base, **config.get('evaluation', {}), sigma_eff_mm=base.sigma_eff_mm)
        for k, costs in ((16, evaluation_cost), (128, final_cost)):
            evaluation_config = replace(evaluation_config, eval_members=k, point_members=max(64,k),
                                        **evaluation_runtime)
            t = time.monotonic()
            evaluate_model(model, dev, evaluation_hours, evaluation_config)
            torch.cuda.synchronize(); costs[family] = (time.monotonic()-t)/len(evaluation_hours)
        example_models[family] = complete
        del model; torch.cuda.empty_cache()
    # Measure the B-only diagnostic scoring, including q-withheld capability,
    # with the larger registered latent size to price its added decoding cost.
    model, _ = load_stage_model(example_models['vae'], device)
    folds = physical_group_folds(_station_metadata(data))
    diagnostic_hours = hours[:48]
    t = time.monotonic()
    score_B_prior(model, data, diagnostic_hours, folds==0, sigma_eff_mm=base.sigma_eff_mm,
                  device=device, deadline=deadline, **evaluation_runtime)
    _posterior_withheld_rmse(model, data, diagnostic_hours, folds, 0, device, deadline, **evaluation_runtime)
    torch.cuda.synchronize()
    diagnostic_per_hour = (time.monotonic()-t)/len(diagnostic_hours)
    # Same actual128-dimensional ELBO path; use sustained telemetry rather
    # than repeating an eight-update cold-start pilot and extrapolating it.
    pilot_cost = measured['vae_representation']
    costs = extrapolate_costs(training_hours=len(all_hours), development_hours=8760,
        preparation_seconds_per_hour=preparation_per_hour, evaluation_seconds_per_hour=evaluation_cost,
        pilot_seconds_per_update=max(pilot_cost, measured['vae_representation']),
        pilot_validation_seconds_per_hour=diagnostic_per_hour*1.25,
        checkpoint_seconds=max(checkpoint_times), representation_check_seconds=diagnostic_per_hour*864,
        final_seconds_per_hour=final_cost, pilot_startup_seconds=startups['vae_representation'])
    costs.update(preparation_costs)
    record = dict(schema=3, measured=True, device='A800', actual_device=name,
        full_working_set=True, scope='36month shuffled sustained objectives; actual missing-row parallel preparation',
        original_benchmark=source_binding,
        runtime_source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
                              ('benchmark.py','preparation.py','evaluation.py','training.py')},
        preparation_measurements=dict(cold=cold_prepare,hot=hot_prepare),
        stage_timing=timings, stage_startup_seconds={k:v for k,v in startups.items() if k!='posterior_cache'},
        posterior_cache_startup_seconds=startups['posterior_cache'],
        evaluation_runtime=evaluation_runtime,
        benchmark_hours=hours.tolist(), development_panel=evaluation_hours.tolist(),
        training_qualified_hours=len(all_hours), condition_identity=fingerprint(data.metadata),
        stage_seconds_per_update={k:v for k,v in measured.items() if k!='posterior_cache'},
        posterior_cache_seconds_per_batch=measured['posterior_cache'], **costs,
        final_evaluation_seconds_per_hour_K128=final_cost,
        benchmark_elapsed_seconds=time.monotonic()-started,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        peak_memory_scope='benchmark_parent_process_only_excludes_preparation_workers',
        parent_peak_reserved_bytes=torch.cuda.max_memory_reserved(), benchmark_candidates_eligible=False,
        priced_latent_dim=128, representation_check_seconds=diagnostic_per_hour*864,
        signal_resume=dict(passed=True,signal='SIGUSR1',continuous_updates=8,split_updates=[4,4],raw_ema_exact=True,CPU_RNG_exact=True),
        objective_results=results)
    immutable_json(output, record)
    write_json(Path(output).parent/'benchmark_summary.json',
               {k:v for k,v in record.items() if k!='objective_results'})
    return record
