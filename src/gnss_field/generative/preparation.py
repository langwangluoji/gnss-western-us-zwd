"""Bounded spawn preparation: one owner per monthly condition/label shard."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import multiprocessing as mp
import math
import os
import signal
import threading
from pathlib import Path
import time

import numpy as np
from gnss_field.artifacts import fingerprint, immutable_json, write_json


_WORKER_VIEWS = None
_WORKER_STARTUP_SECONDS = 0.
_WORKER_STARTUP = {}


class PreparationPaused(TimeoutError):
    """Resumable preparation pause after in-flight monthly writes finish."""


def interval_union_seconds(intervals):
    """Real concurrent task wall time, measured on the shared monotonic clock."""
    spans = sorted((float(start),float(stop)) for start,stop in intervals)
    if any(not math.isfinite(a) or not math.isfinite(b) or b<a for a,b in spans):
        raise ValueError('Task intervals require finite ordered monotonic timestamps')
    if not spans: return 0.
    total = 0.
    start,stop = spans[0]
    for a,b in spans[1:]:
        if a>stop:
            total += stop-start
            start,stop = a,b
        else: stop = max(stop,b)
    return total+stop-start


def price_preparation(report, *, total_hours):
    """One actual pool startup plus measured parallel tasks and verification."""
    missing = report['missing_condition_hours']
    prepared = report['prepared_hours']
    if missing<=0 or prepared<=0 or total_hours<=0:
        raise ValueError('Preparation pricing needs actual cold missing rows and positive populations')
    startup = float(report['startup_bound_seconds'])
    rate = report['task_union_seconds']/missing + report['verification_seconds']/prepared
    if not math.isfinite(startup+rate) or startup<0 or rate<=0:
        raise ValueError('Finite positive measured preparation timing required')
    recurring = total_hours*rate
    return dict(preparation_seconds=recurring,preparation_startup_seconds=startup,
                preparation_total_seconds=startup+recurring)


def preparation_plan(train_hours, development_hours):
    plan = []
    for role, hours, years in (('B', train_hours, (2016,2017,2018)), ('C', development_hours, (2019,))):
        hours = np.unique(np.asarray(hours, 'i8'))
        dates = hours.astype('datetime64[h]')
        if not np.isin(dates.astype('datetime64[Y]').astype(int)+1970, years).all():
            raise PermissionError('Preparation accepts B2016-2018 and C2019 only; D is prohibited')
        months = dates.astype('datetime64[M]').astype(str)
        for month in np.unique(months):
            plan.append(dict(role=role, month=str(month), hours=hours[months==month].tolist()))
    if len({p['month'] for p in plan}) != len(plan):
        raise ValueError('Each condition month must have exactly one owner')
    return plan


def _initialize_worker(config):
    global _WORKER_VIEWS, _WORKER_STARTUP_SECONDS, _WORKER_STARTUP
    started = time.monotonic()
    if hasattr(signal,'SIGUSR1'):
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from .campaign import build_data
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = build_data(config, allow_background_inference=True)
    # Input-only warmup creates no monthly condition/label shard and never
    # calls prepare/commit. It pays each CUDA/background startup once per worker.
    dev = train._other_view('development') if config['runtime'].get('preparation_has_development',False) else None
    warmup_hour = int(config['runtime']['preparation_warmup_hour'])
    train._build_compact(np.asarray([warmup_hour],'i8'))
    finished = time.monotonic()
    _WORKER_VIEWS = {'B':train, 'C':dev}
    _WORKER_STARTUP = dict(started=started,finished=finished,warmup_hour=warmup_hour)
    _WORKER_STARTUP_SECONDS = finished-started


def _prepare_month(task):
    global _WORKER_STARTUP_SECONDS
    import torch
    started = time.monotonic()
    role = task['role']
    if _WORKER_VIEWS[role] is None:
        _WORKER_VIEWS[role] = _WORKER_VIEWS['B']._other_view('development')
    data = _WORKER_VIEWS[role]
    hours = np.asarray(task['hours'], 'i8')
    data._check_role(role, data._hours(hours))
    month = task['month']
    if not (hours.astype('datetime64[h]').astype('datetime64[M]').astype(str)==month).all():
        raise ValueError('Monthly writer was assigned hours outside its owned month')
    label_id = data._label_identity(role)
    paths = ((data.cache_root/'conditions'/f'{month}.npz', data.identity, True),
             (data.cache_root/'labels'/role/label_id[:16]/f'{month}.npz', label_id, False))
    missing_counts = []
    for path, identity, condition in paths:
        old = data._load_shard(str(path))['hour_ids'] if data._has_shard(path) else np.array([], 'i8')
        missing = hours[~np.isin(hours, old)]
        missing_counts.append(len(missing))
        if len(missing):
            arrays = (data._build_preparation_month(missing) if condition else
                      dict(hour_ids=missing, values=data.store.targets(role, missing)))
            data._commit_month(path, arrays, identity=identity)
    # Workers deliberately publish no aggregate preparation_B/C completion file.
    finished = time.monotonic()
    result = {**task, 'task_started':started, 'task_finished':finished,
              'worker_startup':dict(_WORKER_STARTUP), 'condition_identity':data.identity, 'labels_identity':label_id,
              'resolved_cache_root':str(Path(data.cache_root).resolve()),
              'missing_condition_hours':missing_counts[0], 'missing_label_hours':missing_counts[1],
              'elapsed_seconds':finished-started,
              'worker_startup_seconds':_WORKER_STARTUP_SECONDS, 'worker_pid':os.getpid(),
              'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    _WORKER_STARTUP_SECONDS = 0.
    return result


def _dispatch_months(plan, *, config, workers, timeout_seconds, stop_requested=None):
    # Set before spawn/import so every BLAS/OpenMP process is bounded at two CPUs.
    keys = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
    previous = {key:os.environ.get(key) for key in keys}
    for key in keys: os.environ[key] = '2'
    executor = None
    futures = set()
    deadline = None if timeout_seconds is None else time.monotonic()+timeout_seconds
    try:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                       initializer=_initialize_worker, initargs=(config,))
        pending_tasks = iter(plan)
        # Keep at most one task per worker outstanding. On a cooperative pause,
        # no prefetched extra months can prolong shutdown beyond active months.
        for _ in range(min(workers,len(plan))):
            futures.add(executor.submit(_prepare_month,next(pending_tasks)))
        results = []
        while futures:
            if ((stop_requested is not None and stop_requested()) or
                (deadline is not None and time.monotonic()>=deadline)):
                raise PreparationPaused('Preparation paused at monthly boundaries; retry resumes committed shards')
            done,_ = wait(futures,timeout=.25,return_when=FIRST_COMPLETED)
            for future in done:
                futures.remove(future)
                results.append(future.result())
            if stop_requested is not None and stop_requested():
                raise PreparationPaused('Preparation paused at monthly boundaries; retry resumes committed shards')
            for _ in range(len(done)):
                task = next(pending_tasks,None)
                if task is not None: futures.add(executor.submit(_prepare_month,task))
        executor.shutdown(wait=True)
        return results
    except PreparationPaused:
        for future in futures: future.cancel()
        # Finish current atomic monthly transactions; publish no aggregate.
        if executor is not None: executor.shutdown(wait=True,cancel_futures=True)
        raise
    except BaseException:
        for future in futures: future.cancel()
        # A broken process is a failure, rather than a cooperative pause.
        if executor is not None:
            for process in list((getattr(executor,'_processes',None) or {}).values()):
                if process.is_alive(): process.terminate()
            executor.shutdown(wait=True,cancel_futures=True)
        raise
    finally:
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key] = value


def prepare_parallel(config, data=None, *, train_hours=None, development_hours=None,
                     workers=None, timeout_seconds=None, _dispatch=None):
    """Prepare missing rows with 1..4 workers sharing the configured one GPU.

    Parent validates every role/month identity before fast all-hour prepare calls
    publish the aggregate B and C manifests. Failed workers leave resumable atomic
    month shards, with no new aggregate completion claim.
    """
    workers = config.get('runtime',{}).get('preparation_workers',4) if workers is None else workers
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=4:
        raise ValueError('Preparation workers must be an integer from 1 through 4')
    device = config.get('runtime',{}).get('device','cuda')
    if not device.startswith('cuda') or ',' in device:
        raise ValueError('Parallel preparation requires one shared CUDA device')
    if data is None:
        from .campaign import build_data
        data = build_data(config, allow_background_inference=True)
    dev = data._other_view('development')
    train_hours = data.train_hours if train_hours is None else train_hours
    development_hours = dev.audit('C')['hour_ids'] if development_hours is None else development_hours
    plan = preparation_plan(train_hours, development_hours)
    if not plan: raise ValueError('Preparation needs a nonempty population')
    immutable_json(Path(data.cache_root)/'contract.json',data.metadata)
    # data.config.cache_root is the unprefixed cache input. data.cache_root
    # already contains its condition identity, so passing it to build_data
    # would append that namespace twice. Preserve canonical background assets.
    worker_config = deepcopy(config)
    worker_config.setdefault('data',{})['cache_root'] = str(data.config.cache_root)
    worker_config.setdefault('runtime',{})['preparation_warmup_hour'] = plan[0]['hours'][0]
    worker_config['runtime']['preparation_has_development'] = any(p['role']=='C' for p in plan)
    stop = threading.Event()
    previous_handler = None
    signal_installed = False
    if hasattr(signal,'SIGUSR1'):
        try:
            previous_handler = signal.signal(signal.SIGUSR1,lambda signum,frame:stop.set())
            signal_installed = True
        except ValueError:
            pass  # Only the Python main thread can install signal handlers.
    started = time.monotonic()
    try:
        results = (_dispatch or _dispatch_months)(plan,config=worker_config,workers=workers,
                   timeout_seconds=timeout_seconds,stop_requested=stop.is_set)
        if stop.is_set():
            raise PreparationPaused('Preparation paused before aggregate completion')
    finally:
        if signal_installed: signal.signal(signal.SIGUSR1,previous_handler)
    verification_started = time.monotonic()
    expected = {(p['role'],p['month']):p for p in plan}
    if len(results)!=len(plan): raise RuntimeError('Incomplete monthly preparation results')
    seen = set()
    for result in results:
        key = (result['role'],result['month'])
        view = data if result['role']=='B' else dev
        if (key not in expected or key in seen or result['hours']!=expected[key]['hours'] or
            result['condition_identity']!=view.identity or result['labels_identity']!=view._label_identity(result['role']) or
            result.get('resolved_cache_root')!=str(Path(view.cache_root).resolve())):
            raise RuntimeError('Monthly preparation ownership or source identity mismatch')
        seen.add(key)
    reports = {}
    for role,view,hours in (('B',data,train_hours),('C',dev,development_hours)):
        if len(hours):
            view._clear_cache()
            reports[role] = view.prepare(hours,role=role)
    worker_peaks = {}
    for row in results:
        if 'worker_pid' in row:
            peaks = worker_peaks.setdefault(str(row['worker_pid']),dict(allocated=0,reserved=0,startup_seconds=0.))
            peaks['startup_seconds'] = max(peaks['startup_seconds'],row.get('worker_startup_seconds',0.))
            peaks['allocated'] = max(peaks['allocated'],row.get('cuda_peak_allocated_bytes',0))
            peaks['reserved'] = max(peaks['reserved'],row.get('cuda_peak_reserved_bytes',0))
    verification_seconds = time.monotonic()-verification_started
    task_union = interval_union_seconds([(r['task_started'],r['task_finished']) for r in results
                                         if 'task_started' in r and 'task_finished' in r])
    report = dict(schema=2, workers=workers, worker_cpu_threads=2, device=device,
                  task_union_seconds=task_union,verification_seconds=verification_seconds,
                  startup_bound_seconds=sum(p['startup_seconds'] for p in worker_peaks.values()),
                  missing_condition_hours=sum(r.get('missing_condition_hours',0) for r in results),
                  warmup=dict(hour=plan[0]['hours'][0],input_only=True,writes_cache=False),
                  worker_memory_peaks=worker_peaks,
                  worker_peak_allocated_sum_bytes=sum(p['allocated'] for p in worker_peaks.values()),
                  worker_peak_reserved_sum_bytes=sum(p['reserved'] for p in worker_peaks.values()),
                  memory_peak_scope='sum_of_per_worker_process_peaks_conservative_upper_bound_not_whole_device_measurement',
                  prepared_hours=sum(len(p['hours']) for p in plan), monthly=results,
                  roles=reports, elapsed_seconds=time.monotonic()-started,
                  condition_identity=data.identity, plan_sha256=fingerprint(plan))
    write_json(Path(data.cache_root)/'preparation_parallel.json',report)
    return report
