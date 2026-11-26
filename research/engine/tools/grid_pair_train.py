"""Finite paired processor queue, with real-data recovery and measured budgets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import signal
from pathlib import Path
import time
import numpy as np
import torch

from grid_pair_data import GridStudy, PROJECT, RUN
from local_context_train import initialize_zero
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256,model_digest,fingerprint
from gnss_field.contracts import ScienceConfig
from gnss_field.models.grid_processor import GridProcessor
from gnss_field.objectives.scores import energy_score
from gnss_field.evaluation import point_metrics
from gnss_field.training.engine import TrainEngine,StageSpec,PauseRequested,atomic_torch
from gnss_field.training.observable import _same
from gnss_field.runtime import configure

ARMS=('mlp69','resunet69')


def scientific_settings(n):
    if n<8 or n%8:
        raise ValueError('Exact paired epoch contract requires a multiple of eight qualified hours')
    epoch=n//8
    return ScienceConfig(rank=64,condition_dim=69,budgets={'grid_processor_refine':4*epoch},
        learning_rate=1e-4,refinement_lr=1e-4,ema=.99,warmup=100,
        startup_evaluation=200,evaluate_every=epoch//2,log_every=100,
        minimum_refinement_steps=2*epoch,patience=3),4*epoch


def check_evaluation_pause(data):
    if ((data.root/'PAUSE').exists() or
        (getattr(data,'deadline',None) is not None and time.monotonic()>=data.deadline)):
        path=data.root/'queue_status.json'
        status=read_json(path) if path.exists() else {}
        write_json(path,dict(**{k:v for k,v in status.items() if k not in ('status','updated')},
            status='PAUSED_DURING_EVALUATION',updated=time.time()))
        raise PauseRequested('Evaluation paused at an hour boundary; selected checkpoint retained')


def planned_seconds(update_seconds,cold_input_seconds,evaluation_seconds):
    return ((10944+410)*update_seconds+8760*cold_input_seconds+
            (4*10+4*2016/96)*evaluation_seconds)


@torch.no_grad()
def predictions(data,model,hours):
    model.eval(); rows=[]
    for hour in hours:
        check_evaluation_pause(data)
        ix,x,base,gate=data.batch(np.array([hour]))
        rows.append((base+data.delta(model,x,ix,gate)).cpu().numpy())
    return np.concatenate(rows)


@torch.no_grad()
def metrics(data,mean,hours,station_mask,*,members=16,seed=20261121):
    ix=data.indices(hours)
    valid=data.a['valid'][ix]&station_mask[None]
    result=point_metrics(data.a['truth'][ix],mean,valid)
    total=0.
    for start in range(0,len(hours),8):
        check_evaluation_pause(data)
        indices=ix[start:start+8]
        samples=torch.tensor(mean[start:start+8],device=data.device)[:,None]+data.noise(indices,members,seed)
        total+=float(energy_score(samples,data.t['truth'][indices],
            torch.as_tensor(valid[start:start+8],device=data.device)))*len(indices)
    return dict(**result,ES_mm=total/len(hours),members=members,seed=seed)


def make_engine(data,arm,fold,directory,*,engineering_hours=None,max_seconds=None):
    science,budget=scientific_settings(len(data.train_hours))
    hours=data.train_hours if engineering_hours is None else engineering_hours
    if engineering_hours is not None:
        budget=200
    torch.manual_seed(42)
    model=GridProcessor(arm).to(data.device)
    fit=data.folds!=fold
    if (data.a['valid'][data.indices(hours)][:,fit].sum(-1)<4).any():
        raise ValueError('Qualified hour has insufficient fitted B labels')
    dependencies=dict(data=sha256(data.root/'data.lock.json'),runner=sha256(__file__),
        engine=sha256(PROJECT/'src/gnss_field/training/engine.py'),arm=arm,fold=fold,
        objective='real fitted B observation ES K4, frozen shared Gaussian; same stochastic streams',
        output='full144x96 Z0 before public height query; no station output head')
    def telemetry(batch):
        pop=data.a['valid'][data.indices(batch)][:,fit]
        return dict(supervised_station_hours=int(pop.sum()),training_epoch_hours=len(data.train_hours))
    engine=TrainEngine(model,directory,hours,
        StageSpec('grid_processor_refine',budget,budget,8,1,42,dependencies),science,
        pause_paths=[data.root/'PAUSE'],max_seconds=max_seconds,batch_telemetry=telemetry)
    fit_t=torch.as_tensor(fit,device=data.device)
    def objective(batch,current,step):
        ix,x,base,gate=data.batch(batch)
        pred=base+data.delta(current,x,ix,gate)
        samples=pred[:,None]+data.noise(ix,4)
        loss=energy_score(samples,data.t['truth'][ix],data.t['valid'][ix]&fit_t[None])
        return dict(loss=loss,observation_ES_mm=loss)
    def evaluator(current,step):
        mean=predictions(data,current,data.inner_hours)
        m=metrics(data,mean,data.inner_hours,fit)
        return {'joint':dict(value=m['ES_mm'],rmse_mm=m['station_macro_rmse_mm'],
            mae_mm=m['station_macro_mae_mm'],bias_mm=m['station_macro_bias_mm'],
            population='B2018 inner 96 anchors, fitted groups; incremental development',
            n_station_hours=m['n_station_hours'],n_stations=m['n_stations'])}
    return engine,objective,evaluator


def verify_resume(data):
    root=data.root/'engineering_resume'
    signature=dict(runner=sha256(__file__),data=sha256(data.root/'data.lock.json'),
        device=torch.cuda.get_device_name() if str(data.device).startswith('cuda') else 'cpu',
        torch=torch.__version__)
    path=root/'verification.json'
    if path.exists():
        result=read_json(path)
        if result['signature']!=signature:
            raise ValueError('Engineering verification provenance changed; use a new run')
        return result
    design=read_json(data.runtime['design'])
    hours=np.asarray(design['core_hours'][:8],dtype='i8')
    data.prepare(data.indices(hours))
    before_rng=torch.get_rng_state().clone()
    data.frozen_m0(int(data.train_hours[0]))
    if not torch.equal(before_rng,torch.get_rng_state()):
        raise ValueError('Frozen inference changed training RNG')
    start=time.monotonic()
    with torch.random.fork_rng():
        torch.manual_seed(42)
        zero_digest=model_digest(GridProcessor('resunet69'))
    continuous,obj,_=make_engine(data,'resunet69',0,root/'continuous',engineering_hours=hours)
    result=continuous.run(obj,stop_after=200)
    if continuous.step!=200 or result['status']!='COMPLETE':
        raise PauseRequested('Continuous engineering path has not completed step200')
    expected=continuous.state()
    changed=zero_digest!=model_digest(continuous.model)
    del continuous; torch.cuda.empty_cache()
    split,obj,_=make_engine(data,'resunet69',0,root/'split',engineering_hours=hours)
    checkpoint100=root/'split_at_100.pt'
    if split.step<=100:
        result=split.run(obj,stop_after=100)
        if split.step!=100 or result['status']!='PAUSED':
            raise PauseRequested('Split engineering path has not reached step100')
        atomic_torch(checkpoint100,split.state())
    if not checkpoint100.exists() or torch.load(checkpoint100,map_location='cpu',weights_only=False)['step']!=100:
        raise ValueError('Exact engineering recovery requires the actual step100 state')
    del split; torch.cuda.empty_cache()
    resumed,obj,_=make_engine(data,'resunet69',0,root/'split',engineering_hours=hours)
    result=resumed.run(obj,stop_after=200)
    if resumed.step!=200 or result['status']!='COMPLETE':
        raise PauseRequested('Resumed engineering path has not completed step200')
    actual=resumed.state()
    for key in ('model','ema','optimizers','cursor','rng'):
        _same(expected[key],actual[key])
    if not changed:
        raise ValueError('Engineering path did not update actual weights')
    result=dict(passed=True,bitwise=True,signature=signature,
        actual_updates=400,continuous=200,resume=[100,100],weights_changed=changed,
        elapsed_seconds=time.monotonic()-start,step100_sha256=sha256(checkpoint100),zero_digest=zero_digest)
    write_json(path,result)
    print('REAL_GRID_RESUME_PASS',result,flush=True)
    return result


def cached_benchmark(data,contract):
    path=data.root/'benchmark.json'
    if not path.exists(): return None
    result=read_json(path)
    if result.get('contract')!=contract or result.get('updates')!=10:
        raise ValueError('Benchmark provenance differs; refusing to overwrite original budget evidence')
    return result


def benchmark(data):
    contract=fingerprint(dict(data=sha256(data.root/'data.lock.json'),runner=sha256(__file__),
        device=torch.cuda.get_device_name(),torch=torch.__version__))
    existing=cached_benchmark(data,contract)
    if existing is not None: return existing
    cold=[i for i in data.indices(data.train_hours) if not data.done[i]][:4]
    if len(cold)!=4:
        raise ValueError('Fresh cold-input benchmark requires four uncommitted hours')
    # Warm the frozen network itself, then measure actual cold inputs once, not twice.
    data.frozen_m0(int(data.a['hours'][cold[0]]))
    t=time.monotonic(); profiles=data.prepare(cold); cold_total=(time.monotonic()-t)/len(cold)
    sample=data.train_hours[:8]; data.prepare(data.indices(sample))
    attempt=0
    while (data.root/f'benchmark_attempt_{attempt:02d}').exists(): attempt+=1
    engine,obj,_=make_engine(data,'resunet69',0,data.root/f'benchmark_attempt_{attempt:02d}',
                            engineering_hours=sample,max_seconds=300)
    engine.run(obj,stop_after=2)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start=time.monotonic()
    engine.run(obj,stop_after=12)
    if engine.step!=12: raise PauseRequested('Benchmark timed updates interrupted')
    torch.cuda.synchronize(); seconds=(time.monotonic()-start)/10
    data.prepare(data.indices(data.inner_hours))
    evaluation={}; evaluator=None
    for members in (16,64):
        start=time.monotonic()
        mean=predictions(data,engine.ema,data.inner_hours)
        evaluator=metrics(data,mean,data.inner_hours,data.folds!=0,members=members)
        torch.cuda.synchronize(); evaluation[str(members)]=time.monotonic()-start
    estimated=(10944+400)*seconds+8760*cold_total+40*evaluation['16']+4*2016/96*evaluation['64']
    result=dict(contract=contract,device=torch.cuda.get_device_name(),updates=10,warmup_updates=2,update_seconds=seconds,
        cold_m0_seconds=float(np.mean([p['m0_seconds'] for p in profiles])),
        feature_seconds_per_hour=float(np.mean([p['feature_seconds'] for p in profiles])),cold_input_seconds=cold_total,
        evaluation_seconds=evaluation,inner=evaluator,
        peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,
        estimated_queue_seconds=estimated,with_margin_seconds=1.5*estimated,
        card_hours_with_margin=1.5*estimated/3600,updated=time.time())
    immutable_json(data.root/'benchmark.json',result)
    print('BENCHMARK',result,flush=True)
    return result


def remaining_estimate(data,b):
    training=inner=outer=0
    for fold in (0,1):
        for arm in ARMS:
            root=data.root/f'fold_{fold}'/arm
            if (root/'results.json').exists(): continue
            state=read_json(root/'status.json') if (root/'status.json').exists() else {}
            step=state.get('step',0)
            training+=max(0,2736-step) if state.get('status')!='COMPLETE' else 0
            inner+=(not (root/'initial_metrics.json').exists())+(step<200)
            inner+=sum(s>step for s in range(342,2737,342))
            inner+=bool(state.get('pending_evaluation'))
            outer+=1
    engineering=0
    if not (data.root/'engineering_resume/verification.json').exists():
        for name in ('continuous','split'):
            p=data.root/f'engineering_resume/{name}/status.json'
            engineering+=200-(read_json(p)['step'] if p.exists() else 0)
    needed=data.indices(np.unique(np.r_[data.train_hours,data.inner_hours,data.outer_hours]))
    missing=int((~data.done[needed]).sum())
    return 1.5*((training+engineering)*b['update_seconds']+missing*b['cold_input_seconds']+
        inner*b['evaluation_seconds']['16']+outer*2016/96*b['evaluation_seconds']['64'])


def train_one(data,arm,fold,*,max_seconds=None):
    directory=data.root/f'fold_{fold}'/arm
    engine,obj,evaluator=make_engine(data,arm,fold,directory,max_seconds=max_seconds)
    initialize_zero(engine,evaluator)
    result=engine.run(obj,evaluator=evaluator)
    if result['status']!='COMPLETE':
        write_json(data.root/'queue_status.json',dict(status=result['status'],arm=arm,fold=fold,
            step=engine.step,updated=time.time()))
        return False
    selected=min(engine.products['joint'],key=lambda r:(r['value'],r['step']))
    immutable_json(directory/'selection.lock.json',dict(selected=selected,
        sha256=sha256(selected['checkpoint']),rule='inner ES K16, EMA, zero retained; no outer selection'))
    engine.ema.load_state_dict(torch.load(selected['checkpoint'],map_location=data.device,weights_only=False)['ema'])
    # Publish selection before accessing outer-label scores.
    write_json(data.root/'queue_status.json',dict(status='OUTER_EVALUATION',arm=arm,fold=fold,updated=time.time()))
    mean=predictions(data,engine.ema,data.outer_hours)
    score=metrics(data,mean,data.outer_hours,data.folds==fold,members=64,seed=20261122)
    from local_probe_diagnostics import save_npz
    ix=data.indices(data.outer_hours)
    save_npz(directory/'outer_predictions.npz',hours=data.outer_hours,mean=mean,
        truth=data.a['truth'][ix],population=data.a['valid'][ix]&(data.folds==fold)[None])
    write_json(directory/'results.json',dict(arm=arm,fold=fold,selected=selected,outer_B=score,
        population='B2018 full held-out weather weeks, historical foundation exposure',
        scientific_updates=engine.step,source_model=sha256(selected['checkpoint'])))
    del engine; torch.cuda.empty_cache()
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=RUN)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--mode',choices=['engineering','benchmark','train','server-auto'],default='train')
    p.add_argument('--max-seconds',type=float,default=14400)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--fold',type=int,choices=[0,1])
    a=p.parse_args()
    configure(type('Runtime',(),{'cpus':2})(),42)
    if a.mode=='server-auto':
        from grid_pair_release import verify_grid_release
        verify_grid_release(PROJECT)
    data=GridStudy(a.root,runtime=read_json(a.runtime) if a.runtime else None)
    started=time.monotonic()
    data.deadline=started+a.max_seconds
    def on_signal(*_):
        (data.root/'PAUSE').touch()
    for sig in ([signal.SIGUSR1] if hasattr(signal,'SIGUSR1') else [])+[signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,on_signal)
    if a.mode=='engineering':
        verify_resume(data); return
    if a.mode=='benchmark':
        benchmark(data); return
    if a.mode=='server-auto':
        if 'A800' not in torch.cuda.get_device_name():
            raise ValueError('Formal server queue requires one allocated A800')
        write_json(data.root/'queue_status.json',dict(status='BENCHMARK',updated=time.time()))
        b=benchmark(data)
        if remaining_estimate(data,b)>a.max_seconds-(time.monotonic()-started):
            write_json(data.root/'queue_status.json',dict(status='BUDGET_GATE_STOP',benchmark=b))
            raise ValueError('Measured queue estimate exceeds the finite four-card-hour cap')
        immutable_json(data.root/'budget.lock.json',dict(total_allocation_cap_seconds=14400,
            benchmark_sha256=sha256(data.root/'benchmark.json'),max_parallel_gpus=1,
            maximum_scientific_updates=10944,EDM=False,GAN=False))
    write_json(data.root/'queue_status.json',dict(status='ENGINEERING_RECOVERY_CHECK',updated=time.time()))
    verify_resume(data)
    immutable_json(data.root/'training.lock.json',dict(data=sha256(data.root/'data.lock.json'),
        runner=sha256(__file__),arms=ARMS,folds=2,updates_per_arm_per_fold=2736,
        batch_hours=8,micro_hours=1,epochs=4,seed=42,selection='inner_ES_EMA_plus_zero',
        startup_step=200,startup_selectable=False,EDM=False,GAN=False,C2019=False,D2020=False))
    for fold in ([a.fold] if a.fold is not None else [0,1]):
        for arm in ([a.arm] if a.arm else ARMS):
            if (data.root/f'fold_{fold}'/arm/'results.json').exists():
                continue
            remaining=a.max_seconds-(time.monotonic()-started)
            write_json(data.root/'queue_status.json',dict(status='RUNNING',arm=arm,fold=fold,updated=time.time()))
            if not train_one(data,arm,fold,max_seconds=max(0,remaining)):
                return
    write_json(data.root/'queue_status.json',dict(status='COMPLETE',updated=time.time(),
        C2019=False,D2020=False,EDM=False,GAN=False))
    print('GRID_PROCESSOR_QUEUE_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except PauseRequested as exc:
        print('EVALUATION_PAUSED',str(exc),flush=True)
