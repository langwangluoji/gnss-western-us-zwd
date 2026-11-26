"""Finite observation-supervised stochastic/DF/GAN campaign, shared frozen mean."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,copy,time,signal
import numpy as np
import torch
from grid_generation_data import GenerationData
from grid_generation_metrics import score_ensemble
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256,model_digest
from gnss_field.contracts import ScienceConfig
from gnss_field.models.grid_generation import GenerationSystem,critic_gradient_penalty
from gnss_field.objectives.scores import energy_score
from gnss_field.training.engine import TrainEngine,StageSpec,PauseRequested,atomic_torch,capture_rng,restore_rng
from gnss_field.training.observable import _same
from gnss_field.runtime import configure

PROJECT=Path(__file__).resolve().parents[1]
ARMS=('stochastic','diffusion','gan')
_ACTIVE_ROOT=None


def make_engine(data,kind,fold,directory,engineering_hours=None,max_seconds=None,widths=None):
    cfg=read_json(PROJECT/'configs/grid_generation_v02.json')
    hours=data.train_hours if engineering_hours is None else engineering_hours
    if cfg['updates_per_arm_fold']!=684 or cfg['batch_hours']!=8 or cfg['micro_hours']!=8 or cfg['members_train']!=4 or cfg['heun_steps']!=18:
        raise ValueError('Configuration differs from the finite pilot sampling/budget contract')
    budget=cfg['updates_per_arm_fold'] if engineering_hours is None else 200
    science=ScienceConfig(learning_rate=cfg['learning_rate'],refinement_lr=cfg['learning_rate'],ema=cfg['ema'],warmup=cfg['warmup'],
        weight_decay=cfg['weight_decay'],gradient_clip=cfg['gradient_clip'],minimum_lr=cfg['minimum_lr'],
        startup_evaluation=200,evaluate_every=342,log_every=100,minimum_refinement_steps=684,
        train_members=4,screen_members=16,budgets={'generation_refine':budget})
    torch.manual_seed(42)
    system=GenerationSystem(kind,widths=tuple(cfg['widths']) if widths is None else widths,scale_mm=cfg['residual_scale_mm'],steps=cfg['heun_steps'],recompute=cfg['gradient_recomputation']).to(data.device)
    opt=torch.optim.AdamW(system.generator.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    auxiliary=[]
    if kind=='gan': auxiliary=[torch.optim.Adam(system.critic.parameters(),lr=1e-4,betas=(0.,.9))]
    engine=TrainEngine(system,directory,hours,StageSpec('generation_refine',budget,budget,cfg['batch_hours'],cfg['micro_hours'],cfg['seed'],
        dict(data=sha256(data.root/'data.lock.json'),runner=sha256(__file__),model=sha256(PROJECT/'src/gnss_field/models/grid_generation.py'),
        objective='ES K4; GAN adds WGAN-GP observation critic',kind=kind,fold=fold)),science,
        optimizers=[opt],clip_groups=[list(system.generator.parameters())],auxiliary_optimizers=auxiliary,
        pause_paths=[data.root/'PAUSE'],max_seconds=max_seconds,
        batch_telemetry=lambda h:dict(supervised_station_hours=int(data.a['valid'][data.source.indices(h)][:,data.folds!=fold].sum())))
    fit=torch.tensor(data.folds!=fold,device=data.device)

    def objective(hours,current,step):
        result=data.sample(current,hours,fold,4)
        ix=result['indices']; truth=data.source.t['truth'][ix]; mask=data.source.t['valid'][ix]&fit[None]
        es=energy_score(result['samples'],truth,mask); adversarial=es.new_zeros(())
        if kind=='gan':
            _,x,_,_,_=data.batch(hours,fold); conditions=data.local_conditions(x)
            for p in current.critic.parameters(): p.requires_grad_(False)
            normalized=(result['samples']-result['base'][:,None])/10.
            b,k,n=normalized.shape
            adversarial=-current.critic(normalized.flatten(0,1),data.coords,
                conditions[:,None].expand(-1,k,-1,-1).flatten(0,1),mask[:,None].expand(-1,k,-1).flatten(0,1)).mean()
            for p in current.critic.parameters(): p.requires_grad_(True)
        return dict(loss=es+.1*adversarial,observation_ES_mm=es,adversarial=adversarial)

    def before_update(hours,current,step):
        if kind!='gan': return
        opt=auxiliary[0]; opt.zero_grad(set_to_none=True)
        devices=[torch.cuda.current_device()] if str(data.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(420000+step)
            for hour in hours:
                h=np.array([hour]); ix,x,base,_,_=data.batch(h,fold)
                mask=data.source.t['valid'][ix]&fit[None]
                conditions=data.local_conditions(x)
                with torch.no_grad(): fake=(data.sample(current,h,fold,1)['samples'][:,0]-base)/10.
                real=torch.where(mask,(data.source.t['truth'][ix]-base)/10.,0.)
                loss=(current.critic(fake,data.coords,conditions,mask).mean()-current.critic(real,data.coords,conditions,mask).mean()+
                    10.*critic_gradient_penalty(current.critic,real,fake,data.coords,conditions,mask))
                (loss/len(hours)).backward()
        torch.nn.utils.clip_grad_norm_(current.critic.parameters(),1.,error_if_nonfinite=True); opt.step()

    def evaluator(current,step):
        m,_=evaluate(data,current,data.inner_hours,fold,16,20261123,outer=False)
        common=dict(rmse_mm=m['station_macro_rmse_mm'],mae_mm=m['station_macro_mae_mm'],bias_mm=m['station_macro_bias_mm'],
            CRPS_mm=m['CRPS_mm'],coverage90=m['coverage90'],width90_mm=m['width90_mm'],group_VS=m['group_VS'],
            mean_mc_se_mm=m['mean_mc_se_mm'],n_stations=m['n_stations'],n_station_hours=m['n_station_hours'],
            population='B2018 inner96, fitted station groups; incremental development')
        return {'point':dict(value=m['station_macro_rmse_mm'],**common),
                'joint':dict(value=m['group_ES_mm'],**common),'marginal':dict(value=m['CRPS_mm'],**common)}
    return engine,objective,before_update,evaluator


@torch.no_grad()
def evaluate(data,system,hours,fold,members,seed,outer=False):
    samples=[]; means=[]; deltas=[]
    for hour in hours:
        data.check(force=True); row=data.sample(system,np.array([hour]),fold,members,seed,member_chunk=members)
        samples.append(row['samples'].cpu().numpy()); means.append(row['mean'].cpu().numpy()); deltas.append(row['delta'].cpu().numpy())
    samples=np.concatenate(samples); mean=np.concatenate(means); delta=np.concatenate(deltas)
    ix=data.source.indices(hours); mask=data.a['valid'][ix]&((data.folds==fold) if outer else (data.folds!=fold))[None]
    truth=data.a['truth'][ix]
    score=score_ensemble(truth,samples,mean,mask,data.xy)
    se=delta.std(1,ddof=1)/np.sqrt(members)
    score.update(mean_mc_se_mm=float(np.mean(se[mask])),generated_shift_rms_mm=float(np.sqrt(np.mean(delta.mean(1)[mask]**2))),
        point_mean_estimator='frozen mean + MC mean of learned correction; known-zero prior control variate',hours=len(hours))
    return score,dict(hours=np.asarray(hours),station_ids=data.ids,samples=samples,mean=mean,truth=truth,population=mask,mean_mc_se=se)


def initialize(engine,evaluator):
    path=engine.directory/'initial.pt'; metrics=engine.directory/'initial_metrics.json'
    if engine.step and not path.exists(): raise ValueError('Missing original Gaussian candidate')
    if not path.exists(): atomic_torch(path,engine.state())
    if not metrics.exists():
        model=copy.deepcopy(engine.ema)
        model.load_state_dict(torch.load(path,map_location=next(model.parameters()).device,weights_only=False)['ema'])
        state=capture_rng()
        try: result=evaluator(model,0)
        finally: restore_rng(state)
        write_json(metrics,result)
    if engine.step==0:
        engine.products={k:[dict(**v,step=0,checkpoint=str(path.resolve()))] for k,v in read_json(metrics).items()}
        engine.save()


def verify_resume(data):
    report=data.root/'engineering/verification.json'
    signature=dict(data=sha256(data.root/'data.lock.json'),model=sha256(PROJECT/'src/gnss_field/models/grid_generation.py'),runner=sha256(__file__),
        torch=torch.__version__,device=torch.cuda.get_device_name())
    if report.exists():
        result=read_json(report)
        if result['signature']!=signature: raise ValueError('Engineering provenance changed')
        return result
    records={}; hours=data.train_hours[:8]
    for kind in ARMS:
        continuous,obj,before,_=make_engine(data,kind,0,data.root/f'engineering/{kind}/continuous',engineering_hours=hours)
        result=continuous.run(obj,before_update=before,stop_after=200)
        if continuous.step!=200 or result['status']!='COMPLETE': raise PauseRequested('Engineering continuous incomplete')
        expected=copy.deepcopy(continuous.state()); del continuous; torch.cuda.empty_cache()
        split,obj,before,_=make_engine(data,kind,0,data.root/f'engineering/{kind}/split',engineering_hours=hours)
        at100=data.root/f'engineering/{kind}/split_at100.pt'
        if split.step<=100:
            result=split.run(obj,before_update=before,stop_after=100)
            if split.step!=100: raise PauseRequested('Engineering split incomplete')
            atomic_torch(at100,split.state())
        if not at100.exists() or torch.load(at100,map_location='cpu',weights_only=False)['step']!=100: raise ValueError('Not actual100-step restore')
        del split; torch.cuda.empty_cache()
        split,obj,before,_=make_engine(data,kind,0,data.root/f'engineering/{kind}/split',engineering_hours=hours)
        result=split.run(obj,before_update=before,stop_after=200)
        if split.step!=200: raise PauseRequested('Engineering restored path incomplete')
        actual=split.state()
        keys=('model','ema','optimizers','auxiliary_optimizers','cursor','rng','step','products','pending_evaluation')
        for key in keys:
            _same(expected[key],actual[key])
        records[kind]=dict(continuous=200,resumed=[100,100],bitwise=True,split100_sha256=sha256(at100))
        del split,expected,actual; torch.cuda.empty_cache()
    result=dict(passed=True,signature=signature,records=records,engineering_updates=1200)
    write_json(report,result); return result


def measure_updates(engine,objective,before,synchronize=None,clock=None):
    synchronize=torch.cuda.synchronize if synchronize is None else synchronize
    clock=time.monotonic if clock is None else clock
    if hasattr(engine,'spec') and engine.step+5>engine.spec.budget:
        raise ValueError('Unmeasured benchmark checkpoint too near its finite budget')
    warm_target=engine.step+2
    engine.run(objective,before_update=before,stop_after=warm_target)
    if engine.step!=warm_target: raise PauseRequested('Benchmark warmup interrupted')
    first=engine.step; synchronize(); started=clock()
    engine.run(objective,before_update=before,stop_after=first+3); synchronize()
    delta=engine.step-first; elapsed=clock()-started
    if delta!=3 or elapsed<=0: raise PauseRequested('Benchmark did not measure three new updates')
    return dict(update_seconds=elapsed/delta,updates_measured=delta,start_step=first,end_step=engine.step)

def benchmark(data):
    path=data.root/'benchmark.json'
    contract=dict(data=sha256(data.root/'data.lock.json'),runner=sha256(__file__),device=torch.cuda.get_device_name())
    if path.exists():
        result=read_json(path)
        if result['contract']!=contract: raise ValueError('Benchmark provenance changed')
        return result
    times={}; hours=data.train_hours[:8]
    for kind in ARMS:
        engine,obj,before,_=make_engine(data,kind,0,data.root/f'benchmark/{kind}',engineering_hours=hours)
        timing_path=data.root/f'benchmark/{kind}/update_timing.json'
        if timing_path.exists():
            timing=read_json(timing_path)
            if timing['contract']!=contract: raise ValueError('Per-arm benchmark provenance changed')
        else:
            timing=dict(contract=contract,**measure_updates(engine,obj,before))
            immutable_json(timing_path,timing)
        update=timing['update_seconds']
        eval_times={}
        for k in (16,32):
            torch.cuda.synchronize(); start=time.monotonic()
            evaluate(data,engine.ema,data.inner_hours[:2],0,k,20261123)
            torch.cuda.synchronize(); eval_times[str(k)]=(time.monotonic()-start)/2
        times[kind]=dict(update_seconds=update,updates_measured=timing['updates_measured'],evaluation_seconds_per_hour=eval_times,
            peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,forward_calls_per_member=35 if kind=='diffusion' else 1,
            training_recompute=kind=='diffusion' and engine.model.generator.recompute)
        del engine; torch.cuda.empty_cache()
    estimate=0.
    for kind,row in times.items():
        estimate+=(2*684+400)*row['update_seconds']
        # initial, startup, two regular evaluations; at most two selected products.
        estimate+=2*(4*96*row['evaluation_seconds_per_hour']['16']+2*96*row['evaluation_seconds_per_hour']['32'])
    result=dict(contract=contract,timings=times,estimate_seconds=estimate,with_margin_seconds=1.5*estimate,engineering_updates=1200)
    immutable_json(path,result); print('GENERATION_BENCHMARK',result,flush=True); return result


def remaining_estimate(data,benchmark_result):
    total=0.
    for kind,row in benchmark_result['timings'].items():
        engineering=0
        if not (data.root/'engineering/verification.json').exists():
            for name in ('continuous','split'):
                path=data.root/f'engineering/{kind}/{name}/status.json'
                engineering+=max(0,200-(read_json(path)['step'] if path.exists() else 0))
        total+=engineering*row['update_seconds']
        for fold in (0,1):
            directory=data.root/f'fold_{fold}/{kind}'
            if (directory/'results.json').exists(): continue
            state=read_json(directory/'status.json') if (directory/'status.json').exists() else {}
            total+=max(0,684-state.get('step',0))*row['update_seconds']
            evaluated=set()
            if (directory/'development.jsonl').exists():
                import json
                evaluated={json.loads(x)['step'] for x in (directory/'development.jsonl').read_text().splitlines() if x.strip()}
            count=int(not (directory/'initial_metrics.json').exists())+sum(i not in evaluated for i in (200,342,684))
            total+=count*96*row['evaluation_seconds_per_hour']['16']
            total+=sum(not (directory/f'{p}_outer.json').exists() for p in ('point','joint'))*96*row['evaluation_seconds_per_hour']['32']
    total+=2*96*benchmark_result['timings']['stochastic']['evaluation_seconds_per_hour']['32']
    return 1.5*total

@torch.no_grad()
def export_case(data,system,fold,path):
    hour=data.outer_hours[:1]
    result=data.sample(system,hour,fold,32,20261124,member_chunk=32,return_fields=True)
    fields=data.mean_grid(hour,fold)[:,None]+result['fields']
    from gnss_field.models.grid_processor import query_field
    actual=query_field(fields.flatten(0,1),data.source.qindices,data.source.qweights).reshape(1,32,-1)
    if (actual+result['observation_noise']-result['samples']).abs().max()>.001:
        raise ValueError('Published-field requery differs from observation forecast')
    from gnss_field.data import ContextBuilder
    support=ContextBuilder(data.source.store).support(int(hour[0]))[0]
    f=fields[0,:,0].cpu().numpy()
    np.savez_compressed(path,hour=hour,mean_Z0=f.mean(0),std_Z0=f.std(0,ddof=1),members_Z0=f[:4],
        x=data.source.store.grid['x'],y=data.source.store.grid['y'],land=data.source.store.grid['region'],support=support,
        interpretation='predeclared first outer pilot hour; field noise excludes observation tau; 25km output spacing')

def train_one(data,kind,fold,max_seconds):
    directory=data.root/f'fold_{fold}/{kind}'
    engine,obj,before,evaluator=make_engine(data,kind,fold,directory,max_seconds=max_seconds)
    initialize(engine,evaluator)
    if data.deadline: engine.max_seconds=max(0,data.deadline-time.monotonic())
    result=engine.run(obj,evaluator=evaluator,before_update=before)
    if result['status']!='COMPLETE': return False
    selections={p:min(engine.products[p],key=lambda x:(x['value'],x['step'])) for p in ('point','joint')}
    lock=dict(products={k:dict(**v,sha256=sha256(v['checkpoint'])) for k,v in selections.items()},
        rule='inner K16 pilot; outer never selects',mean_mode='frozen_m0_idw',source_data_sha256=sha256(data.source.root/'data.lock.json'))
    immutable_json(directory/'selection.lock.json',lock)
    completed={}
    for product,selected in selections.items():
        output=directory/f'{product}_outer.json'
        if output.exists(): completed[product]=read_json(output); continue
        engine.ema.load_state_dict(torch.load(selected['checkpoint'],map_location=data.device,weights_only=False)['ema'])
        write_json(data.root/'queue_status.json',dict(status='OUTER_EVALUATION',arm=kind,fold=fold,product=product,updated=time.time()))
        scores,arrays=evaluate(data,engine.ema,data.outer_hours,fold,32,20261124,outer=True)
        np.savez_compressed(directory/f'{product}_outer_predictions.npz',**arrays)
        export_case(data,engine.ema,fold,directory/f'{product}_field_case.npz')
        record=dict(product=product,selected=selected,metrics=scores,cohort='B2018 preregistered96 hours within twelve outer weeks; incremental development')
        write_json(output,record); completed[product]=record
    write_json(directory/'results.json',dict(kind=kind,fold=fold,scientific_updates=engine.step,products=completed))
    del engine; torch.cuda.empty_cache(); return True


def main():
    global _ACTIVE_ROOT
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--mode',choices=('benchmark','engineering','train','server-auto'),default='server-auto'); p.add_argument('--max-seconds',type=float,default=14040)
    a=p.parse_args(); _ACTIVE_ROOT=a.root
    if a.mode=='server-auto':
        from grid_pair_release import verify_grid_release
        verify_grid_release(PROJECT)
    configure(type('Runtime',(),{'cpus':2})(),42)
    started=time.monotonic(); data=GenerationData(a.root,read_json(a.runtime)); data.deadline=started+a.max_seconds
    def pause(*_): (data.root/'PAUSE').touch()
    for sig in [signal.SIGTERM,signal.SIGINT]+([signal.SIGUSR1] if hasattr(signal,'SIGUSR1') else []): signal.signal(sig,pause)
    if a.mode=='engineering': verify_resume(data); return
    if a.mode=='benchmark': benchmark(data); return
    if a.mode=='server-auto':
        if 'A800' not in torch.cuda.get_device_name(): raise ValueError('Formal training requires allocated single A800')
        write_json(data.root/'queue_status.json',dict(status='BENCHMARK',updated=time.time()))
        b=benchmark(data)
        if remaining_estimate(data,b)>data.deadline-time.monotonic():
            write_json(data.root/'queue_status.json',dict(status='BUDGET_GATE_STOP',benchmark=b)); raise ValueError('Pilot does not fit four-card-hour cap')
        immutable_json(data.root/'budget.lock.json',dict(cap_seconds=14400,benchmark_sha256=sha256(data.root/'benchmark.json'),max_parallel=1))
    write_json(data.root/'queue_status.json',dict(status='ENGINEERING_RECOVERY_CHECK',updated=time.time()))
    verify_resume(data)
    for fold in (0,1):
        baseline=data.root/f'fold_{fold}/gaussian_outer.json'
        if not baseline.exists():
            score,arrays=evaluate(data,None,data.outer_hours,fold,32,20261124,outer=True)
            baseline.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(baseline.with_suffix('.npz'),**arrays)
            write_json(baseline,score)
        for kind in ARMS:
            if (data.root/f'fold_{fold}/{kind}/results.json').exists(): continue
            write_json(data.root/'queue_status.json',dict(status='RUNNING',arm=kind,fold=fold,updated=time.time()))
            if not train_one(data,kind,fold,max(0,data.deadline-time.monotonic())):
                write_json(data.root/'queue_status.json',dict(status='PAUSED_OR_BUDGET',arm=kind,fold=fold,updated=time.time())); return
    from grid_generation_report import build_report
    build_report(data.root)
    write_json(data.root/'queue_status.json',dict(status='COMPLETE',updated=time.time(),C2019=False,D2020=False))
    print('GENERATION_QUEUE_COMPLETE',flush=True)


if __name__=='__main__':
    try: main()
    except PauseRequested as exc:
        if _ACTIVE_ROOT:
            path=_ACTIVE_ROOT/'queue_status.json'; prior=read_json(path) if path.exists() else {}
            write_json(path,dict(**{k:v for k,v in prior.items() if k not in ('status','updated')},status='PAUSED_DURING_EVALUATION',updated=time.time()))
        print('GENERATION_PAUSED',str(exc),flush=True)
    except Exception as exc:
        if _ACTIVE_ROOT:
            path=_ACTIVE_ROOT/'queue_status.json'; prior=read_json(path) if path.exists() else {}
            status=prior.get('status') if prior.get('status')=='BUDGET_GATE_STOP' else 'FAILED'
            write_json(path,dict(status=status,error=str(exc),updated=time.time()))
        raise
